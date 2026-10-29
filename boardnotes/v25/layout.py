"""Conservative fusion of document-layout and OCR proposals in source pixels.

This module does not claim document categories are correct on lecture boards.
Every model label remains provenance; conflicts remain reviewable, and the full
source image must be retained by the caller. No learned model is imported here.
"""
from copy import deepcopy
import json
import math
import re

TEXT_LABELS = {
    'text', 'title', 'doc_title', 'document_title', 'paragraph_title', 'abstract',
    'reference', 'references', 'reference_content', 'footnote', 'header', 'footer',
    'page_number', 'figure_title', 'figure_caption', 'table_title', 'table_caption',
    'sidebar_text', 'aside_text', 'caption', 'list', 'list_item', 'formula_number',
    'number', 'content', 'chart_title',
}
FORMULA_LABELS = {'formula', 'equation', 'display_formula', 'inline_formula'}
OTHER_LABELS = {'image', 'figure', 'chart', 'table', 'seal', 'header_image', 'footer_image'}


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _box(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('layout bbox must have four source-pixel coordinates')
    if not all(_finite_number(v) for v in value):
        raise ValueError('layout bbox coordinates must be finite numbers')
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        raise ValueError('layout bbox must have positive extent; reversed coordinates are not reordered')
    box = [max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
           max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2)))]
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('layout bbox has no pixels inside the source image')
    return box


def _score(value):
    if value is None:
        return None
    if not _finite_number(value) or not 0 <= value <= 1:
        raise ValueError('detection score must be finite [0,1] or null')
    return float(value)


def _kind(label):
    label = label.strip().lower().replace(' ', '_').replace('-', '_')
    if label in TEXT_LABELS:
        return 'text'
    if label in FORMULA_LABELS:
        return 'formula'
    if label in OTHER_LABELS:
        return 'other'
    return 'unknown'


def _area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def _intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def _iou(a, b):
    overlap = _intersection(a, b)
    return overlap / (_area(a)+_area(b)-overlap)


def _covers(parent, child):
    return _intersection(parent, child)/_area(child)


def _union(a, b):
    return [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]


def layout_to_items(result, model='PP-DocLayout-S'):
    """Parse PaddleOCR LayoutDetection result.json/res.boxes without guessing labels.

    SDK outputs `coordinate=[xmin,ymin,xmax,ymax]` in input coordinates. Dimensions
    and clipping are handled later by fuse_regions. Missing labels are rejected;
    mapping cls_id to a label requires the actual downloaded inference config.
    """
    data = result.json if hasattr(result, 'json') else result
    if callable(data):
        data = data()
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError('layout SDK result must be a mapping')
    data = data.get('res', data)
    boxes = data.get('boxes')
    if not isinstance(boxes, list):
        raise ValueError('layout SDK result requires a boxes list')
    items = []
    for box in boxes:
        label = box.get('label')
        if not isinstance(label, str) or not label.strip():
            raise ValueError('layout SDK box requires its actual label, not an inferred cls_id mapping')
        coordinates = box.get('coordinate')
        if hasattr(coordinates, 'tolist'):
            coordinates = coordinates.tolist()
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
            raise ValueError('layout coordinate must be [xmin,ymin,xmax,ymax]')
        if not all(_finite_number(v) for v in coordinates):
            raise ValueError('layout coordinate must be finite')
        items.append({'bbox':[float(v) for v in coordinates], 'label':label,
                      'score':_score(box.get('score')), 'model':model,
                      **({'cls_id':int(box['cls_id'])} if box.get('cls_id') is not None else {})})
    return items


def _ocr_math_evidence(text):
    """Find explicit symbolic evidence, not a text/formula correctness score.

    Comparisons/equality require a digit or a single-letter token. Integral,
    sum/product/root signs or their literal LaTeX commands are intrinsically
    non-prose evidence. Ordinary '+'/'-' alone deliberately do not trigger this
    guard. Mixed prose such as 'Let x=1' remains ambiguous rather than being
    declared either wholly prose or a formula.
    """
    if not isinstance(text,str) or not text.strip():
        return None
    comparisons=sorted(set(text)&set('=≠<>≤≥'))
    strong=sorted(set(text)&set('∫∑∏√'))
    commands=re.findall(r'\\(?:frac|int|sum|prod|sqrt)\b',text)
    digits=any(char.isdigit() for char in text)
    variables=re.findall(r'\b[^\W\d_]\b',text,flags=re.UNICODE)
    if not (strong or commands or (comparisons and (digits or variables))):
        return None
    return {'markers':comparisons+strong+commands,'contains_digit':digits,
            'single_letter_tokens':variables,
            'basis':'explicit symbolic OCR evidence only; not semantic or calibrated classification'}


def _ambiguous_container(region):
    return region.get('dominant_container_ambiguous') or region.get('ambiguous_ocr_container')


def _region(box, kind, source, confidence, subtype):
    return {'type':kind, 'bbox':box, 'status':'preserved' if kind == 'other' else 'needs_review',
            'content':None, 'latex':None, 'confidence':None, 'error':None,
            'review_required':True, 'suggested_type':kind, 'subtype':subtype,
            'detection_confidence':confidence, 'provenance':[source], 'conflicts':[],
            'embedded_ocr_candidates':[], 'reading_order_method':'y band then x; not semantic mathematical order'}


def _merge(existing, new, conflict=False):
    existing['bbox'] = _union(existing['bbox'], new['bbox'])
    existing['provenance'].extend(new['provenance'])
    if conflict:
        existing['conflicts'].append({'kind':'layout_type_conflict',
            'predicted_types':sorted({existing['type'],new['type']}),
            'notice':'Overlapping layout classes disagree; no class is treated as ground truth.'})
        existing['type']='unknown'
        existing['suggested_type']='unknown'
        existing['status']='needs_review'
        existing['subtype']='conflicting_layout_labels'
    scores = [p['score'] for p in existing['provenance'] if p.get('score') is not None]
    existing['detection_confidence'] = max(scores) if scores else None


def fuse_regions(layout_items, ocr_regions, width, height):
    """Fuse source-coordinate proposals without dropping tiny in-frame symbols.

    Near-duplicate layout boxes (IoU>=.85) collapse, preserving every original
    box/label in provenance. Strong type disagreement (IoU>=.5) forms an unknown
    review region. OCR boxes >=90% contained by one layout region are attached as
    evidence, not independently re-recognized. Ambiguous multi-parent OCR boxes
    and all unattached OCR candidates stay unknown. No minimum-area filter is used.
    Development guard: a layout image/figure/chart/table covering >=60% of the
    frame and >=2 distinct OCR boxes is ambiguous on boards. Its label becomes
    unknown, and its OCR children remain independently reviewable. This guard is
    not evidence of correct classification or a calibrated threshold. A native
    text/title container with explicit symbolic OCR evidence is also unknown,
    with independently retained OCR children; no text-to-formula guess is made.
    """
    if any(not isinstance(v,int) or isinstance(v,bool) or v<=0 for v in [width,height]):
        raise ValueError('source width/height must be positive integers')
    if not isinstance(layout_items,list) or not isinstance(ocr_regions,list):
        raise ValueError('layout/OCR proposals must be lists')
    regions=[]
    for index,item in enumerate(layout_items):
        label=item.get('label')
        if not isinstance(label,str) or not label.strip():
            raise ValueError('layout item requires a nonempty label')
        box=_box(item.get('bbox'),width,height);score=_score(item.get('score'))
        source={'source_kind':'layout','detector':item.get('model','PP-DocLayout-S'),'source_id':f'layout_{index}',
                'bbox':box.copy(),'raw_bbox':[float(v) for v in item['bbox']], 'label':label,
                'predicted_type':_kind(label),'score':score}
        if 'cls_id' in item: source['cls_id']=item['cls_id']
        new=_region(box,_kind(label),source,score,label)
        matches=[r for r in regions if _iou(r['bbox'],box)>=.85 or
                 (r['type']!=new['type'] and _iou(r['bbox'],box)>=.5)]
        if matches:
            target=matches[0]
            _merge(target,new,conflict=target['type']!=new['type'])
        else:
            regions.append(new)
    # A document detector can call the entire board an "image". Do not let
    # that domain failure swallow all OCR candidates into an unrecognized parent.
    ocr_boxes=[_box(item.get('bbox'),width,height) for item in ocr_regions]
    for region in regions:
        native_other=any(p.get('predicted_type')=='other' for p in region['provenance'])
        fraction=_area(region['bbox'])/(width*height)
        contained={tuple(box) for box in ocr_boxes if _covers(region['bbox'],box)>=.9}
        if native_other and fraction>=.6 and len(contained)>=2:
            region.update(type='unknown',suggested_type='unknown',status='needs_review',
                          subtype='dominant_layout_container',dominant_container_ambiguous=True)
            region['conflicts'].append({'kind':'dominant_layout_container',
                'page_area_fraction':fraction,'distinct_contained_ocr_boxes':len(contained),
                'notice':'Page-sized layout image/graphic is ambiguous on boards; OCR candidates are retained independently.'})
    # A broad native text/title box can also swallow embedded math. Clear
    # symbolic OCR evidence makes the container mixed/uncertain; do not relabel
    # it as formula, and retain individual children for the downstream router.
    for region in regions:
        native_text=any(p.get('predicted_type')=='text' for p in region['provenance'])
        if not native_text:
            continue
        evidence=[]
        for item,box in zip(ocr_regions,ocr_boxes):
            text=item.get('candidate_text',item.get('content'))
            marker=_ocr_math_evidence(text)
            if marker and _covers(region['bbox'],box)>=.9:
                evidence.append({'bbox':box.copy(),'candidate_text':text,**marker})
        if evidence:
            region.update(type='unknown',suggested_type='unknown',status='needs_review',
                          ambiguous_ocr_container=True,ocr_math_evidence=evidence)
            if not region.get('dominant_container_ambiguous'):
                region['subtype']='layout_text_ocr_math_conflict'
            region['conflicts'].append({'kind':'layout_label_vs_ocr_math_evidence',
                'evidence':deepcopy(evidence),
                'notice':'Native text/title and symbolic OCR evidence disagree; parent and children need review. No formula class is asserted.'})
    for index,item in enumerate(ocr_regions):
        box=_box(item.get('bbox'),width,height)
        score=_score(item.get('detection_confidence',item.get('confidence')))
        # OCR recognition confidence is evidence about text decoding, NOT formula classification.
        source={'source_kind':'ocr','detector':item.get('model') or 'PP-OCRv5 detector', 'source_id':str(item.get('id',f'ocr_{index}')),
                'bbox':box.copy(),'raw_bbox':[float(v) for v in item['bbox']],
                'label':'ocr_text_candidate','predicted_type':'unknown','score':score,
                'candidate_text':item.get('candidate_text',item.get('content'))}
        ambiguous_containers=[r for r in regions if _ambiguous_container(r)
                              and _covers(r['bbox'],box)>=.9]
        for container in ambiguous_containers:
            container['embedded_ocr_candidates'].append(deepcopy(source))
            container['provenance'].append(deepcopy(source))
        parents=[r for r in regions if not _ambiguous_container(r)
                 and any(p.get('source_kind')=='layout' for p in r['provenance'])
                 and _covers(r['bbox'],box)>=.9]
        if len(parents)==1:
            parents[0]['embedded_ocr_candidates'].append(deepcopy(source))
            parents[0]['provenance'].append(deepcopy(source))
            continue
        duplicate=next((r for r in regions if r['subtype']=='ocr_only_candidate' and _iou(r['bbox'],box)>=.9),None)
        if duplicate:
            duplicate['provenance'].append(source)
            continue
        candidate=_region(box,'unknown',source,score,'ocr_only_candidate')
        candidate['suggested_type']='text'
        candidate['candidate_text']=source['candidate_text']
        if ambiguous_containers:
            candidate['conflicts'].append({'kind':'ambiguous_layout_container_parent',
                'parent_boxes':[r['bbox'].copy() for r in ambiguous_containers],
                'parent_conflict_kinds':sorted({conflict['kind'] for r in ambiguous_containers for conflict in r['conflicts']}),
                'notice':'Candidate preserved independently because layout/OCR evidence makes its container ambiguous on boards.'})
        if parents:
            candidate['conflicts'].append({'kind':'multiple_layout_parents',
                'parent_boxes':[p['bbox'].copy() for p in parents],
                'notice':'OCR box cannot be assigned to one layout parent; retained for manual review.'})
        regions.append(candidate)
    regions.sort(key=lambda r:(r['bbox'][1]//20,r['bbox'][0],r['bbox'][1]))
    for index,region in enumerate(regions):
        region['id']=f'region_{index:04}'
        region['model']=' + '.join(dict.fromkeys(p['detector'] for p in region['provenance']))
    return regions
