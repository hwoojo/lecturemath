"""B0-first automation with preserved RGB, proposals and uncertain predictions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from boardnotes.v2.pipeline import recognize,detect,worker_call,sha
from boardnotes.v2.schema import save_document,validate_regions
from .layout import layout_to_items,fuse_regions
from .routing import route_candidates
from .splitting import propose_rows,assemble_rows
from .pdf import render_pdf
from .resources import run_bounded

ROOT=Path(__file__).resolve().parents[2]


def configuration(model_root,formula_model,threads=2,max_rss_mib=6000):
    root=Path(model_root)
    return {'formula_model':formula_model,'threads':threads,'language':'en','max_rss_mib':max_rss_mib,
            'det_dir':str((root/'PP-OCRv5_mobile_det_infer').resolve()),'rec_dir':str((root/'en_PP-OCRv5_mobile_rec_infer').resolve()),
            'formula_dir':str((root/(formula_model+'_infer')).resolve()),'layout_dir':str((root/'PP-DocLayout-S_infer').resolve())}


def obtain_b0(source,provided,output,v1_python,source_variant='rgb'):
    target=output/'b0.png'
    if source_variant=='b0':provided=source
    if provided:
        image=cv2.imread(str(provided),0);raw=cv2.imread(str(source))
        if image is None or raw is None or image.shape!=raw.shape[:2] or not np.isin(image,[0,255]).all():
            raise ValueError('B0 must be an aligned black-writing binary mask at full input coordinates')
        cv2.imwrite(str(target),image)
        report={'mode':'provided B0','source':str(Path(provided).resolve()),'source_sha256':sha(provided)}
    else:
        command=[str(v1_python),'-m','boardnotes.v25.binarize',str(Path(source).resolve()),'--output',str(target.resolve()),
                 '--checkpoint',str(ROOT/'models/LectureNet_model_BIN_V34_final.dat'),'--config',str(ROOT/'ACCESS2021_release/configs/FCN_LectureNet.conf')]
        import os
        with (output/'v34.log').open('w') as log:
            completed=run_bounded(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),cwd=ROOT,timeout=300,max_rss_mib=6000)
        if completed.returncode or not target.is_file():raise RuntimeError(f'V34 failed; see {output}/v34.log')
        report={'mode':'separate V34 subprocess',**json.loads(target.with_suffix('.json').read_text())}
    return target,report


def analyze_regions(source,b0,output,ocr_python,config,timeout=600):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(b0));h,w=image.shape[:2]
    ocr=detect(b0,output/'ocr',ocr_python,config,timeout=timeout)
    layout=worker_call(ocr_python,'layout',[{'id':'page','image':str(Path(b0).resolve())}],config,output/'layout',timeout)
    value=layout['items'][0]
    items=layout_to_items({'boxes':value.get('layout_boxes',[])})
    fused=fuse_regions(items,ocr['regions'],w,h)
    routed=route_candidates(fused)
    document={'version':'2.0','source_image':str(Path(source).resolve()),'width':w,'height':h,'timestamp':None,'regions':routed,
              'metadata':{'detection_input':'FCN-LectureNet V34 B0','layout_report':layout,'ocr_report':ocr['metadata']['worker'],
                          'fusion':'conservative provenance-preserving fusion','routing':'symbol/prose heuristic; no GT access',
                          'classification_accuracy':'unverified; all predicted types require review'}}
    save_document(document,output/'regions.json')
    return document


def process_image(source,output,ocr_python,config,b0=None,v1_python=sys.executable,regions=None,split_formulas=False,
                  structure_hints=None,timestamp=None,source_variant='rgb',cache_dir=None,font=None,timeout=600):
    source=Path(source);output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('output must be empty; retain prior predictions and use render/review for edits')
    image=cv2.imread(str(source))
    if image is None:raise ValueError('cannot decode source image')
    output.mkdir(parents=True,exist_ok=True);h,w=image.shape[:2]
    started=time.perf_counter()
    mask,b0_report=obtain_b0(source,b0,output,v1_python,source_variant)
    if regions is None:
        analysis=analyze_regions(source,mask,output/'analysis',ocr_python,config,timeout)
        regions=analysis['regions'];provenance={'mode':'automatic','regions_json':str((output/'analysis/regions.json').resolve())}
    else:
        validate_regions(regions,w,h);provenance={'mode':'explicit reviewed/oracle regions'}
    specs=[{'id':str(r['id']),'type':r['type'],'bbox':r['bbox']} for r in regions]
    hints=structure_hints or {};proposals={};children={}
    used_ids={spec['id'] for spec in specs}
    def unique_id(base):
        value=base;index=1
        while value in used_ids:
            value=base+f'_{index}';index+=1
        used_ids.add(value);return value
    binary=cv2.imread(str(mask),0)
    for r in regions:
        if r['type']!='formula':continue
        hint=hints.get(str(r['id']),r.get('structure_hint'))
        proposal=propose_rows(binary,r['bbox'],hint);proposals[str(r['id'])]=proposal
        if split_formulas:
            row_specs=[{'id':unique_id(str(r['id'])+f'__row{i}'),'type':'formula','bbox':row['bbox']} for i,row in enumerate(proposal['rows'])]
            children[str(r['id'])]=[s['id'] for s in row_specs];specs+=row_specs
    docs=recognize(source,specs,output/'recognition',ocr_python,config,variants={'b0':mask},timestamp=timestamp,
                   cache_dir=cache_dir,timeout=timeout)
    document=docs['b0'];byid={r['id']:r for r in document['regions']}
    original={str(r['id']):r for r in regions}
    for item in document['regions']:
        if item['id'] in original:
            proposal=original[item['id']]
            item['detection']={k:v for k,v in proposal.items() if k not in {'content','latex','error','crop_image'}}
            # A successful recognizer cannot resolve an uncertain region type.
            item['review_required']=bool(item.get('review_required') or proposal.get('review_required')
                                         or proposal.get('status')=='needs_review')
        else:item['role']='row recognition candidate; whole-region prediction also retained'
    assemblies={}
    for parent,ids in children.items():
        row_values=[byid[r].get('latex') or '' for r in ids]
        proposal=proposals[parent]
        assemblies[parent]=assemble_rows(row_values,hints.get(parent,original[parent].get('structure_hint'))) if proposal['assembly_allowed'] else {'status':'needs_review','latex':None,'error':'No explicit structure hint or split is unsafe','rows':row_values}
        if assemblies[parent].get('latex'):
            parent_region=byid[parent]
            document['regions'].append({'id':unique_id(parent+'__assembled'),'type':'formula','bbox':parent_region['bbox'],
                'status':'needs_review','content':None,'latex':assemblies[parent]['latex'],'confidence':None,'error':None,
                'model':config['formula_model']+' + explicit row assembly','review_required':True,
                'crop_image':parent_region['crop_image'],'whole_region_id':parent,'row_ids':ids})
    document['metadata'].update(version='2.5',b0=b0_report,source_variant=source_variant,region_origin=provenance,
        formula_model_selection='explicit CLI configuration; never ground truth or nonempty-output routing',
        split_proposals=proposals,row_assemblies=assemblies,image_process_seconds=time.perf_counter()-started)
    save_document(document,output/'result.json')
    pages=render_pdf(document,output/'result.pdf',font)
    rendering=json.loads((output/'result.rendering.json').read_text())
    report={'input':str(source.resolve()),'b0':b0_report,'source_variant':source_variant,'pages':pages,
            'region_count':len(regions),'recognition_count':len(document['regions']),
            'type_counts':{t:sum(r['type']==t for r in regions) for t in ['text','formula','other','unknown']},
            'failed_recognitions':sum(r['status']=='failed' for r in document['regions']),
            'elapsed_seconds':time.perf_counter()-started,'rendering':rendering}
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return document,report


def process_manifest(manifest,output,ocr_python,config,**kwargs):
    manifest=Path(manifest);data=json.loads(manifest.read_text())
    if data.get('format')!='boardnotes-v2-inputs-1':raise ValueError('expected V1 uncropped-keyframe export manifest')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('manifest output must be empty')
    output.mkdir(parents=True,exist_ok=True);pages=[]
    for i,page in enumerate(data['pages']):
        source=Path(page['image'])
        if not source.is_absolute():source=(manifest.parent/source).resolve()
        doc,report=process_image(source,output/f'page_{i:04}',ocr_python,config,timestamp=page['timestamp'],source_variant='b0',**kwargs)
        doc['metadata']['v1_handoff']=page;save_document(doc,output/f'page_{i:04}/result.json')
        pages.append(report)
    combined={'scope':'V1 temporal-composite handoff; single RGB correspondence not assumed','pages':pages}
    files=[output/f'page_{i:04}/result.pdf' for i in range(len(pages))]
    if files and all(path.is_file() for path in files):
        try:
            from pypdf import PdfReader,PdfWriter
            writer=PdfWriter()
            for path in files: writer.append(PdfReader(path))
            with (output/'result.pdf').open('wb') as stream: writer.write(stream)
            combined['combined_pdf']=str((output/'result.pdf').resolve())
        except Exception as error:
            combined['combined_pdf_error']=f'{type(error).__name__}: {error}'
    (output/'manifest-report.json').write_text(json.dumps(combined,indent=2)+'\n')
    return pages


def process_video(video,output,ocr_python,config,v1_python=sys.executable,max_width=1280,fps=1.,timeout=3600,**kwargs):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('video output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    import os
    started=time.perf_counter()
    command=[str(v1_python),'-m','boardnotes.cli',str(Path(video).resolve()),'--output-dir',str((output/'v1').resolve()),
             '--max-width',str(max_width),'--fps',str(fps),'--export-v2-inputs']
    with (output/'v1.log').open('w') as log:
        result=run_bounded(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),cwd=ROOT,
                           timeout=timeout,max_rss_mib=config.get('max_rss_mib',6000))
    if result.returncode:raise RuntimeError(f'V1 video stage failed: see {output}/v1.log')
    reports=process_manifest(output/'v1/v2_inputs/manifest.json',output/'v25',ocr_python,config,
                             v1_python=v1_python,timeout=timeout,**kwargs)
    summary={'source_video':str(Path(video).resolve()),'source_video_sha256':sha(video),
             'v1':json.loads((output/'v1/report.json').read_text()),'v25_pages':reports,
             'elapsed_seconds':time.perf_counter()-started,
             'notice':'Real or synthetic status depends on input provenance; this function never invents source timestamps/RGB correspondence.'}
    (output/'end-to-end-report.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
