"""Conservative row proposals, never a mathematical-layout classifier.

Proposals use image geometry only. The original image is never modified. A tall
side component can be excluded from the *analysis mask*, but this is recorded and
never silently treated as proof of a system of equations.
"""
import re

import cv2
import numpy as np

PROTECTED_HINTS = {'fraction', 'matrix', 'root', 'script'}


def _crop(image, source_bbox):
    if (not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4))):
        raise ValueError('input must be a nonempty uint8 grayscale/BGR image')
    if (not isinstance(source_bbox, (list, tuple)) or len(source_bbox) != 4
            or any(not isinstance(v, int) or isinstance(v, bool) for v in source_bbox)):
        raise ValueError('source_bbox must contain four integer original coordinates')
    x1, y1, x2, y2 = source_bbox
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError('invalid source_bbox')
    h, w = image.shape[:2]
    if (h, w) == (y2-y1, x2-x1):
        crop = image
    elif x2 <= w and y2 <= h:
        crop = image[y1:y2, x1:x2]
    else:
        raise ValueError('image must be the bbox crop or contain source_bbox')
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY if crop.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if np.isin(crop, [0, 255]).all():
        mask = crop == 0
        threshold = 'binary black-writing foreground'
    else:
        _, inverted = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        mask = inverted > 0
        threshold = 'Otsu dark-writing foreground; RGB polarity is not auto-detected'
    return mask, threshold


def _bands(active):
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    bands = []
    start = last = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != last+1:
            bands.append([start, last+1])
            start = value
        last = value
    bands.append([start, last+1])
    return bands


def propose_rows(binary_or_bgr, source_bbox, structure_hint=None):
    """Return JSON-compatible, half-open original-coordinate crop proposals.

    A full source raster or an already cropped raster matching bbox dimensions is
    accepted. Only explicit ``aligned``/``cases`` permits later assembly. The
    optional protective hints fraction/matrix/root/script prevent splitting.
    Even candidate rows require review: geometry is not semantic recognition.
    """
    if structure_hint not in {None, 'aligned', 'cases', *PROTECTED_HINTS}:
        raise ValueError('unknown structure_hint')
    mask, threshold = _crop(binary_or_bgr, source_bbox)
    height, width = mask.shape
    diagnostics = {'threshold': threshold, 'crop_shape': [height, width],
                   'source_bbox': list(source_bbox), 'structure_hint': structure_hint,
                   'original_preserved': True, 'excluded_analysis_components': [],
                   'warnings': ['Geometry cannot establish mathematical layout or correctness.']}

    def result(status, reason, rows=None, allowed=False):
        return {'status': status, 'reason': reason, 'rows': rows or [],
                'assembly_allowed': allowed, 'diagnostics': diagnostics}

    if structure_hint in PROTECTED_HINTS:
        return result('not_split', 'Explicit protected structure hint; keep the entire region')
    if not mask.any():
        return result('not_split', 'No writing foreground')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = [dict(label=i, x=int(s[0]), y=int(s[1]), w=int(s[2]), h=int(s[3]), area=int(s[4]))
                  for i, s in enumerate(stats) if i]
    diagnostics['component_count'] = count-1
    ys, xs = np.nonzero(mask)
    active_width = int(xs.max()-xs.min()+1)
    active_height = int(ys.max()-ys.min()+1)
    excluded = []
    for component in components:
        x, y, w, h = (component[k] for k in ['x', 'y', 'w', 'h'])
        side = 'left' if x <= width*.18 else 'right' if x+w >= width*.78 else None
        if side and h >= max(12, .65*active_height) and w <= max(3, .20*active_width) and h >= 2.5*w:
            excluded.append({**component, 'side': side,
                             'reason': 'Tall narrow side component; may be brace/bracket, not classified'})
    diagnostics['excluded_analysis_components'] = excluded
    if {c['side'] for c in excluded} == {'left', 'right'}:
        return result('needs_review', 'Paired side structures may be a matrix; automatic row division withheld')

    # A wide thin component with ink above and below may be a fraction/radical.
    remaining = [c for c in components if c['label'] not in {e['label'] for e in excluded}]
    typical_height = float(np.median([c['h'] for c in remaining if c['h'] >= 3])) if any(c['h'] >= 3 for c in remaining) else 1.
    diagnostics['median_component_height'] = typical_height
    for bar in remaining:
        if bar['w'] < .45*active_width or bar['h'] > max(3, .18*typical_height):
            continue
        others = [c for c in remaining if c['label'] != bar['label']]
        overlapping = [c for c in others if min(c['x']+c['w'],bar['x']+bar['w'])-max(c['x'],bar['x']) >= .25*min(c['w'],bar['w'])]
        above = any(c['y']+c['h'] <= bar['y'] for c in overlapping)
        below = any(c['y'] >= bar['y']+bar['h'] for c in overlapping)
        if above and below:
            diagnostics['protected_horizontal_component'] = bar
            return result('needs_review', 'Horizontal separator with ink above/below may be a fraction; keep whole region')

    analysis = mask.copy()
    for component in excluded:
        analysis[labels == component['label']] = False
    if not analysis.any():
        return result('not_split', 'Only side structures remain; no independent row content')
    bands = _bands(analysis.any(axis=1))
    # Near upper/lower marks must remain attached to the main text band.
    gap_limit = max(2, int(round(.30*typical_height)))
    merged = []
    for start, end in bands:
        if merged and start-merged[-1][1] <= gap_limit:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    diagnostics['projection_bands'] = bands
    diagnostics['merged_bands'] = merged
    diagnostics['maximum_merged_gap'] = gap_limit
    if len(merged) < 2:
        return result('not_split', 'No sufficiently separated row bands')
    if len(merged) > 8:
        return result('needs_review', 'Too many bands; not a simple row layout')
    heights = [end-start for start, end in merged]
    widths = []
    for start, end in merged:
        row_x = np.flatnonzero(analysis[start:end].any(axis=0))
        widths.append(int(row_x[-1]-row_x[0]+1))
    if min(heights) < .55*max(heights) or min(widths) < .35*max(widths):
        return result('needs_review', 'Small detached band may be an index, numerator, or annotation; keep whole region')
    x0, y0, _, _ = source_bbox
    rows = []
    for index, (start, end) in enumerate(merged):
        row_x = np.flatnonzero(analysis[start:end].any(axis=0))
        left, right = int(row_x[0]), int(row_x[-1]+1)
        # Padding stays inside the original crop. Excluded side strokes can fall
        # outside these candidate row boxes, but are retained by the whole ROI.
        left, right = max(0,left-3), min(width,right+3)
        top, bottom = max(0,start-3), min(height,end+3)
        rows.append({'id': f'row_{index:02}', 'bbox': [x0+left,y0+top,x0+right,y0+bottom],
                     'status': 'needs_review', 'reading_order': index})
    if excluded and structure_hint is None:
        diagnostics['warnings'].append('Candidate row crops exclude the recorded side component; retain and inspect whole-region prediction.')
        return result('needs_review', 'Side structure ignored only for analysis; manual layout hint required for assembly', rows)
    return result('candidate', 'Separated similarly sized row bands; not a semantic classification', rows,
                  allowed=structure_hint in {'aligned','cases'})


def assemble_rows(latex_rows, structure_hint=None):
    """Combine row predictions only with a user-supplied layout hint.

    No equality sign, alignment marker, or missing mathematical term is invented.
    ``cases`` explicitly introduces the requested left-brace environment; do not
    choose it from geometry, particularly when the original has a right brace.
    """
    if structure_hint not in {None, 'aligned', 'cases'}:
        raise ValueError('assembly hint must be aligned/cases or null')
    if not isinstance(latex_rows, list) or any(not isinstance(v, str) for v in latex_rows):
        raise ValueError('latex_rows must be a list of strings')
    if structure_hint is None:
        return {'status': 'needs_review', 'latex': None, 'error': 'Explicit aligned/cases structure hint is required',
                'rows': list(latex_rows), 'review_required': True}
    if len(latex_rows) < 2 or any(not row.strip() for row in latex_rows):
        return {'status': 'failed', 'latex': None, 'error': 'At least two nonempty row predictions are required',
                'rows': list(latex_rows), 'review_required': True}
    # Nested block environments are not safely composable by string joining.
    forbidden = re.compile(r'\\(?:begin|end)\s*\{|\\(?:left|right)(?:\s*\\?[{}.]|\s*\|)')
    if any(forbidden.search(row) for row in latex_rows):
        return {'status': 'needs_review', 'latex': None, 'error': 'Row already contains block/delimiter structure; keep independent predictions',
                'rows': list(latex_rows), 'review_required': True}
    latex = '\\begin{' + structure_hint + '}' + r'\\'.join(row.strip() for row in latex_rows) + '\\end{' + structure_hint + '}'
    return {'status': 'candidate', 'latex': latex, 'error': None, 'rows': list(latex_rows),
            'structure_hint': structure_hint, 'review_required': True,
            'notice': 'Explicit hint supplied; rows joined without adding mathematical relations. Whole-region prediction must also be retained.'}
