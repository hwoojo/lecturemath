"""Conservative structural diagnostics; never a general mathematical equivalence checker."""
import re
import cv2
import numpy as np
from .inputs import ink_box,printed_canvas
from experiments.v1_5.real_report import distance


def strip_outer(text):
    text=text.strip()
    while text.startswith('{') and text.endswith('}'):
        depth=0;whole=True
        for i,c in enumerate(text):
            if c=='{':depth+=1
            elif c=='}':depth-=1
            if depth==0 and i<len(text)-1:whole=False;break
        if not whole:break
        text=text[1:-1].strip()
    return text


def cell_token(text):
    return re.sub(r'\s+','',strip_outer(text))


def structure(latex):
    latex=latex or '';blocks=[]
    for m in re.finditer(r'\\begin\{(bmatrix|pmatrix|Bmatrix|matrix|array|vmatrix|Vmatrix)\}(.*?)\\end\{\1\}',latex,re.S):
        env,body=m.groups();spec=None
        if env=='array':
            specmatch=re.match(r'\s*\{([^{}]*)\}',body)
            if specmatch:spec=specmatch.group(1);body=body[specmatch.end():]
        # Nested environments are deliberately not interpreted as flat matrices.
        if '\\begin' in body:continue
        rows=[row.strip() for row in re.split(r'\\\\',body) if row.strip()]
        cells=[[cell_token(c) for c in row.split('&')] for row in rows]
        blocks.append({'environment':env,'array_spec':spec,'rows':len(cells),'columns_per_row':[len(r) for r in cells],'cells':cells})
    multiline=[]
    for m in re.finditer(r'\\begin\{(aligned|cases|gathered)\}(.*?)\\end\{\1\}',latex,re.S):
        multiline.append({'environment':m.group(1),'rows':[re.sub(r'\s+','',r.replace('&','')) for r in re.split(r'\\\\',m.group(2)) if r.strip()]})
    scripts=[]
    for m in re.finditer(r'([_^])(?:\{([^{}]*)\}|(\\[a-zA-Z]+|[^{}\s]))',latex):
        scripts.append({'marker':m.group(1),'value':cell_token(m.group(2) if m.group(2) is not None else m.group(3))})
    return {'matrices':blocks,'multiline':multiline,'scripts':scripts,
            'script_scope':'Ordered marker/value observations only; attachment bases require visual review.',
            'factorial_count':sum(t=='!' for t in re.findall(r'\\.|!',latex)),
            'fraction_count':len(re.findall(r'\\(?:dfrac|tfrac|frac)\b',latex)),
            'equals_count':latex.count('='),'notice':'Lexical/structural observations, not semantic equivalence.'}


def compare(reference,prediction,status='ok'):
    expected=re.sub(r'\s+','',reference);actual=re.sub(r'\s+','',prediction or '')
    ref,pred=structure(reference),structure(prediction)
    def shapes(s):return [(m['rows'],m['columns_per_row']) for m in s['matrices']]
    matrices=bool(ref['matrices'])
    expected_lines=[m['rows'] for m in ref['multiline']]
    actual_lines=[m['rows'] for m in pred['multiline']]
    # array{r} is also a possible multiline presentation, not always a matrix.
    if expected_lines and not actual_lines and len(pred['matrices'])==1:
        block=pred['matrices'][0]
        if block['environment']=='array' and all(c==1 for c in block['columns_per_row']):
            actual_lines=[[row[0] for row in block['cells']]]
    return {'strict_exact_match':status=='ok' and expected==actual,'edit_distance':distance(expected,actual),
            'normalized_edit_distance':distance(expected,actual)/max(1,len(expected)),
            'reference_structure':ref,'prediction_structure':pred,
            'matrix_shape_match':shapes(ref)==shapes(pred) if matrices else None,
            'matrix_cells_match':[m['cells'] for m in ref['matrices']]==[m['cells'] for m in pred['matrices']] if matrices else None,
            'matrix_vertical_rule_counts_match':[int((m.get('array_spec') or '').count('|')) for m in ref['matrices']]==[int((m.get('array_spec') or '').count('|')) for m in pred['matrices']] if matrices else None,
            'factorial_count_match':ref['factorial_count']==pred['factorial_count'] if ref['factorial_count'] else None,
            'scripts_lexical_match':ref['scripts']==pred['scripts'] if ref['scripts'] else None,
            'multiline_rows_match':expected_lines==actual_lines if expected_lines else None,
            'vector_as_fraction_suspected':bool(any(m['rows']>1 and all(c==1 for c in m['columns_per_row']) for m in ref['matrices']) and not pred['matrices'] and pred['fraction_count'])}


def visual_similarity(reference_path,prediction_path):
    """Tight-ink aligned raster overlap. Descriptive score, no pass threshold."""
    ref=cv2.imread(str(reference_path));pred=cv2.imread(str(prediction_path))
    if ref is None or pred is None:return None
    a=printed_canvas(ref,(512,256),[4,4,508,252]);b=printed_canvas(pred,(512,256),[4,4,508,252])
    ma=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)<180;mb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)<180
    return {'aligned_ink_iou':float((ma&mb).sum()/max(1,(ma|mb).sum())),
            'raw_ink_aspect_ratio_reference':float((ink_box(ref)[2]-ink_box(ref)[0])/max(1,ink_box(ref)[3]-ink_box(ref)[1])),
            'raw_ink_aspect_ratio_prediction':float((ink_box(pred)[2]-ink_box(pred)[0])/max(1,ink_box(pred)[3]-ink_box(pred)[1])),
            'interpretation':'Descriptive raster similarity only; no automatic semantic or visual correctness verdict.'}
