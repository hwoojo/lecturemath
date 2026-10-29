"""Explainable OCR-candidate routing, not a trained or calibrated classifier."""
from copy import deepcopy
import re


def route_candidates(regions):
    result=deepcopy(regions)
    for region in result:
        if region['type']!='unknown' or region.get('subtype')!='ocr_only_candidate':continue
        if any(c.get('kind')=='multiple_layout_parents' for c in region.get('conflicts',[])):continue
        text=region.get('candidate_text') or ''
        words=re.findall(r'[A-Za-z]{2,}',text)
        symbols=re.findall(r'[=<>≤≥≠∫∑√]|\\(?:frac|sqrt|sum|int|begin)',text)
        if symbols and len(words)<=1:
            kind='formula';reason='explicit equation/comparison/math operator with little prose'
        elif len(words)>=2 and not symbols:
            kind='text';reason='multiple alphabetic words without equation operators'
        else:continue
        region['type']=kind
        region['suggested_type']=kind
        region['routing']={'method':'ocr_symbol_prose_heuristic','evidence_text':text,'reason':reason,
                           'calibrated_confidence':None,'notice':'Routing proposal only; mathematical meaning/correctness unverified.'}
        region['status']='needs_review';region['review_required']=True
    return result
