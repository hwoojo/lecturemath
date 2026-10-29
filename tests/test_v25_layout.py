"""Deterministic detector fixtures only; not measured layout model accuracy."""
from copy import deepcopy
import json
import math
import pytest

from boardnotes.v2.schema import validate_regions
from boardnotes.v25.layout import fuse_regions, layout_to_items


def layout(box,label='text',score=.8):
    return {'bbox':box,'label':label,'score':score,'model':'PP-DocLayout-S'}


def ocr(box,text='fixture',score=.7):
    return {'id':'ocr-fixture','bbox':box,'type':'unknown','confidence':score,'candidate_text':text}


def test_class_mapping_clipping_and_original_coordinate_provenance():
    proposals=[layout([-1.2,2.1,20.2,15.8],'doc_title'),layout([45,5,70,25],'formula'),layout([5,40,80,75],'chart')]
    original=deepcopy(proposals)
    result=fuse_regions(proposals,[],100,80)
    validate_regions(result,100,80)
    assert [r['type'] for r in result]==['text','formula','other']
    assert result[0]['bbox']==[0,2,21,16]
    assert result[0]['provenance'][0]['raw_bbox']==[-1.2,2.1,20.2,15.8]
    assert result[-1]['status']=='preserved'
    assert all(r['review_required'] for r in result)
    assert proposals==original
    json.dumps(result,allow_nan=False)


def test_duplicate_layout_proposals_preserve_both_sources():
    result=fuse_regions([layout([5,5,55,35],'formula'),layout([6,5,55,35],'formula',.9)],[],100,80)
    assert len(result)==1 and result[0]['type']=='formula'
    assert result[0]['bbox']==[5,5,55,35]
    assert len(result[0]['provenance'])==2
    assert result[0]['detection_confidence']==.9
    assert result[0]['confidence'] is None


def test_conflicting_classes_are_unknown_not_high_score_winners():
    result=fuse_regions([layout([5,5,55,35],'formula',.99),layout([6,5,55,35],'text',.999)],[],100,80)
    assert len(result)==1 and result[0]['type']=='unknown'
    assert result[0]['status']=='needs_review'
    assert result[0]['conflicts'][0]['kind']=='layout_type_conflict'
    assert {p['label'] for p in result[0]['provenance']}=={'text','formula'}


def test_contained_tiny_ocr_attached_without_independent_duplicate_recognition():
    result=fuse_regions([layout([10,10,90,65],'formula')],[ocr([20,20,21,22]),ocr([91,70,92,71],'small point')],100,80)
    assert len(result)==2
    parent=next(r for r in result if r['type']=='formula')
    assert parent['embedded_ocr_candidates'][0]['bbox']==[20,20,21,22]
    orphan=next(r for r in result if r['type']=='unknown')
    assert orphan['bbox']==[91,70,92,71] and orphan['suggested_type']=='text'


def test_image_parent_preserves_ocr_evidence_and_unknown_class_not_guessed():
    result=fuse_regions([layout([0,0,100,80],'image')],[ocr([4,5,22,16])],100,80)
    assert len(result)==1 and result[0]['type']=='other'
    assert result[0]['embedded_ocr_candidates'][0]['candidate_text']=='fixture'
    assert fuse_regions([layout([0,0,2,2],'new_native_label')],[],100,80)[0]['type']=='unknown'


def test_orphan_duplicates_and_reading_order():
    proposals=[ocr([50,40,60,50],'third'),ocr([30,0,40,10],'second'),ocr([0,0,10,10],'first'),ocr([0,0,10,10],'first copy')]
    result=fuse_regions([],proposals,100,80)
    assert [r['candidate_text'] for r in result]==['first','second','third']
    assert len(result[0]['provenance'])==2
    assert len({r['id'] for r in result})==3


def test_ambiguous_multiple_parent_kept_as_unknown():
    # Two text rows are near enough for this deliberately broad OCR candidate.
    proposals=[layout([0,0,60,60],'text'),layout([20,0,80,60],'text')]
    result=fuse_regions(proposals,[ocr([25,20,45,40])],100,80)
    candidate=next(r for r in result if r['subtype']=='ocr_only_candidate')
    assert candidate['type']=='unknown'
    assert candidate['conflicts'][0]['kind']=='multiple_layout_parents'


@pytest.mark.parametrize('box',[[0,0,0,3],[4,0,2,3],[0,float('nan'),2,3],[True,0,2,3],[101,1,105,4],[0,0,2]])
def test_bad_coordinates_rejected(box):
    with pytest.raises(ValueError): fuse_regions([layout(box)],[],100,80)


@pytest.mark.parametrize('score',[-.1,1.1,float('nan'),float('inf'),True])
def test_invalid_scores_rejected(score):
    with pytest.raises(ValueError): fuse_regions([layout([0,0,10,10],score=score)],[],100,80)


def test_sdk_parser_wrapped_json_and_unknown_class_id_not_guessed():
    class Result:
        json={'res':{'boxes':[{'cls_id':6,'label':'formula','score':.8,'coordinate':[1.,2.,8.,9.]}]}}
    items=layout_to_items(Result())
    assert items==[{'bbox':[1.,2.,8.,9.],'label':'formula','score':.8,'model':'PP-DocLayout-S','cls_id':6}]
    assert layout_to_items(json.dumps(Result.json))==items
    assert layout_to_items({'boxes':[]})==[]
    with pytest.raises(ValueError):layout_to_items({'boxes':[{'cls_id':6,'score':.8,'coordinate':[1,2,8,9]}]})
    with pytest.raises(ValueError):layout_to_items({'not_boxes':[]})
    assert fuse_regions([],[],100,80)==[]


def test_page_sized_image_failure_keeps_ocr_candidates_independent():
    # Development regression for the measured board-domain failure. These boxes
    # are fixtures; this test does not measure actual detector performance.
    proposals=[layout([.43,0,1869.82,1066.49],'image',.564)]
    candidates=[ocr([100,200,300,260],'first'),ocr([400,200,600,260],'second'),ocr([700,300,701,302],'+')]
    result=fuse_regions(proposals,candidates,1920,1080)
    assert len(result)==4
    dominant=next(r for r in result if r['subtype']=='dominant_layout_container')
    assert dominant['type']=='unknown' and dominant['status']=='needs_review'
    assert dominant['provenance'][0]['label']=='image'
    assert dominant['provenance'][0]['score']==.564
    assert len(dominant['embedded_ocr_candidates'])==3
    children=[r for r in result if r['subtype']=='ocr_only_candidate']
    assert len(children)==3
    assert {tuple(r['bbox']) for r in children}=={tuple(c['bbox']) for c in candidates}
    assert all(r['type']=='unknown' and r['conflicts'][0]['kind']=='ambiguous_layout_container_parent' for r in children)
    validate_regions(result,1920,1080)


def test_non_dominant_graphic_still_suppresses_duplicate_ocr_recognition():
    candidates=[ocr([4,5,12,16]),ocr([25,20,32,25])]
    result=fuse_regions([layout([0,0,40,40],'image')],candidates,100,80)
    assert len(result)==1 and result[0]['type']=='other'
    assert len(result[0]['embedded_ocr_candidates'])==2
    assert not result[0].get('dominant_container_ambiguous',False)


def test_one_distinct_ocr_box_does_not_trigger_dominant_guard():
    result=fuse_regions([layout([0,0,100,80],'image')],[ocr([4,5,22,16]),ocr([4,5,22,16])],100,80)
    assert len(result)==1 and result[0]['type']=='other'
    assert not result[0].get('dominant_container_ambiguous',False)


def test_native_title_with_symbolic_ocr_retains_independent_review_candidates():
    # Development regression from the actual E2E clip, not an accuracy test.
    proposals=[layout([1,0,477,163],'doc_title',.6162163)]
    candidates=[ocr([278,10,388,43],'x =]5',.739),ocr([40,45,55,65],'A',.82),ocr([75,70,160,93],'prose',.9)]
    result=fuse_regions(proposals,candidates,480,360)
    assert len(result)==4
    parent=next(r for r in result if r['subtype']=='layout_text_ocr_math_conflict')
    assert parent['type']=='unknown' and parent['ambiguous_ocr_container']
    assert parent['status']=='needs_review'
    assert parent['provenance'][0]['label']=='doc_title'
    assert parent['provenance'][0]['predicted_type']=='text'
    assert len(parent['embedded_ocr_candidates'])==3
    assert parent['ocr_math_evidence'][0]['candidate_text']=='x =]5'
    assert parent['ocr_math_evidence'][0]['contains_digit']
    children=[r for r in result if r['subtype']=='ocr_only_candidate']
    assert len(children)==3 and all(r['type']=='unknown' for r in children)
    assert all(r['conflicts'][0]['parent_conflict_kinds']==['layout_label_vs_ocr_math_evidence'] for r in children)
    assert not any(r['type']=='formula' for r in result)
    validate_regions(result,480,360)


@pytest.mark.parametrize('text',['Let x=1 denote a value','a ≠ b','x < 2','y ≥ 3','∫ f(x) dx',r'\frac{a}{b}'])
def test_mixed_or_symbolic_text_is_ambiguous_not_formula(text):
    result=fuse_regions([layout([0,0,80,40],'text')],[ocr([5,5,70,25],text)],100,80)
    assert len(result)==2
    assert all(r['type']=='unknown' for r in result)
    assert any(r.get('ambiguous_ocr_container') for r in result)


@pytest.mark.parametrize('text',['System of linear equations','Addition + subtraction','Grade > average','='])
def test_prose_without_clear_symbolic_evidence_keeps_text_parent(text):
    result=fuse_regions([layout([0,0,80,40],'text')],[ocr([5,5,70,25],text)],100,80)
    assert len(result)==1 and result[0]['type']=='text'
    assert len(result[0]['embedded_ocr_candidates'])==1
    assert not result[0].get('ambiguous_ocr_container')
