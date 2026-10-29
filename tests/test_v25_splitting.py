"""Geometry fixtures only, not evidence of real formula recognition accuracy."""
import json

import cv2
import numpy as np
import pytest

from boardnotes.v25.splitting import assemble_rows, propose_rows


def two_rows():
    image = np.full((120, 200), 255, np.uint8)
    for y in [15, 75]:
        for x in [15, 45, 80, 115]:
            cv2.rectangle(image, (x,y), (x+12,y+22), 0, -1)
    return image


def test_two_rows_original_coordinates_and_input_preservation():
    image = two_rows()
    original = image.copy()
    result = propose_rows(image, [100,200,300,320])
    assert result['status'] == 'candidate'
    assert not result['assembly_allowed']
    assert len(result['rows']) == 2
    assert result['rows'][0]['bbox'] == [112,212,231,241]
    assert result['rows'][1]['bbox'] == [112,272,231,301]
    assert np.array_equal(image,original)
    json.dumps(result,allow_nan=False)
    assert propose_rows(image,[100,200,300,320],'aligned')['assembly_allowed']


def test_full_frame_and_cropped_paths_match():
    image = two_rows()
    full = np.full((300,400),255,np.uint8)
    full[100:220,50:250] = image
    full_result=propose_rows(full,[50,100,250,220],'aligned')
    crop_result=propose_rows(image,[50,100,250,220],'aligned')
    assert full_result == crop_result
    bgr=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    assert propose_rows(bgr,[0,0,200,120])['rows'] == propose_rows(image,[0,0,200,120])['rows']


def test_side_brace_is_analysis_only_and_never_implies_cases():
    image = two_rows()
    cv2.line(image,(178,10),(178,104),0,4)
    before=image.copy()
    result=propose_rows(image,[0,0,200,120])
    assert result['status']=='needs_review'
    assert not result['assembly_allowed']
    assert len(result['rows'])==2
    assert len(result['diagnostics']['excluded_analysis_components'])==1
    assert all(r['bbox'][2] < 178 for r in result['rows'])
    assert np.array_equal(before,image)
    assert propose_rows(image,[0,0,200,120],'aligned')['assembly_allowed']


def test_fraction_separator_is_not_equation_row_split():
    image=np.full((100,140),255,np.uint8)
    cv2.rectangle(image,(40,10),(85,28),0,-1)
    cv2.rectangle(image,(15,46),(120,48),0,-1)
    cv2.rectangle(image,(40,65),(85,84),0,-1)
    result=propose_rows(image,[0,0,140,100],'aligned')
    assert result['status']=='needs_review'
    assert 'fraction' in result['reason']
    assert result['rows']==[]
    assert not result['assembly_allowed']


def test_matrix_paired_side_structures_are_protected():
    image=two_rows()
    cv2.line(image,(2,6),(2,108),0,3)
    cv2.line(image,(190,6),(190,108),0,3)
    result=propose_rows(image,[0,0,200,120])
    assert result['status']=='needs_review'
    assert 'matrix' in result['reason']
    assert not result['rows']


@pytest.mark.parametrize('below',[False,True])
def test_small_upper_lower_script_not_promoted_to_equation(below):
    image=np.full((100,160),255,np.uint8)
    cv2.rectangle(image,(10,38),(110,62),0,-1)
    y=80 if below else 8
    cv2.rectangle(image,(100,y),(110,y+7),0,-1)
    result=propose_rows(image,[0,0,160,100],'aligned')
    assert result['status'] in {'not_split','needs_review'}
    assert not result['assembly_allowed']
    assert not result['rows']


def test_root_like_connected_structure_not_cut_and_explicit_protection():
    image=np.full((100,160),255,np.uint8)
    cv2.line(image,(10,45),(25,65),0,3)
    cv2.line(image,(25,65),(35,20),0,3)
    cv2.line(image,(35,20),(145,20),0,3)
    cv2.rectangle(image,(55,37),(130,61),0,-1)
    # Connected radical overlaps the entire vertical range: no false equation rows.
    result=propose_rows(image,[0,0,160,100])
    assert result['status'] in {'not_split','needs_review'}
    assert not result['assembly_allowed']
    for hint in ['fraction','matrix','root','script']:
        protected=propose_rows(two_rows(),[0,0,200,120],hint)
        assert protected['status']=='not_split'
        assert protected['rows']==[]


def test_single_empty_small_and_invalid_inputs():
    for shape in [(1,1),(2,3),(30,40)]:
        assert propose_rows(np.full(shape,255,np.uint8),[0,0,shape[1],shape[0]])['status']=='not_split'
    with pytest.raises(ValueError):
        propose_rows(np.ones((10,10)),[0,0,10,10])
    for bbox in [[0,0,0,10],[-1,0,9,10],[0,0,100,100],[0.,0,10,10]]:
        with pytest.raises(ValueError):
            propose_rows(np.full((10,10),255,np.uint8),bbox)
    with pytest.raises(ValueError):
        propose_rows(two_rows(),[0,0,200,120],'automatic_cases')


def test_assembly_requires_explicit_hint_and_adds_no_relations():
    rows=['3x=1','x+2y=3']
    unhinted=assemble_rows(rows)
    assert unhinted['status']=='needs_review' and unhinted['latex'] is None
    aligned=assemble_rows(rows,'aligned')
    assert aligned['latex']==r'\begin{aligned}3x=1\\x+2y=3\end{aligned}'
    assert '&' not in aligned['latex']
    assert aligned['review_required']
    cases=assemble_rows(rows,'cases')
    assert cases['latex']==r'\begin{cases}3x=1\\x+2y=3\end{cases}'
    assert assemble_rows(['3x=1',''],'aligned')['status']=='failed'
    assert assemble_rows([r'\begin{matrix}1\end{matrix}','x=1'],'aligned')['status']=='needs_review'
    with pytest.raises(ValueError):
        assemble_rows(['x=1'],'matrix')
