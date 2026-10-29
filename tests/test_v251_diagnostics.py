"""Meaningful safeguards for fixed-condition geometry and conservative scoring."""
import cv2
import numpy as np
import pytest
from experiments.v251.inputs import crop_box,condition_crop,printed_canvas,ink_box
from experiments.v251.evaluate import compare,structure,visual_similarity


def test_context_padding_is_clipped_source_crop_and_never_fabricated():
    image=np.arange(30*40*3,dtype=np.uint16).reshape(30,40,3).astype(np.uint8)
    crop,box=condition_crop(image,[1,2,11,12],'pad20')
    assert box==[0,0,13,14]
    np.testing.assert_array_equal(crop,image[:14,:13])
    with pytest.raises(ValueError):crop_box([0,0,4,4],10,10,-.1)


def test_tight_keeps_isolated_dot_and_records_original_coordinates():
    image=np.full((40,60,3),255,np.uint8)
    image[8:22,18:21]=0;image[28:31,18:21]=0
    crop,box=condition_crop(image,[10,5,35,35],'tight')
    assert box==[18,8,21,31]
    assert (crop[-3:]==0).all()
    assert (image[28:31,18:21]==0).all()


def test_matrix_wrapper_difference_is_not_strict_or_general_semantic_match():
    ref=r'\begin{bmatrix}1&2\\3&4\end{bmatrix}'
    pred=r'\left[\begin{array}{cc}{1}&{2}\\{3}&{4}\end{array}\right]'
    report=compare(ref,pred)
    assert report['strict_exact_match'] is False
    assert report['matrix_cells_match'] is True and report['matrix_shape_match'] is True
    assert 'semantic_match' not in report
    assert compare(ref,r'\begin{matrix}1&2\\3&5\end{matrix}')['matrix_cells_match'] is False
    assert compare(ref,r'\begin{matrix}1&2&3&4\end{matrix}')['matrix_shape_match'] is False


def test_vector_fraction_and_factorial_errors_remain_errors():
    assert compare(r'\begin{bmatrix}p\\q\\r\end{bmatrix}',r'\frac{p}{q}')['vector_as_fraction_suspected']
    assert compare('3!=6','31=6')['factorial_count_match'] is False
    assert structure(r'3\!+1')['factorial_count']==0
    assert compare('x_i^2','x_2^i')['scripts_lexical_match'] is False
    assert compare('x=1','x=1','failed')['strict_exact_match'] is False
    ref=r'\begin{aligned}x&=1\\y&=2\end{aligned}'
    assert compare(ref,r'\begin{array}{r}{x=1}\\{y=2}\end{array}')['multiline_rows_match']
    assert not compare(ref,r'\begin{array}{r}{x=1}\end{array}')['multiline_rows_match']


def test_print_placement_preserves_aspect_and_visual_score_is_not_verdict(tmp_path):
    image=np.full((20,40,3),255,np.uint8);image[3:17,5:35]=0
    placed=printed_canvas(image,(100,100),[10,10,90,90]);a,b,c,d=ink_box(placed)
    assert abs((c-a)/(d-b)-30/14)<.1
    one=tmp_path/'a.png';two=tmp_path/'b.png';cv2.imwrite(str(one),image);cv2.imwrite(str(two),image)
    report=visual_similarity(one,two)
    assert report['aligned_ink_iou']==1
    assert 'correct' not in report
