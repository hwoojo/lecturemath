"""No learned predictions: deterministic adapter fixtures only."""
from boardnotes.v2.worker import text_result, as_dict


def test_text_order_uses_geometry_and_preserves_raw():
    raw={'rec_texts':['Equations.','System of Linear',''], 'rec_scores':[.9,.8,0.],
         'rec_boxes':[[400,0,597,69],[0,2,402,82],[374,17,422,55]]}
    result=text_result([raw])
    assert result['content']=='System of Linear Equations.'
    assert result['raw_engine_lines']['texts']==raw['rec_texts']
    assert result['line_boxes'][0]==[0.,2.,402.,82.]
    assert result['confidence']==(.8+.9)/2


def test_separate_rows_and_empty():
    result=text_result([{'rec_texts':['bottom','right','left'],'rec_scores':[.5,.8,.9],
                         'rec_boxes':[[2,50,60,70],[80,0,120,20],[0,0,50,20]]}])
    assert result['content']=='left right\nbottom'
    assert text_result([])['content']==''
    assert text_result([])['confidence'] is None


def test_wrapped_sdk_result():
    class Result:
        json={'res':{'rec_formula':'x+1'}}
    assert as_dict(Result())['rec_formula']=='x+1'
