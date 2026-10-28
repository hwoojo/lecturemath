"""Model-free V2 contract tests. Fake workers are NOT OCR accuracy evidence."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from pypdf import PdfReader
import pytest

from boardnotes.v2 import pipeline
from boardnotes.v2.schema import load_document, save_document, validate_document, validate_regions


def source_image(tmp_path, width=24, height=16):
    # Arbitrary RGB pixels make crop-coordinate errors observable without a model.
    source = np.random.default_rng(31).integers(0,256,(height,width,3),dtype=np.uint8)
    path = tmp_path/'input.png'
    assert cv2.imwrite(str(path), source)
    return path, source


def fake_worker_calls(monkeypatch, transform=None):
    """Stable fake IPC response. No Paddle/Torch/model loading occurs here."""
    calls = []

    def fake(python, kind, jobs, config, folder, timeout=300, probe=False):
        calls.append({'kind':kind, 'jobs':deepcopy(jobs), 'probe':probe})
        if probe:
            return {'protocol':1, 'versions':{'fixture':'model-free-v1'}}
        if transform is not None:
            return transform(kind, jobs)
        items = []
        for job in jobs:
            items.append({'id':job['id'], 'status':'ok',
                          'content':'Fixture: α + y' if kind=='text' else None,
                          'latex':r'x+1=y' if kind=='formula' else None,
                          'confidence':0.75 if kind=='text' else None,
                          'error':None, 'seconds':0.0})
        return {'items':items,'versions':{'fixture':'model-free-v1'},'kind':kind,
                'model_load_seconds':0.0,'process_peak_rss_mib':0.0}

    monkeypatch.setattr(pipeline, 'worker_call', fake)
    return calls


def test_region_schema_half_open_ids_bounds_and_json_roundtrip(tmp_path):
    regions = [{'id':'a','type':'text','bbox':[0,0,24,16]},
               {'id':2,'type':'formula','bbox':[23,15,24,16]}]
    assert validate_regions(regions,24,16) is regions
    for box in [[0,0,0,1],[-1,0,1,1],[0,0,25,1],[1.2,0,3,4],[False,0,3,4]]:
        with pytest.raises(ValueError):
            validate_regions([{'id':'bad','type':'text','bbox':box}],24,16)
    with pytest.raises(ValueError,match='unique'):
        validate_regions([{'id':1,'type':'text','bbox':[0,0,2,2]},
                          {'id':'1','type':'formula','bbox':[2,2,4,4]}],24,16)
    with pytest.raises(ValueError,match='type'):
        validate_regions([{'id':'a','type':'guessed-equation','bbox':[0,0,2,2]}],24,16)
    document={'version':'2.0','source_image':'/fixture-not-an-inference.png','width':24,'height':16,
              'timestamp':1.25,'regions':[{**regions[0],'content':'한글 α','status':'ok','confidence':None}]}
    path=tmp_path/'fixture.json'
    save_document(document,path)
    assert load_document(path)==document
    assert '한글 α' in path.read_text()
    for value in [float('nan'),float('inf'),-1,True]:
        with pytest.raises(ValueError,match='timestamp'):
            validate_document({**document,'timestamp':value})


def test_recognition_preserves_source_mask_crops_and_offsets(monkeypatch,tmp_path):
    source, pixels = source_image(tmp_path)
    mask = np.full(pixels.shape[:2],255,np.uint8)
    mask[4:8,5:10] = 0
    mask_path=tmp_path/'binary.png'
    cv2.imwrite(str(mask_path),mask)

    def respond(kind,jobs):
        assert kind=='text'
        return {'items':[{'id':job['id'],'status':'ok','content':'FAKE transcript',
                         'latex':None,'confidence':.6,'error':None,
                         'line_boxes':[[-1,-1,2.2,3.1],[2,1,5,4]]} for job in jobs]}

    calls=fake_worker_calls(monkeypatch,respond)
    output=tmp_path/'out'
    regions=[{'id':'title','type':'text','bbox':[3,2,11,10]}]
    original=deepcopy(regions)
    result=pipeline.recognize(source,regions,output,Path('/unused/fake-python'),
                              variants={'rgb':source,'b0':mask_path,'b3':mask_path},timestamp=4.25)
    assert regions==original
    np.testing.assert_array_equal(cv2.imread(str(output/'source.png')),pixels)
    for variant,document in result.items():
        assert document['source_image']==str((output/'source.png').resolve())
        assert (document['width'],document['height'],document['timestamp'])==(24,16,4.25)
        region=document['regions'][0]
        assert region['bbox']==[3,2,11,10]
        assert region['line_boxes']==[[3,2,6,6],[5,3,8,6]]
        crop=cv2.imread(region['crop_image'])
        expected=pixels[2:10,3:11] if variant=='rgb' else cv2.cvtColor(mask[2:10,3:11],cv2.COLOR_GRAY2BGR)
        np.testing.assert_array_equal(crop,expected)
        assert region['status']=='ok' and document['metadata']['input_variant']==variant
        assert load_document(output/(variant+'.json'))==document
    assert [call['kind'] for call in calls]==['probe','text']


def test_cache_reuse_and_changed_crop_miss(monkeypatch,tmp_path):
    source,pixels=source_image(tmp_path)
    calls=fake_worker_calls(monkeypatch)
    regions=[{'id':'text','type':'text','bbox':[1,1,8,8]},
             {'id':'math','type':'formula','bbox':[12,2,22,10]}]
    cache=tmp_path/'shared-cache'
    first=pipeline.recognize(source,regions,tmp_path/'first',Path('/fake/python'),cache_dir=cache)
    assert all(not r['cache_hit'] for r in first['rgb']['regions'])
    assert [c['kind'] for c in calls]==['probe','text','formula']
    calls.clear()
    second=pipeline.recognize(source,regions,tmp_path/'second',Path('/fake/python'),cache_dir=cache)
    assert all(r['cache_hit'] for r in second['rgb']['regions'])
    assert [c['kind'] for c in calls]==['probe']
    assert [(r['content'],r['latex']) for r in first['rgb']['regions']]==[(r['content'],r['latex']) for r in second['rgb']['regions']]
    pixels[3,3]=[255,254,253]
    cv2.imwrite(str(source),pixels)
    calls.clear()
    third=pipeline.recognize(source,regions,tmp_path/'third',Path('/fake/python'),cache_dir=cache)
    assert [r['cache_hit'] for r in third['rgb']['regions']]==[False,True]
    assert [c['kind'] for c in calls]==['probe','text']


def test_empty_failed_other_and_unknown_remain_distinct(monkeypatch,tmp_path):
    source,pixels=source_image(tmp_path)
    def respond(kind,jobs):
        return {'items':[{'id':j['id'],'status':'empty' if kind=='text' else 'failed',
                          'content':'' if kind=='text' else None,'latex':None,'confidence':None,
                          'error':None if kind=='text' else 'Controlled fake formula failure'} for j in jobs]}
    calls=fake_worker_calls(monkeypatch,respond)
    regions=[{'id':'text','type':'text','bbox':[0,0,6,6]},
             {'id':'math','type':'formula','bbox':[6,0,12,6]},
             {'id':'graph','type':'other','bbox':[0,6,12,15]},
             {'id':'uncertain','type':'unknown','bbox':[12,6,24,15]}]
    cache=tmp_path/'cache'
    docs=pipeline.recognize(source,regions,tmp_path/'out',Path('/fake/python'),cache_dir=cache)
    result=docs['rgb']['regions']
    assert [r['status'] for r in result]==['empty','failed','preserved','needs_review']
    assert result[1]['error']=='Controlled fake formula failure'
    assert result[2]['model'] is None and result[3]['content'] is None
    assert all(Path(r['crop_image']).is_file() for r in result)
    calls.clear()
    again=pipeline.recognize(source,regions,tmp_path/'again',Path('/fake/python'),cache_dir=cache)
    assert again['rgb']['regions'][0]['cache_hit']
    assert not again['rgb']['regions'][1]['cache_hit']
    assert [c['kind'] for c in calls]==['probe','formula']
    np.testing.assert_array_equal(cv2.imread(str(tmp_path/'out/source.png')),pixels)


def test_variant_alignment_and_binary_png_contract(monkeypatch,tmp_path):
    source,pixels=source_image(tmp_path)
    fake_worker_calls(monkeypatch)
    regions=[{'id':'t','type':'text','bbox':[0,0,5,5]}]
    wrong=tmp_path/'wrong.png';cv2.imwrite(str(wrong),np.full((8,12),255,np.uint8))
    with pytest.raises(ValueError,match='coordinates|resolution'):
        pipeline.recognize(source,regions,tmp_path/'wrong-out',Path('/fake'),variants={'b0':wrong})
    gray=tmp_path/'gray.png';cv2.imwrite(str(gray),np.full((16,24),127,np.uint8))
    with pytest.raises(ValueError,match='binary'):
        pipeline.recognize(source,regions,tmp_path/'gray-out',Path('/fake'),variants={'b3':gray})
    jpeg=tmp_path/'lossy.jpg';cv2.imwrite(str(jpeg),np.full((16,24),255,np.uint8))
    with pytest.raises(ValueError,match='PNG|png'):
        pipeline.recognize(source,regions,tmp_path/'jpeg-out',Path('/fake'),variants={'b0':jpeg})
    disguised=tmp_path/'not-really-png.png';disguised.write_bytes(jpeg.read_bytes())
    with pytest.raises(ValueError,match='PNG|png'):
        pipeline.recognize(source,regions,tmp_path/'disguised-out',Path('/fake'),variants={'b0':disguised})


def test_detection_proposals_are_unknown_sorted_clipped_and_deduplicated(monkeypatch,tmp_path):
    source,_=source_image(tmp_path,width=80,height=60)
    boxes=[[30,35,60,45],[-2,2,15,12],[0,2,16,13],[70,2,90,10],[4,4,4,4]]
    def respond(kind,jobs):
        assert kind=='detect'
        return {'items':[{'id':'page','status':'ok','line_boxes':boxes,
                          'line_texts':['late','left','duplicate','right','zero'],
                          'line_scores':[.8,.9,.7,.6,.3]}]}
    fake_worker_calls(monkeypatch,respond)
    result=pipeline.detect(source,tmp_path/'detect',Path('/fake/python'),timestamp=3.5)
    assert [r['bbox'] for r in result['regions']]==[[0,2,15,12],[70,2,80,10],[30,35,60,45]]
    assert [r['candidate_text'] for r in result['regions']]==['left','right','late']
    assert all(r['type']=='unknown' and r['status']=='needs_review' for r in result['regions'])
    assert all(r['content'] is None and r['latex'] is None for r in result['regions'])
    assert result['timestamp']==3.5
    assert load_document(tmp_path/'detect/regions.json')==result


def test_inconsistent_detection_response_is_structured_failure(monkeypatch,tmp_path):
    source,pixels=source_image(tmp_path)
    def respond(kind,jobs):
        return {'items':[{'id':'page','status':'ok','line_boxes':[[0,0,10,10]],'line_texts':[],'line_scores':[]}]}
    fake_worker_calls(monkeypatch,respond)
    result=pipeline.detect(source,tmp_path/'detect',Path('/fake/python'))
    assert result['regions']==[]
    worker=result['metadata']['worker']['items'][0]
    assert worker['status']=='failed' and 'inconsistent' in worker['error'].lower()
    assert (tmp_path/'detect/regions.json').is_file()
    np.testing.assert_array_equal(cv2.imread(result['source_image']),pixels)


def test_missing_worker_python_serializes_failures_without_fallback(tmp_path):
    source,pixels=source_image(tmp_path)
    regions=[{'id':'t','type':'text','bbox':[0,0,8,8]},
             {'id':'f','type':'formula','bbox':[8,0,20,8]}]
    result=pipeline.recognize(source,regions,tmp_path/'missing',tmp_path/'python-does-not-exist')
    loaded=load_document(tmp_path/'missing/rgb.json')
    assert loaded==result['rgb']
    assert all(r['status']=='failed' and r['content'] is None and r['latex'] is None for r in loaded['regions'])
    assert all('FileNotFoundError' in r['error'] for r in loaded['regions'])
    assert loaded['metadata']['probe_error']
    np.testing.assert_array_equal(cv2.imread(loaded['source_image']),pixels)


def test_render_cli_reuses_edited_json_without_loading_models(tmp_path):
    source,_=source_image(tmp_path)
    document={'version':'2.0','source_image':str(source),'width':24,'height':16,'timestamp':0.,
              'regions':[{'id':'edited','type':'text','bbox':[0,0,24,16],
                          'status':'ok','content':'Edited from model-free fixture','latex':None,
                          'confidence':None,'model':'manual-fixture','error':None}]}
    file=tmp_path/'edited.json';save_document(document,file)
    output=tmp_path/'rendered.pdf'
    # Fail loudly if the render path ever imports an inference framework.
    program="""import builtins, sys
original_import=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','paddle','paddleocr','paddlex'}:
        raise AssertionError('render must not import model frameworks')
    return original_import(name,*args,**kwargs)
builtins.__import__=guarded
from boardnotes.v2.cli import main
sys.argv=['v2','render',sys.argv[1],'--output',sys.argv[2]]
main()
"""
    run=subprocess.run([sys.executable,'-c',program,str(file),str(output)],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout)['pages']==2
    assert 'Edited from model-free fixture' in PdfReader(output).pages[1].extract_text()
    assert 'paddle' not in run.stderr.lower()
