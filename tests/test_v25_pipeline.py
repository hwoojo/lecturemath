"""Model-free V2.5 orchestration contracts; fixtures are not OCR quality data."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from boardnotes.v25 import cli, pipeline, resources
from boardnotes.v25.routing import route_candidates


def images(tmp_path):
    rgb=np.random.default_rng(14).integers(0,256,(40,60,3),dtype=np.uint8)
    source=tmp_path/'source.png';assert cv2.imwrite(str(source),rgb)
    mask=np.full((40,60),255,np.uint8);mask[5:10,6:20]=0;mask[22:28,6:25]=0
    b0=tmp_path/'existing-b0.png';assert cv2.imwrite(str(b0),mask)
    return source,b0,rgb,mask


def mocked_models(monkeypatch,mask_path,render_failure=False):
    """Record calls while returning explicitly fake, deterministic predictions."""
    calls={'recognize':[],'obtain':[],'render':[]}
    def obtain(source,provided,output,v1_python,source_variant='rgb'):
        calls['obtain'].append((Path(source),provided,source_variant))
        target=Path(output)/'b0.png';target.write_bytes(mask_path.read_bytes())
        return target,{'mode':'fixture B0; no V34 executed'}
    def recognize(source,specs,output,ocr_python,config,**kwargs):
        calls['recognize'].append({'source':Path(source),'specs':deepcopy(specs),'kwargs':deepcopy(kwargs)})
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        source_copy=output/'source.png';source_copy.write_bytes(Path(source).read_bytes())
        regions=[]
        for index,spec in enumerate(specs):
            kind=spec['type'];key=spec['id']
            regions.append({**deepcopy(spec),'status':'ok' if kind in ('text','formula') else 'preserved',
                'content':'model-free fixture text' if kind=='text' else None,
                'latex':('x=1' if key.endswith('__row0') else 'y=2' if key.endswith('__row1') else 'whole_fixture') if kind=='formula' else None,
                'model':'fixture, not actual inference','confidence':None,'error':None,
                'crop_image':str(source_copy),'review_required':True})
        return {'b0':{'version':'2.0','source_image':str(source_copy.resolve()),'width':60,'height':40,
                      'timestamp':kwargs.get('timestamp'),'regions':regions,'metadata':{'input_variant':'b0'}}}
    def render(document,destination,font=None):
        calls['render'].append(deepcopy(document));destination=Path(destination)
        destination.write_bytes(b'%PDF-model-free-test-fixture')
        records=[{'region_id':r['id'],'status':'failed' if render_failure else 'rendered',
                  'error':'fixture rendering syntax failure' if render_failure else None} for r in document['regions'] if r['type']=='formula']
        destination.with_suffix('.rendering.json').write_text(json.dumps({'regions':records}))
        return 2
    monkeypatch.setattr(pipeline,'obtain_b0',obtain)
    monkeypatch.setattr(pipeline,'recognize',recognize)
    monkeypatch.setattr(pipeline,'render_pdf',render)
    return calls


def two_rows(monkeypatch,allow=False):
    def propose(binary,bbox,hint=None):
        return {'status':'candidate','assembly_allowed':allow and hint=='aligned',
                'rows':[{'bbox':[6,5,25,12]},{'bbox':[6,22,25,30]}],
                'diagnostics':{'structure_hint':hint,'fixture':'geometry only'}}
    monkeypatch.setattr(pipeline,'propose_rows',propose)


def test_b0_is_recognition_default_and_rgb_is_untouched(monkeypatch,tmp_path):
    source,b0,rgb,mask=images(tmp_path);calls=mocked_models(monkeypatch,b0)
    before=source.read_bytes()
    specs=[{'id':'t','type':'text','bbox':[2,3,35,18]},
           {'id':'graph','type':'other','bbox':[30,20,55,36]}]
    doc,report=pipeline.process_image(source,tmp_path/'out','/fixture/python',{'formula_model':'PP-FormulaNet-S'},
                                      b0=b0,regions=specs,timestamp=12.25)
    call=calls['recognize'][0]
    assert call['source']==source and call['kwargs']['variants']=={'b0':tmp_path/'out/b0.png'}
    assert source.read_bytes()==before
    np.testing.assert_array_equal(cv2.imread(doc['source_image']),rgb)
    np.testing.assert_array_equal(cv2.imread(str(call['kwargs']['variants']['b0']),0),mask)
    assert doc['timestamp']==12.25 and doc['metadata']['version']=='2.5'
    assert doc['metadata']['region_origin']['mode']=='explicit reviewed/oracle regions'
    assert report['type_counts']=={'text':1,'formula':0,'other':1,'unknown':0}
    assert report['failed_recognitions']==0


def test_auto_detection_and_routing_evidence_survives_recognition(monkeypatch,tmp_path):
    source,b0,_,_=images(tmp_path);calls=mocked_models(monkeypatch,b0)
    recognize=pipeline.recognize
    def confident_text(*args,**kwargs):
        result=recognize(*args,**kwargs)
        result['b0']['regions'][0]['review_required']=False
        return result
    monkeypatch.setattr(pipeline,'recognize',confident_text)
    regions=[{'id':'candidate','type':'text','bbox':[2,3,35,18],'status':'needs_review',
              'review_required':True,'provenance':[{'detector':'fixture-layout','score':.51}],
              'routing':{'method':'fixture routing, not trained classifier'},'candidate_text':'hello board'}]
    def analysis(source_arg,mask,output,*args):
        assert Path(source_arg)==source and Path(mask).name=='b0.png'
        return {'regions':deepcopy(regions)}
    monkeypatch.setattr(pipeline,'analyze_regions',analysis)
    doc,report=pipeline.process_image(source,tmp_path/'out','fixture-python',{'formula_model':'PP-FormulaNet-S'},b0=b0)
    detection=doc['regions'][0]['detection']
    assert doc['regions'][0]['review_required'] is True
    assert doc['regions'][0]['status']=='ok'
    assert detection['provenance']==regions[0]['provenance']
    assert detection['routing']==regions[0]['routing']
    assert doc['metadata']['region_origin']['mode']=='automatic'
    assert 'analysis/regions.json' in doc['metadata']['region_origin']['regions_json']
    assert 'never ground truth' in doc['metadata']['formula_model_selection']


def test_rows_do_not_replace_whole_prediction_or_invent_structure(monkeypatch,tmp_path):
    source,b0,_,_=images(tmp_path);calls=mocked_models(monkeypatch,b0);two_rows(monkeypatch)
    def forbidden(*args):pytest.fail('assembly called without an explicit safe structure hint')
    monkeypatch.setattr(pipeline,'assemble_rows',forbidden)
    specs=[{'id':'eq','type':'formula','bbox':[3,2,40,35]}]
    doc,report=pipeline.process_image(source,tmp_path/'out','fixture',{'formula_model':'PP-FormulaNet-S'},
                                      b0=b0,regions=specs,split_formulas=True)
    assert [r['id'] for r in doc['regions']]==['eq','eq__row0','eq__row1']
    assert doc['regions'][0]['latex']=='whole_fixture'
    assert doc['regions'][1]['latex']=='x=1' and doc['regions'][2]['latex']=='y=2'
    assembly=doc['metadata']['row_assemblies']['eq']
    assert assembly['status']=='needs_review' and assembly['latex'] is None
    assert 'whole-region prediction also retained' in doc['regions'][1]['role']
    assert report['region_count']==1 and report['recognition_count']==3


def test_explicit_aligned_hint_adds_review_candidate_and_keeps_whole(monkeypatch,tmp_path):
    source,b0,_,_=images(tmp_path);mocked_models(monkeypatch,b0);two_rows(monkeypatch,allow=True)
    specs=[{'id':'eq','type':'formula','bbox':[3,2,40,35]}]
    doc,_=pipeline.process_image(source,tmp_path/'out','fixture',{'formula_model':'PP-FormulaNet-S'},
                                 b0=b0,regions=specs,split_formulas=True,structure_hints={'eq':'aligned'})
    assert [r['id'] for r in doc['regions']]==['eq','eq__row0','eq__row1','eq__assembled']
    whole,assembled=doc['regions'][0],doc['regions'][-1]
    assert whole['latex']=='whole_fixture'
    assert assembled['latex']==r'\begin{aligned}x=1\\y=2\end{aligned}'
    assert assembled['bbox']==whole['bbox'] and assembled['status']=='needs_review'
    assert assembled['whole_region_id']=='eq' and assembled['row_ids']==['eq__row0','eq__row1']
    assert assembled['confidence'] is None and assembled['review_required']


def test_manual_edits_forward_cache_without_reclassifying(monkeypatch,tmp_path):
    source,b0,_,_=images(tmp_path);calls=mocked_models(monkeypatch,b0)
    monkeypatch.setattr(pipeline,'analyze_regions',lambda *a,**k:pytest.fail('manual review unexpectedly redetected'))
    cache=tmp_path/'shared-cache'
    for index,box in enumerate([[2,3,25,16],[3,4,28,18]]):
        pipeline.process_image(source,tmp_path/f'out-{index}','fixture',{'formula_model':'PP-FormulaNet-S'},
                               b0=b0,regions=[{'id':'reviewed','type':'text','bbox':box}],cache_dir=cache)
    assert [c['kwargs']['cache_dir'] for c in calls['recognize']]==[cache,cache]
    assert [c['specs'][0]['bbox'] for c in calls['recognize']]==[[2,3,25,16],[3,4,28,18]]


def test_rendering_failure_is_not_a_recognition_failure(monkeypatch,tmp_path):
    source,b0,_,_=images(tmp_path);mocked_models(monkeypatch,b0,render_failure=True)
    two_rows(monkeypatch)
    doc,report=pipeline.process_image(source,tmp_path/'out','fixture',{'formula_model':'PP-FormulaNet-S'},
                                      b0=b0,regions=[{'id':'eq','type':'formula','bbox':[3,2,40,35]}])
    assert doc['regions'][0]['status']=='ok'
    assert report['failed_recognitions']==0
    assert report['rendering']['regions'][0]['status']=='failed'
    assert report['rendering']['regions'][0]['error']=='fixture rendering syntax failure'
    assert (tmp_path/'out/result.json').is_file()


def test_manifest_keeps_seconds_and_temporal_composite_context(monkeypatch,tmp_path):
    pages=[]
    image=tmp_path/'composite.png';ImageFixture=np.full((10,20),255,np.uint8);cv2.imwrite(str(image),ImageFixture)
    page={'image':'composite.png','variant':'b0','timestamp':12.75,'interval_seconds':[10.5,12.75],
          'coordinate_system':'uncropped_st3d_composite_pixels','original_rgb':None,'width':20,'height':10,
          'scale_to_video':[96.,108.]}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'format':'boardnotes-v2-inputs-1','pages':[page]}))
    def fake_process(source,output,ocr_python,config,**kwargs):
        assert source==image.resolve();assert kwargs['timestamp']==12.75 and kwargs['source_variant']=='b0'
        Path(output).mkdir(parents=True)
        doc={'version':'2.0','source_image':str(source),'width':20,'height':10,'timestamp':kwargs['timestamp'],
             'regions':[],'metadata':{}}
        pages.append(kwargs)
        return doc,{'failed_recognitions':0,'pages':2}
    monkeypatch.setattr(pipeline,'process_image',fake_process)
    reports=pipeline.process_manifest(manifest,tmp_path/'handoff','fixture',{},cache_dir=tmp_path/'cache')
    doc=json.loads((tmp_path/'handoff/page_0000/result.json').read_text())
    assert doc['metadata']['v1_handoff']['original_rgb'] is None
    assert doc['metadata']['v1_handoff']['coordinate_system']=='uncropped_st3d_composite_pixels'
    assert doc['timestamp']==12.75 and pages[0]['cache_dir']==tmp_path/'cache'
    report=json.loads((tmp_path/'handoff/manifest-report.json').read_text())
    assert 'single RGB correspondence not assumed' in report['scope']
    assert reports==[{'failed_recognitions':0,'pages':2}]


def test_routing_is_reviewable_and_does_not_override_detector_conflicts():
    items=[{'id':'equation','type':'unknown','subtype':'ocr_only_candidate','candidate_text':'x+3y=4','conflicts':[]},
           {'id':'prose','type':'unknown','subtype':'ocr_only_candidate','candidate_text':'Linear equations','conflicts':[]},
           {'id':'mixed','type':'unknown','subtype':'ocr_only_candidate','candidate_text':'if x = 1 then y','conflicts':[]},
           {'id':'conflict','type':'unknown','subtype':'ocr_only_candidate','candidate_text':'x=1',
            'conflicts':[{'kind':'multiple_layout_parents'}]},
           {'id':'layout','type':'other','subtype':'layout_label','candidate_text':'x=1','conflicts':[]}]
    before=deepcopy(items);routed=route_candidates(items)
    assert [r['type'] for r in routed]==['formula','text','unknown','unknown','other']
    assert routed[0]['routing']['calibrated_confidence'] is None and routed[0]['status']=='needs_review'
    assert routed[1]['review_required'] and 'unverified' in routed[0]['routing']['notice']
    assert items==before


@pytest.mark.parametrize('model',['PP-FormulaNet-S','PP-FormulaNet_plus-S','PP-FormulaNet-L','UniMERNet'])
def test_cli_model_selection_is_explicit_and_existing_default_is_s(monkeypatch,tmp_path,model):
    captured=[]
    def process(*args,**kwargs):captured.append((args,kwargs));return {},{'failed_recognitions':0,'pages':2}
    monkeypatch.setattr(cli,'process_image',process)
    args=['boardnotes.v25.cli','image',str(tmp_path/'source.png'),'--output',str(tmp_path/'out'),
          '--ocr-python','fixture-python']
    if model!='PP-FormulaNet-S':args+=['--formula-model',model]
    monkeypatch.setattr(sys,'argv',args);cli.main()
    configuration=captured[0][0][3]
    assert configuration['formula_model']==model
    assert configuration['formula_dir'].endswith(model+'_infer')
    assert captured[0][1]['split_formulas'] is False


def test_resource_supervisor_success_timeout_and_small_rss_budget(tmp_path):
    log=tmp_path/'bounded.log'
    with log.open('w') as stream:
        result=resources.run_bounded([sys.executable,'-c','print("fixture done")'],stdout=stream,
            stderr=subprocess.STDOUT,env=os.environ.copy(),cwd=tmp_path,timeout=5,max_rss_mib=128)
    assert result.returncode==0 and result.supervisor_peak_rss_mib>=0
    assert 'fixture done' in log.read_text()
    with log.open('w') as stream,pytest.raises(subprocess.TimeoutExpired):
        resources.run_bounded([sys.executable,'-c','import time; time.sleep(3)'],stdout=stream,
            stderr=subprocess.STDOUT,env=os.environ.copy(),cwd=tmp_path,timeout=.15,max_rss_mib=128)
    with log.open('w') as stream,pytest.raises(resources.ResourceLimitError,match='RSS budget'):
        resources.run_bounded([sys.executable,'-c','import time; time.sleep(3)'],stdout=stream,
            stderr=subprocess.STDOUT,env=os.environ.copy(),cwd=tmp_path,timeout=5,max_rss_mib=1)


def test_generated_ids_do_not_collide_with_reviewed_regions(monkeypatch,tmp_path):
    source,b0,_,_=images(tmp_path);calls=mocked_models(monkeypatch,b0);two_rows(monkeypatch,allow=True)
    specs=[{'id':'eq','type':'formula','bbox':[3,3,30,33]},
           {'id':'eq__row0','type':'other','bbox':[35,4,55,15]},
           {'id':'eq__assembled','type':'other','bbox':[35,20,55,35]}]
    doc,_=pipeline.process_image(source,tmp_path/'out','fixture',{'formula_model':'PP-FormulaNet-S'},
        b0=b0,regions=specs,split_formulas=True,structure_hints={'eq':'aligned'})
    ids=[r['id'] for r in doc['regions']]
    assert len(ids)==len(set(ids))
    assert 'eq__row0_1' in ids and 'eq__assembled_1' in ids
