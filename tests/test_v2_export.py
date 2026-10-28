"""V1->V2 export contracts using synthetic rasters and an explicit fake model.

These fixtures verify preservation/integration only, not actual V34/ST3D quality.
"""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from boardnotes.config import Settings
from boardnotes.v2.export import export_keyframes


def binary_fixture(height=90,width=160):
    image=np.full((height,width),255,np.uint8)
    image[0,0]=0
    image[-1,-1]=0
    image[15:55,30]=0
    image[40,20:100]=0
    return image


def test_export_preserves_uncropped_pixels_seconds_and_coordinate_notice(tmp_path):
    first=binary_fixture()
    second=cv2.cvtColor(binary_fixture(180,320),cv2.COLOR_GRAY2BGR)
    keyframes=[(first,1.25,3.5),(second,4.,9.125)]
    before=[image.copy() for image,_,_ in keyframes]
    output=tmp_path/'handoff'
    result=export_keyframes(keyframes,{'width':1920,'height':1080},output)
    assert result['format']=='boardnotes-v2-inputs-1'
    assert result==json.loads((output/'manifest.json').read_text())
    assert 'Temporal composites' in result['notice']
    assert 'no single matching RGB' in result['notice']
    assert 'not proof of RGB alignment' in result['notice']
    assert 'before PDF crop/smoothing' in result['notice']
    for i,page in enumerate(result['pages']):
        original,start,end=keyframes[i]
        np.testing.assert_array_equal(original,before[i])
        expected=original if original.ndim==2 else original[:,:,0]
        saved=cv2.imread(page['image'],0)
        np.testing.assert_array_equal(saved,expected)
        assert saved.shape==expected.shape
        assert saved[0,0]==saved[-1,-1]==0
        assert set(np.unique(saved))<={0,255}
        assert page['width']==expected.shape[1] and page['height']==expected.shape[0]
        assert page['id']==i and page['variant']=='b0'
        assert page['interval_seconds']==[start,end]
        assert page['timestamp']==end  # source seconds, not V1 sampled milliseconds
        assert page['coordinate_system']=='uncropped_st3d_composite_pixels'
        assert page['original_rgb'] is None
        assert page['source_video_dimensions']==[1920,1080]
        assert page['scale_to_video']==([12.,12.] if i==0 else [6.,6.])


@pytest.mark.parametrize('bad', [np.full((10,10),127,np.uint8),
                                np.full((10,10),1,np.uint8),
                                np.zeros((10,10),np.float32)])
def test_export_rejects_nonbinary_or_wrong_dtype(tmp_path,bad):
    with pytest.raises(ValueError,match='binary'):
        export_keyframes([(bad,0.,1.)],{'width':1920,'height':1080},tmp_path/'bad')
    assert not (tmp_path/'bad/manifest.json').exists()


@pytest.mark.parametrize('start,end', [(-1.,1.),(3.,2.),(float('nan'),1.),
                                       (0.,float('inf')),(0.,float('nan'))])
def test_export_rejects_invalid_intervals(tmp_path,start,end):
    with pytest.raises(ValueError,match='interval'):
        export_keyframes([(binary_fixture(),start,end)],{'width':1920,'height':1080},tmp_path/'bad-time')
    assert not (tmp_path/'bad-time/manifest.json').exists()


def test_v1_opt_in_export_preserves_default_pdf_invocation(monkeypatch,tmp_path):
    # All V1 inference/temporal/PDF dependencies below are explicit model-free fakes.
    from boardnotes import pipeline
    checkpoint=tmp_path/'fixture.dat'
    checkpoint.write_bytes(b'Not a checkpoint; fake model fixture only')
    frame=np.full((90,160,3),200,np.uint8)
    keyframe=binary_fixture()
    pdf_calls=[]
    temporal_calls=[]
    model_calls=[]

    class FakeLectureNet:
        def __init__(self,settings):
            model_calls.append(settings.export_v2_inputs)

        def extract(self,image):
            assert image.shape==frame.shape
            return 255-keyframe  # V1 ST3D convention: white writing on black

    def fake_frames(video,meta,settings):
        yield 0,0.,frame.copy()
        yield 10,1000.,frame.copy()

    def fake_summarize(data,settings,folder,progress):
        times,indices,compressed=data
        assert times==[0.,1000.] and indices==[0,10]
        for encoded in compressed:
            np.testing.assert_array_equal(cv2.imdecode(encoded,0),255-keyframe)
        temporal_calls.append(settings.export_v2_inputs)
        return [(keyframe.copy(),0.,1.)],{'fixture_temporal':True}

    def fake_pdf(keyframes,destination,timestamps,image_dir,style):
        snapshot=[(image.copy(),start,end) for image,start,end in keyframes]
        pdf_calls.append({'keyframes':snapshot,'timestamps':timestamps,'image_dir':image_dir,'style':style})
        Path(destination).write_bytes(b'%PDF-FAKE-model-free-test-only')
        return 1

    monkeypatch.setattr(pipeline,'LectureNet',FakeLectureNet)
    monkeypatch.setattr(pipeline,'metadata',lambda video,settings:{'width':1920,'height':1080,'estimated_samples':2})
    monkeypatch.setattr(pipeline,'frames',fake_frames)
    monkeypatch.setattr(pipeline,'summarize',fake_summarize)
    monkeypatch.setattr(pipeline,'generate_pdf',fake_pdf)
    baseline=Settings(checkpoint=checkpoint,pdf_style='binary',timestamps=False)
    assert baseline.export_v2_inputs is False
    baseline_report=pipeline.process_video(tmp_path/'not-a-real-video.mp4',tmp_path/'default',baseline)
    exported=Settings(checkpoint=checkpoint,pdf_style='binary',timestamps=False,export_v2_inputs=True)
    export_report=pipeline.process_video(tmp_path/'not-a-real-video.mp4',tmp_path/'opt-in',exported)
    assert not (tmp_path/'default/v2_inputs').exists()
    manifest=json.loads((tmp_path/'opt-in/v2_inputs/manifest.json').read_text())
    assert len(manifest['pages'])==1
    np.testing.assert_array_equal(cv2.imread(manifest['pages'][0]['image'],0),keyframe)
    assert manifest['pages'][0]['interval_seconds']==[0.,1.]
    assert model_calls==temporal_calls==[False,True]
    assert len(pdf_calls)==2
    for call in pdf_calls:
        assert call['style']=='binary' and call['timestamps'] is False and call['image_dir'] is None
        assert len(call['keyframes'])==1
        np.testing.assert_array_equal(call['keyframes'][0][0],keyframe)
        assert call['keyframes'][0][1:]==(0.,1.)
    assert baseline_report['pages']==export_report['pages']==1
    assert baseline_report['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert baseline_report['checkpoint_sha256']==export_report['checkpoint_sha256']
    assert baseline_report['fixture_temporal'] is export_report['fixture_temporal'] is True


def test_v1_cli_export_flag_is_explicit_opt_in(monkeypatch,tmp_path,capsys):
    from boardnotes import cli
    fake_video=tmp_path/'fixture.mp4'
    fake_video.write_bytes(b'Not a video; CLI argument fixture only')
    observed=[]

    def fake_process(video,output,settings,progress):
        observed.append(settings.export_v2_inputs)
        return {'fixture':True}

    monkeypatch.setattr(cli,'process_video',fake_process)
    for i,extra in enumerate([[],['--export-v2-inputs']]):
        monkeypatch.setattr(sys,'argv',['boardnotes',str(fake_video),'--output-dir',str(tmp_path/f'cli-{i}'),*extra])
        cli.main()
    assert observed==[False,True]
    assert capsys.readouterr().out.count('"fixture": true')==2
