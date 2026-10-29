"""Renderer/PDF contracts use synthetic data, never model accuracy claims."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from pypdf import PdfReader
import pytest

from boardnotes.v25 import rendering
from boardnotes.v25.pdf import render_pdf


@pytest.mark.parametrize('value', [r'\input{/etc/passwd}',r'\write18{touch /tmp/forbidden}',
                                  r'\csname input\endcsname{a}',r'\href{https://example.com}{a}',
                                  r'\begin{document}x\end{document}',r'^^5cinput{a}',
                                  r'\newcommand{\a}{b}',r'$x$',r'x%comment',r'\frac{1}{2',
                                  r'\begin{cases}a\end{aligned}', 'a'*6001])
def test_unsupported_input_never_reaches_tex(monkeypatch,tmp_path,value):
    monkeypatch.setattr(rendering,'engine_info',lambda:pytest.fail('unsafe input reached renderer'))
    output=tmp_path/'formula.png';output.write_bytes(b'existing-output')
    result=rendering.render_latex(value,output)
    assert result['status']=='unsupported' and result['path'] is None and result['error']
    assert output.read_bytes()==b'existing-output'


def test_supported_math_validation_and_unknown_renderer(monkeypatch,tmp_path):
    for value in [r'\frac{a_1}{\sqrt{x}}',r'\begin{cases}x+y=1\\x-y=2\end{cases}',
                  r'\left.\begin{aligned}3x&=1\\x+2y&=3\end{aligned}\right\}',
                  r'\begin{pmatrix}1&2\\3&4\end{pmatrix}',r'\text{Area}=\pi r^2']:
        assert rendering.validate_latex(value) is None
    monkeypatch.setattr(rendering,'engine_info',lambda:None)
    result=rendering.render_latex('x+1',tmp_path/'missing.png')
    assert result['status']=='unsupported' and 'required' in result['error']
    with pytest.raises(ValueError,match='PNG'):
        rendering.render_latex('x',tmp_path/'wrong.svg')


def test_timeout_is_explicit_and_preserves_existing_output(monkeypatch,tmp_path):
    monkeypatch.setattr(rendering,'engine_info',lambda:{'tex':'fixture-tex','raster':'fixture-raster','version':'fixture'})
    def timeout(*args,**kwargs):
        assert '-no-shell-escape' in args[0]
        assert kwargs['env']['openin_any']=='p'
        assert kwargs['env']['openout_any']=='p'
        assert kwargs['timeout']<=12
        raise subprocess.TimeoutExpired(args[0],kwargs['timeout'])
    monkeypatch.setattr(rendering.subprocess,'run',timeout)
    output=tmp_path/'math.png';output.write_bytes(b'previous')
    result=rendering.render_latex('x+1',output)
    assert result['status']=='failed' and 'TimeoutExpired' in result['error']
    assert output.read_bytes()==b'previous'


@pytest.mark.parametrize('latex',[r'x+1=2',r'\frac{27}{19}',
                                 r'\left.\begin{aligned}3x&=1\\x+2y&=3\end{aligned}\right\}',
                                 r'\begin{cases}x+y=1\\x-y=2\end{cases}',
                                 r'\begin{bmatrix}1&2\\3&4\end{bmatrix}'])
def test_real_toolchain_and_cache(tmp_path,latex):
    if rendering.engine_info() is None: pytest.skip('TeX toolchain not installed')
    path=tmp_path/'rendered.png'
    result=rendering.render_latex(latex,path)
    assert result['status']=='rendered',result
    with Image.open(path) as image:
        assert image.width>0 and image.height>0
        assert np.any(np.asarray(image.convert('L'))<128)
    assert result['semantic_correctness']=='not evaluated'
    again=rendering.render_latex(latex,path)
    assert again['cache_hit'] and again['png_sha256']==result['png_sha256']


def test_compiler_rejects_balanced_but_incomplete_fraction(tmp_path):
    if rendering.engine_info() is None: pytest.skip('TeX toolchain not installed')
    result=rendering.render_latex(r'\frac{1}',tmp_path/'syntax.png')
    assert result['status']=='failed' and 'compilation failed' in result['error']


def fixture_document(tmp_path):
    pixels=np.random.default_rng(7).integers(0,256,(80,120,3),dtype=np.uint8)
    source=tmp_path/'source.png';Image.fromarray(pixels).save(source)
    return {'version':'2.0','source_image':str(source),'width':120,'height':80,'timestamp':None,
            'metadata':{'input_variant':'fixture'},'regions':[
              {'id':'text','type':'text','bbox':[0,0,30,20],'status':'ok','content':'Fixture α & <tag>',
               'latex':None,'model':'fixture only','confidence':None},
              {'id':'formula','type':'formula','bbox':[30,10,110,50],'status':'ok','content':None,
               'latex':r'\frac{1}{2}','model':'fixture only','confidence':None},
              {'id':'graph','type':'other','bbox':[5,55,60,78],'status':'preserved','content':None,
               'latex':None,'model':None,'confidence':None}]},pixels


def test_pdf_preserves_raster_searchable_source_and_rendering_report(tmp_path):
    if rendering.engine_info() is None: pytest.skip('TeX toolchain not installed')
    document,pixels=fixture_document(tmp_path);before=deepcopy(document)
    path=tmp_path/'notes.pdf'
    count=render_pdf(document,path)
    reader=PdfReader(path)
    assert count==len(reader.pages)>=2
    np.testing.assert_array_equal(np.asarray(reader.pages[0].images[0].image.convert('RGB')),pixels)
    text='\n'.join(page.extract_text() for page in reader.pages)
    assert 'Fixture α & <tag>' in text and r'\frac{1}{2}' in text
    assert 'does not establish' in text and 'Preserved source crop' in text
    record=json.loads(path.with_suffix('.rendering.json').read_text())['regions'][0]
    assert record['status']=='rendered' and record['recognition_status']=='ok'
    assert Path(record['path']).is_file()
    assert document==before
    assert 'paddle' not in sys.modules and 'torch' not in sys.modules


def test_pdf_failed_render_preserves_source_crop_and_literal(tmp_path):
    document,pixels=fixture_document(tmp_path)
    document['regions'][1]['latex']=r'\input{/etc/passwd}'
    path=tmp_path/'failed.pdf'
    render_pdf(document,path)
    text='\n'.join(p.extract_text() for p in PdfReader(path).pages)
    assert r'\input{/etc/passwd}' in text and 'Rendering status: unsupported' in text
    records=json.loads(path.with_suffix('.rendering.json').read_text())['regions']
    assert records[0]['path'] is None
    crop=Image.open(tmp_path/'failed_rendered/source_region_0001.png')
    np.testing.assert_array_equal(np.asarray(crop),pixels[10:50,30:110])
    assert not Path('/tmp/forbidden').exists()
