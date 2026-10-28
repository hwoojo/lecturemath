"""Model-free PDF fixtures; strings below are not real OCR predictions."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader
import pytest

from boardnotes.v2.pdf import render_pdf


def fixture_document(tmp_path):
    source = np.full((96, 160, 3), 250, dtype=np.uint8)
    # Preserve graph-like colored content even if no region recognizes it.
    source[20:75, 80] = [0, 0, 255]
    source[60, 20:145] = [255, 0, 0]
    path = tmp_path/'original.png'
    Image.fromarray(source).save(path)
    doc = {'version':'2.0', 'source_image':str(path), 'width':160, 'height':96, 'timestamp':12.5,
           'regions':[
               {'id':'title', 'type':'text', 'bbox':[5,5,150,30], 'status':'ok',
                'content':'Fixture only: Linear equations <x> & y', 'latex':None,
                'model':'model-free-fixture', 'confidence':None, 'error':None},
               {'id':'equation', 'type':'formula', 'bbox':[10,32,150,60], 'status':'ok',
                'content':None, 'latex':r'\frac{x+1}{2}=y', 'model':'model-free-fixture',
                'confidence':None, 'error':None},
               {'id':'graph', 'type':'other', 'bbox':[10,60,150,90], 'status':'preserved',
                'content':None, 'latex':None, 'model':None, 'confidence':None, 'error':None},
               {'id':'failed', 'type':'formula', 'bbox':[10,30,25,45], 'status':'failed',
                'content':None, 'latex':None, 'model':'model-free-fixture', 'confidence':None,
                'error':'controlled fixture error'},
           ], 'metadata':{'test_fixture':True}}
    return doc, source


def test_source_retained_and_searchable_literal_latex(tmp_path):
    document, source = fixture_document(tmp_path)
    original = deepcopy(document)
    path = tmp_path/'notes.pdf'
    assert render_pdf(document, path) == 2
    assert document == original
    reader = PdfReader(path)
    assert len(reader.pages) == 2
    assert len(reader.pages[0].images) == 1
    np.testing.assert_array_equal(np.array(reader.pages[0].images[0].image.convert('RGB')), source)
    text = reader.pages[1].extract_text()
    assert 'Fixture only: Linear equations <x> & y' in text
    assert r'\frac{x+1}{2}=y' in text
    assert 'Status: failed' in text and 'controlled fixture error' in text
    assert 'Status: preserved' in text and 'graph' in text
    assert '[10, 32, 150, 60]' in text


def test_json_edit_regenerates_without_model_and_paginates(tmp_path):
    document, _ = fixture_document(tmp_path)
    saved = tmp_path/'document.json'
    saved.write_text(json.dumps(document))
    edited = json.loads(saved.read_text())
    edited['regions'][0]['content'] = 'Edited searchable text\n' + ('long-literal-token_'*40+'\n')*60
    pdf = tmp_path/'edited.pdf'
    pages = render_pdf(edited, pdf)
    reader = PdfReader(pdf)
    assert pages == len(reader.pages) > 2
    assert 'Edited searchable text' in reader.pages[1].extract_text()
    assert not any(p.images for p in reader.pages[1:])


def test_empty_results_and_unsupported_glyph_are_explicit(tmp_path):
    document, _ = fixture_document(tmp_path)
    document['regions'] = []
    pdf = tmp_path/'empty.pdf'
    assert render_pdf(document, pdf) == 2
    assert 'No recognition regions supplied' in PdfReader(pdf).pages[1].extract_text()
    previous = pdf.read_bytes()
    document['regions'] = [{'id':'ko', 'bbox':[0,0,20,20], 'content':'한글', 'type':'text', 'status':'ok'}]
    with pytest.raises(ValueError, match='font_path'):
        render_pdf(document, pdf)
    assert pdf.read_bytes() == previous
    assert not list(tmp_path.glob('.v2-pdf-*'))


@pytest.mark.parametrize('box', [[0,0,200,20], [10,10,5,20], [0,0,0,1], [0.5,0,10,20]])
def test_invalid_coordinates_do_not_write_pdf(tmp_path, box):
    document, _ = fixture_document(tmp_path)
    document['regions'][0]['bbox'] = box
    with pytest.raises(ValueError, match='bbox'):
        render_pdf(document, tmp_path/'bad.pdf')
    assert not (tmp_path/'bad.pdf').exists()


def test_document_image_dimensions_must_match(tmp_path):
    document, _ = fixture_document(tmp_path)
    document['width'] = 161
    with pytest.raises(ValueError, match='dimensions'):
        render_pdf(document, tmp_path/'bad.pdf')
