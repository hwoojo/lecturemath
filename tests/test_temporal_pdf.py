from pathlib import Path
import cv2
import numpy as np
import pytest
from pypdf import PdfReader
from boardnotes.config import Settings
from boardnotes.temporal import summarize
from boardnotes.pdf import generate_pdf, render_image


def mask_sequence():
    masks = []
    for index in range(32):
        mask = np.zeros((120, 240), np.uint8)
        # Original writing, temporarily occluded, then erased permanently.
        if index < 14 and index not in (4, 5):
            mask[20:35, 15:60] = 255
        # Persistent content must not disappear when the other region is erased.
        mask[70:85, 150:200] = 255
        if index >= 17:
            mask[25:45, 75:115] = 255
        masks.append(mask)
    return masks


def test_upstream_deletion_occlusion_and_pdf(tmp_path):
    masks = mask_sequence()
    data = ([i*1000 for i in range(len(masks))], list(range(len(masks))),
            [cv2.imencode(".png", mask)[1] for mask in masks])
    keyframes, stats = summarize(data, Settings(), tmp_path / "temporal")
    assert stats["raw_components"] > stats["unique_components"]
    assert len(keyframes) >= 2, stats
    # Old content is retained in an early summary despite absence at the end.
    assert any(np.all(img[20:35, 15:60] == 0) for img, _, _ in keyframes)
    assert np.all(keyframes[-1][0][25:45, 75:115] == 0)
    count = generate_pdf(keyframes, tmp_path / "notes.pdf")
    pdf = PdfReader(tmp_path / "notes.pdf")
    assert len(pdf.pages) == count >= 2
    assert all(len(page.images) == 1 for page in pdf.pages)
    assert "00:00:" in pdf.pages[0].extract_text()


def test_pdf_omits_blank_and_exact_duplicate_preserves_ratio(tmp_path):
    blank = np.full((100, 200), 255, np.uint8)
    ink = blank.copy()
    ink[20:70, 30:150] = 0
    count = generate_pdf([(blank, 0, 1), (ink, 2, 3), (ink.copy(), 4, 5)], tmp_path / "out.pdf", image_dir=tmp_path / "pages", style="binary")
    assert count == len(PdfReader(tmp_path / "out.pdf").pages) == 1
    image = cv2.imread(str(tmp_path / "pages/page_0001.png"), 0)
    assert np.count_nonzero(image == 0) == 50*120
    assert image.shape == (90, 160)
    with pytest.raises(ValueError, match="No stable handwriting"):
        generate_pdf([(blank, 0, 1)], tmp_path / "empty.pdf")


def test_smooth_export_preserves_small_marks_and_pdf_geometry(tmp_path):
    ink = np.full((64,96),255,np.uint8)
    ink[10,10] = 0  # decimal point must survive
    ink[15:45,20] = 0  # one-pixel thin stroke
    cv2.circle(ink,(55,30),12,0,1)  # preserve enclosed white space
    original = ink.copy()
    smooth = render_image(ink)
    assert np.array_equal(ink, original)
    assert np.array_equal(smooth[1::3,1::3], ink)
    assert np.any((smooth > 0) & (smooth < 255))
    assert smooth[30*3+1,55*3+1] == 255
    for style in ("binary","smooth"):
        generate_pdf([(ink,0,1)],tmp_path / f"{style}.pdf",image_dir=tmp_path / style,style=style)
    binary_pdf, smooth_pdf = [PdfReader(tmp_path / f"{style}.pdf") for style in ("binary","smooth")]
    assert list(binary_pdf.pages[0].mediabox) == list(smooth_pdf.pages[0].mediabox)
    transforms = lambda reader: [operands for operands, operator in reader.pages[0].get_contents().operations if operator == b"cm"]
    assert transforms(binary_pdf) == transforms(smooth_pdf)
    saved = cv2.imread(str(tmp_path / "smooth/page_0001.png"),0)
    assert np.array_equal(saved, np.asarray(smooth_pdf.pages[0].images[0].image))
    assert np.array_equal(render_image(ink,"binary"),ink)
    with pytest.raises(ValueError):
        Settings(pdf_style="invalid")


def test_empty_binary_sequence_returns_clear_no_content_error(tmp_path):
    masks = [np.zeros((64, 64), np.uint8) for _ in range(5)]
    data = ([i*1000 for i in range(5)], list(range(5)), [cv2.imencode(".png", m)[1] for m in masks])
    keyframes, _ = summarize(data, Settings(), tmp_path / "temporal")
    with pytest.raises(ValueError, match="No stable handwriting"):
        generate_pdf(keyframes, tmp_path / "empty.pdf")


def test_short_erase_rewrite_preserves_both_conflicting_shapes(tmp_path):
    # Original minimum segment length=3 cannot split six samples at deletion idx=2.
    # Both shapes are stable for 3 observations; their lifetimes do not overlap.
    masks = []
    for index in range(6):
        mask = np.zeros((64,128), np.uint8)
        if index < 3:
            mask[10:30,10:40] = 255
        else:
            mask[20:40,20:60] = 255
        masks.append(mask)
    data = ([i*1000 for i in range(6)], list(range(6)), [cv2.imencode(".png",m)[1] for m in masks])
    keyframes, stats = summarize(data, Settings(), tmp_path)
    assert stats["deletion_intervals"] == [(0,5)]
    assert stats["conflict_preservation_splits"] == 1
    assert stats["segments"] == [(0,2),(3,5)]
    assert np.all(keyframes[0][0][10:30,10:40] == 0)
    assert np.all(keyframes[1][0][20:40,20:60] == 0)
