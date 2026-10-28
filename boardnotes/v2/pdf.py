"""PDF projection of editable V2 JSON; never imports or runs recognition models.

The first page keeps the complete source raster. Subsequent pages contain actual
Unicode text and literal LaTeX, not an OCR text layer or rendered mathematics.
"""
from hashlib import sha256
import math
import os
from pathlib import Path
import tempfile
from xml.sax.saxutils import escape


def _font(font_path=None):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    path = Path(font_path) if font_path is not None else Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not path.is_file():
        raise ValueError('A Unicode TrueType PDF font is required; provide font_path to an installed .ttf file')
    name = 'BoardNotesV2_' + sha256(path.read_bytes()).hexdigest()[:16]
    if name not in pdfmetrics.getRegisteredFontNames():
        try:
            pdfmetrics.registerFont(TTFont(name, str(path)))
        except Exception as error:
            raise ValueError(f'Cannot load PDF TrueType font: {path}') from error
    return name, pdfmetrics.getFont(name)


def _clean(value):
    return str(value).replace('\r\n', '\n').replace('\r', '\n').replace('\t', '    ')


def _validate(document):
    from PIL import Image

    source = Path(document['source_image'])
    with Image.open(source) as image:
        width, height = image.size
    if (document.get('width'), document.get('height')) != (width, height):
        raise ValueError('Document dimensions must match the complete source image')
    timestamp = document.get('timestamp')
    if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                                  or not math.isfinite(timestamp) or timestamp < 0):
        raise ValueError('timestamp must be finite nonnegative seconds or null')
    regions = document.get('regions', [])
    if not isinstance(regions, list):
        raise ValueError('regions must be a list')
    for region in regions:
        box = region.get('bbox')
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) for v in box)):
            raise ValueError('bbox must contain four integer source coordinates')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('bbox must be nonempty and inside the source image')
    return source, width, height, regions


def render_pdf(document: dict, destination: Path, font_path: Path | None = None) -> int:
    """Render complete source + searchable result pages from JSON, atomically.

    Unsupported glyphs fail before writing a PDF. In particular, the default
    DejaVu font does not support Korean: supply a Korean-capable TrueType font.
    Literal LaTeX is searchable but is deliberately not mathematical typesetting.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate,
        Paragraph, Spacer,
    )

    source, width, height, regions = _validate(document)
    destination = Path(destination)
    if destination.suffix.lower() != '.pdf' or destination.resolve() == source.resolve():
        raise ValueError('destination must be a PDF separate from the source image')
    font_name, font = _font(font_path)
    image_header = 'Original board image (complete frame; no overlays or crop)'
    image_caption = f'Source dimensions: {width} x {height} | Timestamp (seconds): {document.get("timestamp")} | Recognition input: {document.get("metadata", {}).get("input_variant", "unspecified")}'
    paragraphs = [('heading', 'Recognition data'),
                  ('body', 'Text and literal LaTeX below are searchable and copyable. LaTeX is not rendered mathematics.')]
    if not regions:
        paragraphs.append(('body', 'No recognition regions supplied. The source image remains preserved.'))
    for region in regions:
        paragraphs.append(('heading', f'Region {_clean(region.get("id", ""))}'))
        paragraphs.append(('body', f'Type: {region.get("type", "unknown")} | Status: {region.get("status", "unknown")} | BBox [x1,y1,x2,y2]: {region["bbox"]}'))
        paragraphs.append(('body', f'Model: {region.get("model")} | Confidence: {region.get("confidence")}'))
        if region.get('content') is not None:
            paragraphs.append(('body', 'Text: ' + _clean(region['content'])))
        if region.get('type') == 'formula':
            paragraphs.append(('body', 'Formula prediction requires human review; nonempty output does not establish mathematical correctness.'))
        if region.get('latex') is not None:
            paragraphs.append(('body', 'LaTeX source: ' + _clean(region['latex'])))
        if region.get('error'):
            paragraphs.append(('body', 'Error: ' + _clean(region['error'])))
        if region.get('status') in ('preserved', 'failed', 'empty', 'needs_review'):
            paragraphs.append(('body', 'Review required; the original visual content is retained on the image page.'))
    texts = [image_header, image_caption, 'Recognition data | page ', *[_clean(text) for _, text in paragraphs]]
    missing = sorted({ord(char) for text in texts for char in text if char not in '\n' and ord(char) not in font.face.charToGlyph})
    if missing:
        sample = ', '.join(f'U+{code:04X}' for code in missing[:12])
        raise ValueError(f'PDF font does not cover all document glyphs ({sample}); provide a suitable font_path. JSON content has not been replaced.')

    image_size = landscape(A4) if width > height else A4
    body_size = A4
    body = ParagraphStyle('v2body', fontName=font_name, fontSize=9, leading=13,
                          spaceAfter=8, splitLongWords=True, wordWrap='CJK')
    heading = ParagraphStyle('v2heading', parent=body, fontSize=12, leading=16, spaceBefore=12)

    class CountedDocument(BaseDocTemplate):
        page_count = 0

        def afterPage(self):
            self.page_count += 1

    def original_page(pdf, doc):
        pdf.saveState()
        pw, ph = image_size
        pdf.setFont(font_name, 11)
        pdf.drawString(28, ph-28, image_header)
        pdf.setFont(font_name, 9)
        pdf.drawString(28, ph-45, image_caption)
        scale = min((pw-56)/width, (ph-100)/height)
        dw, dh = width*scale, height*scale
        pdf.drawImage(ImageReader(str(source)), (pw-dw)/2, 28+(ph-100-dh)/2,
                      width=dw, height=dh, mask='auto')
        pdf.restoreState()

    def results_page(pdf, doc):
        pdf.saveState()
        pdf.setFont(font_name, 8)
        pdf.setFillColor(colors.HexColor('#555555'))
        pdf.drawString(36, 20, f'Recognition data | page {doc.page}')
        pdf.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.v2-pdf-', suffix='.pdf', dir=destination.parent)
    os.close(fd)
    try:
        doc = CountedDocument(temporary, title='Lecture board notes V2', author='BoardNotes',
                              allowSplitting=True)
        doc.addPageTemplates([
            PageTemplate(id='source', pagesize=image_size, onPage=original_page,
                         frames=[Frame(28, 28, image_size[0]-56, image_size[1]-56)]),
            PageTemplate(id='results', pagesize=body_size, onPage=results_page,
                         frames=[Frame(36, 40, body_size[0]-72, body_size[1]-80,
                                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]),
        ])
        story = [Spacer(1, 1), NextPageTemplate('results'), PageBreak()]
        for kind, text in paragraphs:
            markup = escape(_clean(text)).replace('\n', '<br/>')
            story.append(Paragraph(markup, heading if kind == 'heading' else body))
        doc.build(story)
        os.replace(temporary, destination)
        return doc.page_count
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
