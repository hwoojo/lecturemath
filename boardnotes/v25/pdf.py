"""V2.5 PDF: original raster, searchable source, rendered math, honest fallback."""
import json
import os
from pathlib import Path
import tempfile
from xml.sax.saxutils import escape

from boardnotes.v2.pdf import _clean, _font, _validate
from .rendering import render_latex


def render_pdf(document: dict, destination: Path, font_path: Path | None = None) -> int:
    """Render edited V2 JSON without any recognition model imports or inference.

    Each formula's rendering state is written to ``*.rendering.json``. Rendering
    is syntax/presentation only, never verification of mathematical correctness.
    Source pixels are never overwritten by recognition or rendered equations.
    """
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
        PageTemplate, Paragraph, Spacer,
    )

    source, width, height, regions = _validate(document)
    destination = Path(destination)
    if destination.suffix.lower() != '.pdf' or destination.resolve() == source.resolve():
        raise ValueError('destination must be a PDF separate from the source image')
    destination.parent.mkdir(parents=True, exist_ok=True)
    assets = destination.parent/(destination.stem+'_rendered')
    assets.mkdir(exist_ok=True)
    font_name, font = _font(font_path)
    image_size = landscape(A4) if width > height else A4
    body_size = A4
    body = ParagraphStyle('v25body', fontName=font_name, fontSize=9, leading=13,
                          spaceAfter=8, splitLongWords=True, wordWrap='CJK')
    heading = ParagraphStyle('v25heading', parent=body, fontSize=12, leading=16, spaceBefore=12)
    records = []
    pieces = [('heading', 'Recognition data and rendered mathematics'),
              ('body', 'Unicode text and literal LaTeX are searchable and copyable. '
               'Rendered mathematics verifies supported syntax only; it does not establish recognition or mathematical correctness.')]
    if not regions:
        pieces.append(('body', 'No recognition regions supplied. The complete original image is retained.'))
    with PILImage.open(source) as image:
        original = image.convert('RGB')
        for index, region in enumerate(regions):
            pieces.extend([
                ('heading', f'Region {_clean(region.get("id", ""))}'),
                ('body', f'Type: {region.get("type", "unknown")} | Recognition status: {region.get("status", "unknown")} | BBox [x1,y1,x2,y2]: {region["bbox"]}'),
                ('body', f'Model: {region.get("model")} | Confidence: {region.get("confidence")}'),
            ])
            if region.get('content') is not None:
                pieces.append(('body', 'Text: '+_clean(region['content'])))
            if region.get('error'):
                pieces.append(('body', 'Recognition error: '+_clean(region['error'])))
            show_source = region.get('type') in ('other', 'unknown') or region.get('status') in ('failed', 'empty', 'preserved', 'needs_review')
            if region.get('type') == 'formula':
                # A valid-looking rendering may still represent the wrong formula.
                # Keep the source crop next to every unverified prediction.
                show_source = True
                literal = region.get('latex')
                pieces.append(('body', 'LaTeX source: '+_clean(literal if literal is not None else '(no output)')))
                pieces.append(('body', 'Formula prediction requires human review, including when it renders successfully.'))
                result = render_latex(literal, assets/f'formula_{index:04}.png')
                records.append({'region_id':region.get('id'), 'bbox':region['bbox'],
                                'recognition_status':region.get('status'), 'latex':literal, **result})
                pieces.append(('body', f'Rendering status: {result["status"]} | Engine: {result["engine"]}'))
                if result['status'] == 'rendered':
                    pieces.append(('formula_image', result['path']))
                else:
                    pieces.append(('body', 'Rendering error: '+_clean(result.get('error'))))
                    show_source = True
            if show_source:
                crop = assets/f'source_region_{index:04}.png'
                original.crop(tuple(region['bbox'])).save(crop)
                pieces.append(('body', 'Preserved source crop (not replaced by a guessed expression):'))
                pieces.append(('source_image', str(crop)))
    report = {'format':'boardnotes-v25-rendering-1',
              'notice':'Recognition errors and rendering errors are separate; rendering success is not mathematical correctness.',
              'regions':records}
    destination.with_suffix('.rendering.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    image_header = 'Original board image (complete frame; no overlays or crop)'
    image_caption = f'Source dimensions: {width} x {height} | Timestamp (seconds): {document.get("timestamp")} | Recognition input: {document.get("metadata",{}).get("input_variant","unspecified")}'
    texts = [image_header,image_caption,'Recognition data | page ',*[text for kind,text in pieces if kind in ('heading','body')]]
    missing = sorted({ord(c) for text in texts for c in _clean(text) if c!='\n' and ord(c) not in font.face.charToGlyph})
    if missing:
        codes=', '.join(f'U+{code:04X}' for code in missing[:12])
        raise ValueError(f'PDF font does not cover all document glyphs ({codes}); provide a suitable font_path')

    class CountedDocument(BaseDocTemplate):
        page_count = 0
        def afterPage(self): self.page_count += 1

    def source_page(pdf, doc):
        pdf.saveState()
        pw,ph = image_size
        pdf.setFont(font_name,11);pdf.drawString(28,ph-28,image_header)
        pdf.setFont(font_name,9);pdf.drawString(28,ph-45,image_caption)
        scale=min((pw-56)/width,(ph-100)/height)
        dw,dh=width*scale,height*scale
        pdf.drawImage(ImageReader(str(source)),(pw-dw)/2,28+(ph-100-dh)/2,width=dw,height=dh,mask='auto')
        pdf.restoreState()

    def result_page(pdf,doc):
        pdf.saveState();pdf.setFont(font_name,8);pdf.setFillColor(colors.HexColor('#555555'))
        pdf.drawString(36,20,f'Recognition data | page {doc.page}');pdf.restoreState()

    fd,temporary=tempfile.mkstemp(prefix='.v25-pdf-',suffix='.pdf',dir=destination.parent)
    os.close(fd)
    try:
        doc=CountedDocument(temporary,title='Lecture board notes V2.5',author='BoardNotes',allowSplitting=True)
        doc.addPageTemplates([
            PageTemplate(id='source',pagesize=image_size,onPage=source_page,
                         frames=[Frame(28,28,image_size[0]-56,image_size[1]-56)]),
            PageTemplate(id='results',pagesize=body_size,onPage=result_page,
                         frames=[Frame(36,40,body_size[0]-72,body_size[1]-80,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]),
        ])
        story=[Spacer(1,1),NextPageTemplate('results'),PageBreak()]
        for kind,text in pieces:
            if kind in ('heading','body'):
                story.append(Paragraph(escape(_clean(text)).replace('\n','<br/>'),heading if kind=='heading' else body))
            else:
                with PILImage.open(text) as image:
                    iw,ih=image.size
                if kind=='formula_image':
                    # TeX uses 10pt math. 1.8x remains readable without making a
                    # short expression occupy the entire result-page width.
                    dw,dh=iw/180*72*1.8,ih/180*72*1.8
                else:
                    dw,dh=iw*.65,ih*.65
                scale=min(1, (body_size[0]-72)/dw, 420/dh)
                story.append(Image(text,width=dw*scale,height=dh*scale))
                story.append(Spacer(1,10))
        doc.build(story)
        os.replace(temporary,destination)
        return doc.page_count
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
