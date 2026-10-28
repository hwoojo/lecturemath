"""Render cached real-frame comparisons with algorithm names, without inference."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .real_report import MATH_CROPS

# File identifiers remain internal. All visible labels use actual technique names.
METHODS = [
    ('Original RGB', 'Unprocessed lecture frame', 'results', 'original'),
    ('Reference annotation', 'Public pixel ground truth', 'results', 'ground_truth'),
    ('FCN-LectureNet V34', 'Original 8-bit threshold = 128', 'results', 'B0'),
    ('Otsu global thresholding', 'Grayscale; Gaussian blur disabled', 'results', 'B1_blur0'),
    ('Sauvola local thresholding', 'Window = 25; k = 0.2; CLAHE disabled', 'results', 'B2_w25_k0.2_c0'),
    ('LectureNet-guided Sauvola', 'Window = 25; k = 0.2; dilation radius = 2', 'results', 'B3_w25_k0.2_c0'),
    ('LectureNet-guided Sauvola + CLAHE', 'Window = 25; k = 0.2; dilation radius = 2', 'clahe', 'B3_w25_k0.2_c1'),
    ('FCN-LectureNet V34: threshold sweep', 'Writing probability threshold = 0.35', 'results', 'B4_t0.35'),
    ('FCN-LectureNet V34: threshold sweep', 'Writing probability threshold = 0.65', 'results', 'B4_t0.65'),
]


def image_reader(image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('cannot encode comparison crop')
    return ImageReader(BytesIO(encoded.tobytes()))


def render(root, output):
    manifest = json.loads((root/'source_manifest.json').read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    font_dir = Path('/usr/share/fonts/truetype/dejavu')
    if (font_dir/'DejaVuSans.ttf').exists() and (font_dir/'DejaVuSans-Bold.ttf').exists():
        pdfmetrics.registerFont(TTFont('ComparisonSans',str(font_dir/'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('ComparisonSansBold',str(font_dir/'DejaVuSans-Bold.ttf')))
        regular,bold = 'ComparisonSans','ComparisonSansBold'
    else:
        regular,bold = 'Helvetica','Helvetica-Bold'
    pdf = canvas.Canvas(str(output), pagesize=landscape(A4))
    pdf.setTitle('Actual lecture frames: FCN-LectureNet V34 and binarization techniques')
    pdf.setAuthor('LectureMath V1.5 experiments')
    width, height = landscape(A4)
    pages = 0
    for sample in manifest['samples']:
        name = sample['id']
        paths = [root/folder/'1080p'/name/(filename+'.png') for _, _, folder, filename in METHODS]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        # One comparison page at identical source coordinates, followed by full-frame pages.
        pdf.setFont(bold, 14)
        pdf.drawString(18, height-25, 'Actual lecture: technique comparison')
        pdf.setFont(regular, 9)
        pdf.drawString(18, height-42, f'{name} | Native 1920 x 1080 | Source crop {MATH_CROPS[name]}')
        pdf.drawString(18, height-56, 'Same crop and scale throughout; nearest-neighbor 2x raster; no display contrast correction.')
        cell_w = (width-36)/3
        cell_h = (height-100)/3
        x,y,cw,ch = MATH_CROPS[name]
        for i, ((title, setting, _, _), path) in enumerate(zip(METHODS, paths)):
            left = 18+(i%3)*cell_w
            top = height-72-(i//3)*cell_h
            pdf.setFont(bold, 8)
            pdf.drawString(left+3, top, title)
            pdf.setFont(regular, 6.8)
            pdf.drawString(left+3, top-12, setting)
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (1080,1920):
                raise ValueError(f'invalid native frame: {path}')
            crop = image[y:y+ch,x:x+cw]
            crop = cv2.resize(crop, (cw*2,ch*2), interpolation=cv2.INTER_NEAREST)
            draw_w = cell_w-8
            draw_h = draw_w*ch/cw
            if draw_h > cell_h-30:
                draw_h = cell_h-30
                draw_w = draw_h*cw/ch
            pdf.drawImage(image_reader(crop),left+3,top-24-draw_h,width=draw_w,height=draw_h)
        pdf.setFont(regular, 7)
        pdf.drawString(18, 14, 'Only FCN-LectureNet is a trained neural network; guided Sauvola is a refinement, not a separately trained model.')
        pdf.showPage()
        pages += 1
        for (title, setting, _, _), path in zip(METHODS, paths):
            pdf.setFont(bold, 16)
            pdf.drawString(20, height-27, title)
            pdf.setFont(regular, 10)
            pdf.drawString(20, height-45, setting)
            pdf.setFont(regular, 9)
            pdf.drawString(20, height-61, f'Real AccessMath2015 | {name} | Native 1920 x 1080 | No smoothing')
            draw_w = width-40
            draw_h = draw_w*1080/1920
            pdf.drawImage(str(path),20,(height-draw_h)/2-15,width=draw_w,height=draw_h)
            pdf.setFont(regular, 7)
            pdf.drawString(20, 15, 'Same full-frame placement on every page. Three frames only; mathematical OCR accuracy was not evaluated.')
            pdf.showPage()
            pages += 1
    pdf.save()
    print(f'{pages} pages: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('benchmark-output/v1_5-real'))
    parser.add_argument('--output',type=Path,default=Path('docs/assets/v1_5_real/comparison_named.pdf'))
    args = parser.parse_args()
    render(args.input,args.output)


if __name__ == '__main__':
    main()
