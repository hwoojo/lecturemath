import hashlib
from pathlib import Path
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image


def timestamp(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def render_image(image, style="smooth"):
    """Smooth display edges only; preserve original pixels at 3x pixel centers.

    This does not remove components, close holes or reconstruct lost strokes.
    CC tracking and duplicate detection always use the original binary image.
    """
    if style == "binary":
        return image
    if style != "smooth":
        raise ValueError("PDF style must be smooth or binary")
    return cv2.resize(image, (image.shape[1]*3, image.shape[0]*3), interpolation=cv2.INTER_CUBIC)


def generate_pdf(keyframes, destination, timestamps=True, image_dir=None, style="smooth"):
    """Upstream keyframes have black ink on white; no stroke erosion/OCR."""
    if style not in ("smooth", "binary"):
        raise ValueError("PDF style must be smooth or binary")
    pages, seen = [], set()
    for image, start, end in sorted(keyframes, key=lambda item: item[1]):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = np.where(gray < 128, 0, 255).astype(np.uint8)
        if not np.any(gray == 0):
            continue
        digest = hashlib.sha256(str(gray.shape).encode() + gray.tobytes()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        # Only trim exterior whitespace, preserving every remaining ink pixel.
        points = cv2.findNonZero(255 - gray)
        x, y, w, h = cv2.boundingRect(points)
        pad = 20
        cropped = cv2.copyMakeBorder(gray[y:y+h, x:x+w], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        pages.append((cropped, start, end))
    if not pages:
        raise ValueError("No stable handwriting was extracted; PDF has no nonempty pages. Check board visibility, model, sampling and stability settings.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(destination), pagesize=A4)
    pdf.setTitle("Lecture board notes")
    for index, (image, start, end) in enumerate(pages):
        height, width = image.shape
        page_w, page_h = landscape(A4) if width > height else A4
        pdf.setPageSize((page_w, page_h))
        margin, footer = 28, 24 if timestamps else 0
        scale = min((page_w - 2 * margin) / width, (page_h - 2 * margin - footer) / height)
        dw, dh = width * scale, height * scale
        rendered = render_image(image, style)
        pdf.drawImage(ImageReader(Image.fromarray(rendered)), (page_w-dw)/2,
                      margin + footer + (page_h-2*margin-footer-dh)/2, width=dw, height=dh)
        if timestamps:
            pdf.setFont("Helvetica", 9)
            pdf.drawCentredString(page_w/2, 22, f"{timestamp(start)} - {timestamp(end)}  |  {index+1}")
        if image_dir:
            Path(image_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(image_dir) / f"page_{index+1:04}.png"), rendered)
        pdf.showPage()
    pdf.save()
    return len(pages)
