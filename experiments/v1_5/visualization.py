import cv2
import numpy as np


def contact_sheet(original, panels, path, crop=None, zoom=1):
    """One panel row per method avoids retaining multiple full-size RGB panels."""
    h, w = original.shape[:2]
    x, y, cw, ch = crop or (0, 0, w, h)
    rows = []
    for label, img in [('Original', original), *panels.items()]:
        part = img[y:y+ch, x:x+cw]
        if part.ndim == 2:
            part = cv2.cvtColor(part, cv2.COLOR_GRAY2BGR)
        part = cv2.resize(part, (cw*zoom, ch*zoom), interpolation=cv2.INTER_NEAREST)
        row = cv2.copyMakeBorder(part, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(row, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1)
        rows.append(row)
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise OSError(path)
