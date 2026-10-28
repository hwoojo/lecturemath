"""CPU binarizers. Boolean True always denotes writing foreground."""
import cv2
import numpy as np


def gray(image, polarity='dark', clahe=False):
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if polarity == 'light':
        g = 255 - g
    if clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return g


def otsu(image, polarity='dark', blur=0):
    g = gray(image, polarity)
    if blur:
        g = cv2.GaussianBlur(g, (blur, blur), 0)
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0


def sauvola(image, polarity='dark', window=25, k=.2, clahe=False):
    """Sauvola with R=128 and replicated borders; no scikit-image dependency."""
    if window < 3 or window % 2 != 1 or not np.isfinite(k):
        raise ValueError('window must be odd >=3; k must be finite')
    g = gray(image, polarity, clahe).astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window), borderType=cv2.BORDER_REPLICATE)
    second = cv2.boxFilter(g*g, -1, (window, window), borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(second - mean*mean, 0))
    threshold = mean * (1 + k * (std / 128.0 - 1))
    return g < threshold


def hybrid(image, baseline, polarity='dark', window=25, k=.2, radius=2, clahe=False):
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    region = cv2.dilate(baseline.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
    return baseline | (region & sauvola(image, polarity, window, k, clahe))


def png(mask):
    return np.where(mask, 0, 255).astype(np.uint8)
