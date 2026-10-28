import cv2
import numpy as np


def evaluate(prediction, roi, truth=None):
    if prediction.shape != roi.shape or not roi.any():
        raise ValueError('invalid or empty evaluation ROI')
    foreground = prediction & roi
    n, _, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    result = {'foreground_ratio': float(foreground.sum()/roi.sum()), 'components': n-1,
              'component_area_quantiles': np.quantile(sizes, [0, .25, .5, .75, 1]).tolist() if sizes.size else []}
    if truth is not None:
        if truth.shape != prediction.shape:
            raise ValueError('GT coordinates/resolution must match input')
        tp = int((prediction & truth & roi).sum())
        fp = int((prediction & ~truth & roi).sum())
        fn = int((~prediction & truth & roi).sum())
        # Explicit zero division convention, including empty GT/prediction.
        result.update(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0.,
                      recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                      iou=tp/(tp+fp+fn) if tp+fp+fn else 0.)
    return result
