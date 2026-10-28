"""Actual FCN-LectureNet test on one real frame. No video/ST3D claims.

Example (run from checkout with dependencies installed):
python scripts/evaluate_keyframe.py frame.png --ground-truth gt.png --width 1920 --output-dir /tmp/frame-1920
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import cv2
import numpy as np
from boardnotes.config import Settings
from boardnotes.model import LectureNet
from boardnotes.pdf import generate_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.width < 32:
        parser.error("width must be at least 32")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    original = cv2.imread(str(args.image))
    if original is None:
        parser.error("cannot decode input image")
    h,w = original.shape[:2]
    inference_width = min(args.width,w)
    frame = cv2.resize(original,(inference_width, round(h*inference_width/w)),interpolation=cv2.INTER_AREA) if inference_width != w else original
    if min(frame.shape[:2]) < 32:
        parser.error("resized input must be at least 32x32")
    load_start = time.monotonic()
    settings = Settings()
    model = LectureNet(settings)
    loaded = time.monotonic()
    mask = model.extract(frame)
    inferred = time.monotonic()
    del model
    binary = 255-mask
    cv2.imwrite(str(out/"input.png"),frame)
    cv2.imwrite(str(out/"fcn_binary.png"),binary)
    generate_pdf([(binary,0,0)],out/"result.pdf",timestamps=False,image_dir=out/"final",style=settings.pdf_style)
    projected = cv2.resize(binary,(w,h),interpolation=cv2.INTER_NEAREST) if binary.shape!=(h,w) else binary
    cv2.imwrite(str(out/"binary_at_source_size.png"),projected)
    result = {"test_type":"single real keyframe; no temporal integration or video decoding",
              "input_sha256":hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "checkpoint_sha256":hashlib.sha256(settings.checkpoint.read_bytes()).hexdigest(),
              "device":settings.device,"threads":settings.threads,
              "source_dimensions":[w,h],"inference_dimensions":[frame.shape[1],frame.shape[0]],
              "model_load_seconds":round(loaded-load_start,3),"inference_seconds":round(inferred-loaded,3),
              "process_peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
              "pdf_style":settings.pdf_style,"pages":1}
    if args.ground_truth:
        gt=cv2.imread(str(args.ground_truth),0)
        if gt is None or gt.shape!=(h,w):
            raise ValueError("ground truth must match the source image dimensions")
        truth, predicted = gt<128, projected<128
        tp=int(np.count_nonzero(truth & predicted))
        fp=int(np.count_nonzero(~truth & predicted))
        fn=int(np.count_nonzero(truth & ~predicted))
        result["exact_pixel_metrics"]={"tp":tp,"fp":fp,"fn":fn,
            "precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"f1":2*tp/max(1,2*tp+fp+fn),
            "note":"Strict pixel overlap at source resolution; predictions resized with nearest neighbor for evaluation only. Not OCR accuracy or dataset-wide generalization."}
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
