"""Short-lived V34 process. No Paddle model shares its memory."""
import argparse
import json
from pathlib import Path
import resource
import time
import cv2
from experiments.v1_5.adapter import LectureNetAdapter
from experiments.v1_5.methods import png


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();start=time.perf_counter()
    image=cv2.imread(str(args.image))
    if image is None:raise ValueError('cannot decode RGB source')
    model=LectureNetAdapter(args.config,args.checkpoint,2);loaded=time.perf_counter()
    mask,probability=model.infer(image);inferred=time.perf_counter()
    if not cv2.imwrite(str(args.output),png(mask)):raise OSError(args.output)
    args.output.with_suffix('.json').write_text(json.dumps({'model':'FCN-LectureNet V34','model_load_seconds':loaded-start,
        'inference_seconds':inferred-loaded,'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'shape':list(mask.shape),'internal_resize_policy':'V1 >2.5MP half-size policy; source coordinates restored'},indent=2)+'\n')


if __name__=='__main__':main()
