"""Reproducible functional demo, NOT lecture quality evidence or a V34 substitute."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .cli import parser, run
from .methods import png


def generate(root):
    for directory in ['inputs_dark','inputs_light','fixture_baseline','gt']:
        (root/directory).mkdir(parents=True,exist_ok=True)
    provenance = {'warning':'Synthetic functional fixtures. Baselines are deliberately damaged GT, NOT LectureNet outputs.', 'samples':[]}
    for name,w,h,polarity in [('white_720',1280,720,'dark'),('white_1080',1920,1080,'dark'),('chalk_1080',1920,1080,'light')]:
        scale = w/1280
        gt = np.zeros((h,w),np.uint8)
        for text,y in [('f(x) = x + 1',160),('(a+b) = a + b',280),('x = (1 + 2)',400)]:
            cv2.putText(gt,text,(int(90*scale),int(y*scale)),cv2.FONT_HERSHEY_SIMPLEX,1.7*scale,255,max(1,int(2*scale)),cv2.LINE_8)
        board = np.broadcast_to(np.linspace(150,250,w,dtype=np.uint8),(h,w)).copy()
        board[gt > 0] = 15
        # Non-writing object: direct threshold methods should expose its false positives.
        cv2.rectangle(board,(int(.77*w),int(.35*h)),(int(.93*w),h-1),35,-1)
        cv2.circle(board,(int(.85*w),int(.24*h)),int(.06*w),60,-1)
        if polarity == 'light':
            board = 255-board
        image = cv2.cvtColor(board,cv2.COLOR_GRAY2BGR)
        baseline = gt > 0
        baseline[:,::13] = False  # controlled missing strokes, never model output
        cv2.imwrite(str(root/f'inputs_{polarity}'/(name+'.png')),image)
        cv2.imwrite(str(root/'gt'/(name+'.png')),png(gt > 0))
        cv2.imwrite(str(root/'fixture_baseline'/(name+'.png')),png(baseline))
        provenance['samples'].append({'name':name,'width':w,'height':h,'polarity':polarity})
    (root/'fixture_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('experiments/v1_5/artifacts/synthetic'))
    args = p.parse_args()
    generate(args.output)
    for polarity in ['dark','light']:
        experiment = parser().parse_args([str(args.output/f'inputs_{polarity}'),'--output',str(args.output/f'results_{polarity}'),
            '--polarity',polarity,'--baseline-dir',str(args.output/'fixture_baseline'),'--gt-dir',str(args.output/'gt'),
            '--crop','80,100,480,200','--zoom','2'])
        report = run(experiment)
        report['fixture_warning'] = 'B0 is damaged synthetic GT, NOT V34; all quality metrics are synthetic only.'
        (experiment.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
