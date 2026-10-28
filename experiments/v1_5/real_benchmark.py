"""Small pinned AccessMath2015 benchmark. Never trains or downloads checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

import cv2
import numpy as np

COMMIT = 'b0278437b977679cf57001d7765227ef9e99038f'
SAMPLES = [('lecture_01','10423'), ('lecture_02','44601'), ('nm_lecture_01','19899')]


def prepare(root):
    manifest = {'dataset':'AccessMath2015','repository':'bhargavaurala/accessmath-icfhr2018','commit':COMMIT,
                'selection':'Previously evaluated lecture_01/10423 plus middle annotated keyframes of lecture_02 and nm_lecture_01; fixed before inference.',
                'resizing':'1080p native; 720p RGB INTER_AREA, GT INTER_NEAREST; each resolution evaluated in its own coordinates.',
                'train_test_caveat':'V34 training inclusion unknown. These are annotated keyframes, not a continuous video.', 'samples':[]}
    for lecture,frame in SAMPLES:
        name = lecture+'_'+frame
        record = {'id':name,'files':{}}
        for kind,folder in [('image','keyframes'),('gt','binary')]:
            url = f'https://raw.githubusercontent.com/bhargavaurala/accessmath-icfhr2018/{COMMIT}/output/annotations/AccessMath2015_{lecture}/{folder}/{frame}.png'
            path = root/'source'/kind/(name+'.png')
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                with urllib.request.urlopen(url,timeout=30) as response:
                    path.write_bytes(response.read())
            record['files'][kind] = {'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        image = cv2.imread(str(root/'source/image'/(name+'.png')))
        gt = cv2.imread(str(root/'source/gt'/(name+'.png')),0)
        if image is None or gt is None or image.shape[:2] != (1080,1920) or gt.shape != (1080,1920) or not np.isin(gt,[0,255]).all():
            raise ValueError(f'invalid or misaligned source: {name}')
        record['native_shape'] = list(gt.shape)
        for resolution,w,h in [('1080p',1920,1080),('720p',1280,720)]:
            for kind,array,interpolation in [('image',image,cv2.INTER_AREA),('gt',gt,cv2.INTER_NEAREST)]:
                dest = root/'prepared'/resolution/kind/(name+'.png')
                dest.parent.mkdir(parents=True,exist_ok=True)
                resized = array if resolution=='1080p' else cv2.resize(array,(w,h),interpolation=interpolation)
                if not cv2.imwrite(str(dest),resized):
                    raise OSError(dest)
        manifest['samples'].append(record)
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('benchmark-output/v1_5-real'))
    p.add_argument('--prepare-only',action='store_true')
    args = p.parse_args()
    prepare(args.output)
    if args.prepare_only:
        return
    for resolution in ['1080p','720p']:
        command = [sys.executable,'-m','experiments.v1_5.cli',str(args.output/'prepared'/resolution/'image'),
                   '--output',str(args.output/'results'/resolution),'--gt-dir',str(args.output/'prepared'/resolution/'gt'),
                   '--polarity','dark','--resolution',resolution,'--limit','3','--threads','2',
                   '--checkpoint','models/LectureNet_model_BIN_V34_final.dat',
                   '--config','ACCESS2021_release/configs/FCN_LectureNet.conf',
                   '--cache-dir',str(args.output/'cache'/resolution)]
        print('Running:', ' '.join(command),flush=True)
        subprocess.run(command,check=True)
        # A single fixed CLAHE setting, exploratory; do not select using GT.
        baseline_dir = args.output/'baseline'/resolution
        baseline_dir.mkdir(parents=True,exist_ok=True)
        for lecture,frame in SAMPLES:
            name = lecture+'_'+frame
            baseline = cv2.imread(str(args.output/'results'/resolution/name/'B0.png'),0)
            if not cv2.imwrite(str(baseline_dir/(name+'.png')),baseline):
                raise OSError(name)
        subprocess.run([sys.executable,'-m','experiments.v1_5.cli',str(args.output/'prepared'/resolution/'image'),
            '--output',str(args.output/'clahe'/resolution),'--gt-dir',str(args.output/'prepared'/resolution/'gt'),
            '--baseline-dir',str(baseline_dir),'--polarity','dark','--resolution',resolution,
            '--methods','B2','B3','--windows','25','--ks','0.2','--clahe','on'],check=True)


if __name__ == '__main__':
    main()
