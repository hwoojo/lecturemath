"""Run from repository root: python -m experiments.v1_5.cli --help."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

import cv2
import numpy as np
from . import methods
from .adapter import LectureNetAdapter, load_mask, validate_probability
from .evaluation import evaluate
from .visualization import contact_sheet


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def box(value):
    values = tuple(map(int, value.split(',')))
    if len(values) != 4 or min(values[:2]) < 0 or min(values[2:]) <= 0:
        raise argparse.ArgumentTypeError('expected x,y,width,height with positive size')
    return values


def checked_box(value, shape):
    x, y, w, h = value
    if x+w > shape[1] or y+h > shape[0]:
        raise ValueError('ROI/crop extends outside input')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=['B0','B1','B2','B3','B4'], default=['B0','B1','B2','B3','B4'])
    p.add_argument('--baseline-dir', type=Path)
    p.add_argument('--baseline-polarity', choices=['dark','light'], default='dark')
    p.add_argument('--probability-dir', type=Path, help='float NPY: probability of writing, NOT uint8 or logits')
    p.add_argument('--gt-dir', type=Path)
    p.add_argument('--gt-polarity', choices=['dark','light'], default='dark')
    p.add_argument('--polarity', choices=['dark','light'], required=True, help='writing polarity in source')
    p.add_argument('--limit', type=int, default=5)
    p.add_argument('--resolution', choices=['native','720p','1080p'], default='native', help='validate input size; never resize')
    p.add_argument('--roi', type=box)
    p.add_argument('--crop', type=box, help='contact sheet crop in original coordinates')
    p.add_argument('--zoom', type=int, default=2)
    p.add_argument('--windows', type=int, nargs='+', default=[25,51])
    p.add_argument('--ks', type=float, nargs='+', default=[.2,.35])
    p.add_argument('--blur', type=int, nargs='+', default=[0,3])
    p.add_argument('--clahe', choices=['off','on','both'], default='off')
    p.add_argument('--radius', type=int, default=2)
    p.add_argument('--thresholds', type=float, nargs='+', default=[.35,.5,.65])
    p.add_argument('--config', type=Path)
    p.add_argument('--checkpoint', type=Path, help='optional trusted V34 state_dict; no download')
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--threads', type=int, default=2)
    return p


def run(args):
    if args.limit < 1 or args.zoom < 1 or args.threads < 1 or args.radius < 0:
        raise ValueError('limit/zoom/threads must be positive; radius nonnegative')
    if any(w < 3 or w%2 == 0 for w in args.windows) or any(b < 0 or (b and b%2 == 0) for b in args.blur):
        raise ValueError('windows odd >=3; blur zero or positive odd')
    if any(not np.isfinite(k) for k in args.ks) or any(not 0 < t < 1 for t in args.thresholds):
        raise ValueError('invalid k/threshold')
    if args.checkpoint and (not args.checkpoint.is_file() or not args.config or not args.config.is_file()):
        raise ValueError('inference requires existing checkpoint and config')
    cv2.setNumThreads(args.threads)
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}) if args.input.is_dir() else [args.input]
    paths = paths[:args.limit]
    if not paths:
        raise ValueError('no input images')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('duplicate image stems; use unique names for cache/mask matching')
    args.output.mkdir(parents=True, exist_ok=True)
    records, sources, skipped = [], [], []
    model = None
    model_id = {'checkpoint_sha256': digest(args.checkpoint), 'config_sha256': digest(args.config),
                'model_source_sha256': digest(Path(__file__).resolve().parents[2]/'ACCESS2021_release/AccessMath/lecturenet_v1/FCN_lecturenet.py')} if args.checkpoint else None
    for path in paths:
        image_start = time.perf_counter()
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'cannot decode {path}')
        shape = image.shape[:2]
        expected = {'720p': (720,1280), '1080p': (1080,1920)}.get(args.resolution)
        if expected and shape != expected:
            raise ValueError(f'{path}: expected {expected}, got {shape}; resize sources explicitly before comparison')
        roi = np.ones(shape, bool)
        if args.roi:
            x,y,w,h = checked_box(args.roi, shape)
            roi[:] = False
            roi[y:y+h,x:x+w] = True
        crop = checked_box(args.crop, shape) if args.crop else (0,0,min(320,shape[1]),min(180,shape[0]))
        dest = args.output/path.stem
        dest.mkdir(exist_ok=True)
        write(dest/'original.png', image)
        truth = load_mask(args.gt_dir/(path.stem+'.png'), shape, args.gt_polarity) if args.gt_dir else None
        source = {'input': str(path.resolve()), 'sha256': digest(path), 'shape': list(shape), 'baseline': None, 'probability': None}
        if truth is not None:
            write(dest/'ground_truth.png', methods.png(truth))
            source['ground_truth_sha256'] = digest(args.gt_dir/(path.stem+'.png'))
        baseline, probability = None, None
        if set(args.methods) & {'B0','B3','B4'}:
            if args.baseline_dir and (args.baseline_dir/(path.stem+'.png')).exists():
                bp = args.baseline_dir/(path.stem+'.png')
                baseline = load_mask(bp, shape, args.baseline_polarity)
                source['baseline'] = {'path': str(bp.resolve()), 'sha256': digest(bp)}
            if args.probability_dir and (args.probability_dir/(path.stem+'.npy')).exists():
                pp = args.probability_dir/(path.stem+'.npy')
                probability = validate_probability(np.load(pp, allow_pickle=False), shape)
                source['probability'] = {'path': str(pp.resolve()), 'sha256': digest(pp)}
            if args.checkpoint and (baseline is None or ('B4' in args.methods and probability is None)):
                key_data = {'input_sha256': source['sha256'], **model_id, 'adapter_version': 2}
                key = hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()
                cache = args.cache_dir or args.output/'inference_cache'
                cache.mkdir(parents=True, exist_ok=True)
                cp = cache/(key+'.npz')
                infer_start = time.perf_counter()
                if cp.exists():
                    with np.load(cp, allow_pickle=False) as data:
                        inferred = data['baseline'].astype(bool)
                        prob = validate_probability(data['probability'], shape)
                    if inferred.shape != shape:
                        raise ValueError('misaligned inference cache')
                    cache_hit = True
                else:
                    if model is None:
                        model = LectureNetAdapter(args.config, args.checkpoint, args.threads)
                    inferred, prob = model.infer(image)
                    np.savez_compressed(cp, baseline=inferred, probability=prob)
                    cache_hit = False
                source['inference'] = {**key_data, 'cache': str(cp.resolve()), 'cache_hit': cache_hit,
                                       'seconds_including_load_and_cache': time.perf_counter()-infer_start}
                if baseline is None:
                    baseline = inferred
                    source['baseline'] = 'checkpoint/cache'
                if probability is None:
                    probability = prob
                    source['probability'] = 'checkpoint/cache'
        sources.append(source)
        panels = {}
        def emit(label, mask, seconds, parameters):
            if mask.shape != shape:
                raise ValueError('output resolution changed')
            result = methods.png(mask)
            write(dest/(label+'.png'), result)
            x,y,w,h = crop
            panels[label] = result[y:y+h,x:x+w].copy()
            metric = evaluate(mask, roi, truth)
            records.append({'image': path.stem, 'method': label, 'parameters': parameters,
                            'postprocess_seconds': seconds, **metric})
        for method in args.methods:
            if method in {'B0','B3'} and baseline is None:
                skipped.append({'image':path.stem,'method':method,'reason':'no aligned baseline or V34 checkpoint'})
                continue
            if method == 'B4' and probability is None:
                skipped.append({'image':path.stem,'method':method,'reason':'no continuous probability or V34 checkpoint'})
                continue
            if method == 'B0':
                emit('B0', baseline, 0., {'source': source['baseline']})
            elif method == 'B1':
                for blur in args.blur:
                    start = time.perf_counter()
                    mask = methods.otsu(image,args.polarity,blur)
                    emit(f'B1_blur{blur}',mask,time.perf_counter()-start,{'blur':blur})
            elif method in {'B2','B3'}:
                for window in args.windows:
                    for k in args.ks:
                        for clahe in ([False,True] if args.clahe == 'both' else [args.clahe == 'on']):
                            start = time.perf_counter()
                            if method == 'B2':
                                mask = methods.sauvola(image,args.polarity,window,k,clahe)
                            else:
                                mask = methods.hybrid(image,baseline,args.polarity,window,k,args.radius,clahe)
                            emit(f'{method}_w{window}_k{k:g}_c{int(clahe)}',mask,time.perf_counter()-start,
                                 {'window':window,'k':k,'clahe':clahe, **({'radius':args.radius} if method == 'B3' else {})})
            else:
                for threshold in args.thresholds:
                    start = time.perf_counter()
                    mask = probability >= threshold
                    emit(f'B4_t{threshold:g}',mask,time.perf_counter()-start,{'threshold':threshold})
        x,y,w,h = crop
        contact_sheet(image[y:y+h,x:x+w],panels,dest/'contact_sheet.png',zoom=args.zoom)
        source['total_image_seconds'] = time.perf_counter()-image_start
        # Linux ru_maxrss is process-lifetime high water mark, NOT per-method usage.
        source['process_peak_rss_mib_so_far'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    versions = {}
    for package in ['numpy','opencv-python','opencv-python-headless','torch','torchvision','Pillow']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':versions},
              'sources':sources,'skipped':skipped,'metrics':records,
              'notes':['No GT means no precision/recall/F1/IoU.', 'B0 copy has zero postprocess time; inference timing is separate.',
                       'Peak RSS is process-lifetime high water mark; not per-method memory.',
                       'External masks/probabilities require user-verified provenance and alignment.']}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(f'{len(paths)} images; {len(records)} outputs; {len(skipped)} skipped; {args.output}/metrics.json')
    return report


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
