"""Fetch only the three fixed additional V2.5 RGB keyframes.

The reference annotations are never generated or modified by this utility. No
recognition, parameter selection, GT-based routing, or dataset-wide fetch occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
COMMIT = 'b0278437b977679cf57001d7765227ef9e99038f'
URL_PREFIX = f'https://raw.githubusercontent.com/bhargavaurala/accessmath-icfhr2018/{COMMIT}/'


def validate_file(path, sample):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sample['sha256']:
        raise ValueError(f'SHA256 mismatch for {path}; use --refresh to replace a corrupt cached source')
    if data[:8] != b'\x89PNG\r\n\x1a\n' or len(data) < 24:
        raise ValueError(f'Expected native PNG: {path}')
    dimensions = struct.unpack('>II', data[16:24])
    if dimensions != (sample['width'],sample['height']):
        raise ValueError(f'Source dimensions differ from frozen annotations: {path}')
    return len(data)


def prepare(annotations, output_root=ROOT, verify_only=False, refresh=False):
    annotations = Path(annotations)
    output_root = Path(output_root).resolve()
    before = annotations.read_bytes()
    document = json.loads(before)
    if document['source_commit'] != COMMIT:
        raise ValueError('Only the pinned source commit is supported')
    if len(document['samples']) != 3:
        raise ValueError('Expected exactly the three frozen additional source frames')
    prepared = []
    for sample in document['samples']:
        url = sample['source_url']
        if not url.startswith(URL_PREFIX) or not url.endswith('.png'):
            raise ValueError('Only pinned RGB PNG URLs from the selected source are allowed')
        path = (output_root/sample['source_path']).resolve()
        if not path.is_relative_to(output_root):
            raise ValueError('Source destination must remain inside output_root')
        if not path.exists() or refresh:
            if verify_only:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            partial = path.with_suffix('.download-part')
            try:
                with urllib.request.urlopen(url,timeout=60) as response, partial.open('wb') as stream:
                    for block in iter(lambda: response.read(1024*1024), b''):
                        stream.write(block)
                validate_file(partial,sample)
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        size = validate_file(path,sample)
        prepared.append({'id':sample['id'],'path':str(path),'bytes':size,'sha256':sample['sha256']})
    if annotations.read_bytes() != before:
        raise RuntimeError('Reference annotations changed during source preparation')
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,default=ROOT/'experiments/v25/additional_regions.json')
    parser.add_argument('--output-root',type=Path,default=ROOT)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    if args.verify_only and args.refresh:
        parser.error('--verify-only and --refresh cannot be combined')
    for result in prepare(args.annotations,args.output_root,args.verify_only,args.refresh):
        print(json.dumps(result))


if __name__=='__main__':
    main()
