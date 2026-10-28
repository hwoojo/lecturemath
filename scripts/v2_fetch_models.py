"""Download only the pinned official CPU inference artifacts used by V2."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

BASE='https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('models/v2'))
    p.add_argument('--formula',action='store_true',help='include PP-FormulaNet-S (~224MiB); printed-formula model')
    p.add_argument('--language',choices=['en','ko'],default='en')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    names=['PP-OCRv5_mobile_det','en_PP-OCRv5_mobile_rec' if args.language=='en' else 'korean_PP-OCRv5_mobile_rec']
    if args.formula: names.append('PP-FormulaNet-S')
    manifest={}
    for name in names:
        archive=args.output/(name+'_infer.tar');url=BASE+archive.name
        if not archive.exists():
            partial=archive.with_suffix('.part')
            print('Downloading',name,flush=True)
            try:
                with urllib.request.urlopen(url,timeout=60) as response,partial.open('wb') as out:
                    total=0;reported=0
                    while True:
                        data=response.read(1024*1024)
                        if not data: break
                        out.write(data);total+=len(data)
                        if total-reported>=20*1024*1024:
                            print(name,total//(1024*1024),'MiB',flush=True);reported=total
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        digest=hashlib.sha256()
        with archive.open('rb') as f:
            for data in iter(lambda:f.read(1024*1024),b''): digest.update(data)
        with tarfile.open(archive) as tar:
            tar.extractall(args.output,filter='data')
        candidates=[d for d in args.output.iterdir() if d.is_dir() and (d/'inference.yml').exists() and name in d.name]
        if len(candidates)!=1:
            raise RuntimeError(f'cannot uniquely locate extracted {name}: {candidates}')
        model_dir=candidates[0].resolve()
        manifest[name]={'url':url,'archive_bytes':archive.stat().st_size,'archive_sha256':digest.hexdigest(),'directory':str(model_dir),
                        'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in model_dir.iterdir() if f.is_file()}}
        print('Prepared',model_dir,flush=True)
    (args.output/'models.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
