"""Sequential real inference, fixed manifest, resumable per-model files."""
import argparse
import hashlib
import json
from pathlib import Path
from boardnotes.v2.pipeline import worker_call


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path('docs/assets/v251'))
    p.add_argument('--output',type=Path,default=Path('benchmark-output/v251'))
    p.add_argument('--ocr-python',type=Path,required=True)
    p.add_argument('--models',nargs='+')
    args=p.parse_args();path=args.assets/'manifest.json'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!= (args.assets/'manifest.sha256').read_text().split()[0]:raise ValueError('manifest changed')
    data=json.loads(path.read_text());args.output.mkdir(parents=True,exist_ok=True)
    for r in data['cases']:
        if hashlib.sha256(Path(r['input']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('input changed: '+r['id'])
    jobs=[{'id':r['id'],'image':str(Path(r['input']).resolve())} for r in data['cases'] if not r.get('reuse_v25')]
    old=json.loads(Path('docs/assets/v25/comparison.json').read_text())
    weights=json.loads(Path('models/v2/models.json').read_text())
    target=args.assets/'predictions';target.mkdir(exist_ok=True)
    for name in args.models or data['models']:
        if name not in data['models']:raise ValueError(name)
        destination=target/(name+'.json')
        if destination.exists():
            previous=json.loads(destination.read_text())
            if previous['manifest_sha256']!=digest:raise ValueError('refusing to reuse mismatched predictions')
            print('Reusing',name,flush=True);continue
        print('Starting',name,len(jobs),'fresh crops',flush=True)
        config={'formula_model':name,'formula_dir':str(Path('models/v2',name+'_infer').resolve()),'threads':2,'max_rss_mib':6000}
        result=worker_call(args.ocr_python,'formula',jobs,config,args.output/name,timeout=3600)
        fresh={r['id']:r for r in result['items']}
        baseline={r['id']:r for r in old['predictions'] if r['model']==name and r.get('reference') is not None}
        rows=[]
        for case in data['cases']:
            pred=baseline[case['source_id']] if case.get('reuse_v25') else fresh[case['id']]
            rows.append({'id':case['id'],'latex':pred.get('latex'),'status':pred['status'],'error':pred.get('error'),
                         'seconds':pred.get('seconds'),'evidence':'reused_v25_same_weights' if case.get('reuse_v25') else 'fresh_v251_inference'})
        report={'model':name,'manifest_sha256':digest,'worker_sha256':hashlib.sha256(Path('boardnotes/v2/worker.py').read_bytes()).hexdigest(),
                'weights':weights[name],'run':{k:v for k,v in result.items() if k!='items'},'predictions':rows}
        destination.write_text(json.dumps(report,indent=2)+'\n')
        print('Completed',name,'failed',sum(r['status']=='failed' for r in rows),flush=True)


if __name__=='__main__':main()
