"""Oracle-ROI model comparison, independent from automatic region detection."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
from boardnotes.v2.pipeline import worker_call
from experiments.v2.benchmark import SAMPLES
from experiments.v1_5.real_report import distance
from boardnotes.v25.splitting import propose_rows,assemble_rows


def collect_jobs(root,additional=None,regression=True,split=False):
    jobs=[];records=[]
    sources=[]
    if regression:
        for name,spec in SAMPLES.items():
            regions=[{'id':rid,'type':'formula','bbox':box,'reference':ref,'category':'regression',
                      'structure_hint':'aligned' if rid=='formula_system' else None} for rid,(box,ref) in spec.items() if rid.startswith('formula')]
            sources.append({'id':name,'source_path':f'benchmark-output/v1_5-real/prepared/1080p/image/{name}.png',
                            'b0_path':f'benchmark-output/v1_5-real/results/1080p/{name}/B0.png','regions':regions,'split':'development-regression'})
    if additional:
        for sample in json.loads(Path(additional).read_text())['samples']:
            sources.append({**sample,'split':'evaluation-unseen-lectures'})
    for sample in sources:
        name=sample['id'];b0=sample.get('b0_path')
        candidate=Path('benchmark-output/v25/b0')/name/'b0.png'
        if b0 is None and candidate.exists(): b0=str(candidate)
        for variant,source in [('rgb',sample['source_path']),('b0',b0)]:
            if source is None:continue
            image=cv2.imread(source)
            if image is None: raise ValueError(source)
            h,w=image.shape[:2]
            for region in sample['regions']:
                if region['type']!='formula':continue
                rid=region['id'];box=region['bbox'];x1,y1,x2,y2=box
                if not 0<=x1<x2<=w or not 0<=y1<y2<=h: raise ValueError(box)
                key=f'{name}__{variant}__{rid}'
                path=root/'crops'/(key+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(path),image[y1:y2,x1:x2])
                ref=region.get('reference')
                records.append({'id':key,'image':name,'variant':variant,'region_id':rid,'bbox':box,'reference':ref,
                                'category':region.get('category'),'split':sample['split'],'crop_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                jobs.append({'id':key,'image':str(path.resolve())})
                if split and variant=='b0':
                    proposed=propose_rows(image,box,region.get('structure_hint'))
                    (root/'crops'/(key+'_split.json')).write_text(json.dumps(proposed,indent=2)+'\n')
                    for i,row in enumerate(proposed['rows']):
                        a,b,c,d=row['bbox'];rowid=key+f'__row{i}'
                        rowpath=root/'crops'/(rowid+'.png');cv2.imwrite(str(rowpath),image[b:d,a:c])
                        jobs.append({'id':rowid,'image':str(rowpath.resolve())})
                        records.append({'id':rowid,'image':name,'variant':variant,'region_id':rid,'bbox':row['bbox'],
                                        'reference':None,'category':'split-row','split':sample['split'],'parent_id':key,
                                        'assembly_hint':region.get('structure_hint'),'assembly_allowed':proposed['assembly_allowed']})
    return jobs,records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ocr-python',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['PP-FormulaNet-S','PP-FormulaNet_plus-S'],choices=['PP-FormulaNet-S','PP-FormulaNet_plus-S','PP-FormulaNet-L','UniMERNet'])
    p.add_argument('--additional',type=Path)
    p.add_argument('--no-regression',action='store_true')
    p.add_argument('--split',action='store_true')
    p.add_argument('--timeout',type=float,default=1800)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise ValueError('output must be empty; previous predictions must be retained')
    args.output.mkdir(parents=True,exist_ok=True)
    jobs,records=collect_jobs(args.output,args.additional,not args.no_regression,args.split)
    runs=[];rows=[]
    (args.output/'annotations.json').write_text(json.dumps(records,indent=2)+'\n')
    for name in args.models:
        config={'formula_model':name,'formula_dir':str(Path('models/v2',name+'_infer').resolve()),'threads':2,'max_rss_mib':6000}
        print('Starting',name,len(jobs),'crops',flush=True)
        result=worker_call(args.ocr_python,'formula',jobs,config,args.output/name,timeout=args.timeout)
        runs.append({'model':name,**{k:v for k,v in result.items() if k!='items'}})
        predictions={item['id']:item for item in result['items']}
        for record in records:
            prediction=predictions[record['id']];reference=record['reference'];latex=prediction.get('latex')
            success=prediction['status']=='ok' and reference is not None
            expected=re.sub(r'\s+','',reference) if reference is not None else None
            actual=re.sub(r'\s+','',latex or '')
            rows.append({**record,'model':name,**prediction,'exact_match':expected==actual if success else None,
                         'edit_distance':distance(expected,actual) if success else None})
        for parent in {r['parent_id'] for r in records if r.get('parent_id')}:
            parts=[r for r in records if r.get('parent_id')==parent]
            values=[predictions[r['id']].get('latex','') for r in parts]
            assembly=assemble_rows(values,parts[0].get('assembly_hint')) if parts[0]['assembly_allowed'] else {'status':'needs_review','latex':None,'error':'Assembly was not authorized by structure hint/geometry'}
            rows.append({'id':parent+'__assembled','parent_id':parent,'model':name,'category':'assembled','row_ids':[r['id'] for r in parts],**assembly})
        print('Completed',name,'failed',sum(x['status']=='failed' for x in result['items']),flush=True)
        (args.output/'results.json').write_text(json.dumps({'scope':'oracle ROI; no GT-based runtime model routing','runs':runs,'predictions':rows},indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
