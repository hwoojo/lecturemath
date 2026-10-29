"""Run fixed additional text ROIs, without passing references to inference."""
import argparse
import json
from pathlib import Path
import re
import cv2
from boardnotes.v2.pipeline import worker_call
from boardnotes.v25.pipeline import configuration
from experiments.v1_5.real_report import distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ocr-python',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[];records=[]
    for sample in json.loads(Path('experiments/v25/additional_regions.json').read_text())['samples']:
        for variant,path in [('rgb',sample['source_path']),('b0',f"benchmark-output/v25/b0/{sample['id']}/b0.png")]:
            image=cv2.imread(path)
            if image is None:raise ValueError(path)
            for region in sample['regions']:
                if region['type']!='text':continue
                rid=f"{sample['id']}__{variant}__{region['id']}"
                x1,y1,x2,y2=region['bbox'];crop=args.output/(rid+'.png')
                cv2.imwrite(str(crop),image[y1:y2,x1:x2])
                jobs.append({'id':rid,'image':str(crop.resolve())})
                records.append({**region,'id':rid,'variant':variant,'sample':sample['id']})
    (args.output/'annotations.json').write_text(json.dumps(records,indent=2)+'\n')
    config=configuration('models/v2','PP-FormulaNet-S')
    result=worker_call(args.ocr_python,'text',jobs,config,args.output/'worker',timeout=600)
    predictions={item['id']:item for item in result['items']}
    rows=[]
    for record in records:
        item=predictions[record['id']]
        expected=re.sub(r'\s+',' ',record['reference']).strip()
        actual=re.sub(r'\s+',' ',item.get('content') or '').strip()
        rows.append({**record,**item,'character_errors':distance(expected,actual),
                     'reference_characters':len(expected),'cer':distance(expected,actual)/max(1,len(expected))})
    summary={v:{'count':len([r for r in rows if r['variant']==v]),
                'macro_cer':sum(r['cer'] for r in rows if r['variant']==v)/len([r for r in rows if r['variant']==v])}
             for v in ['rgb','b0']}
    (args.output/'results.json').write_text(json.dumps({'scope':'oracle text ROI; case/punctuation preserved; whitespace collapsed',
        'summary':summary,'run':{k:v for k,v in result.items() if k!='items'},'predictions':rows},indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
