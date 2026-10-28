"""Fixed specified-region evaluation; hand transcriptions never reach models."""
import argparse
import csv
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
from boardnotes.v2.pipeline import recognize
from boardnotes.v2.pdf import render_pdf
from experiments.v1_5.real_report import distance, normalized, grid

SAMPLES={
 'lecture_01_10423':{'text':([965,300,1435,380],'Algebraic Interpretation.'),'formula':([340,440,515,510],'3x=1'),'formula_system':([290,430,600,610],r'\left.\begin{aligned}3x&=1\\x+2y&=3\end{aligned}\right\}')},
 'lecture_02_44601':{'text':([275,270,745,340],'Multiplication by a'),'formula':([405,490,710,565],'x+3y=4'),'formula_fraction':([990,315,1095,385],r'y=\frac{27}{19}')},
 'nm_lecture_01_19899':{'text':([225,315,825,410],'System of Linear Equations.'),'formula':([180,480,410,535],'3x-y+z=7'),'formula_fraction':([1470,420,1820,555],r'y=\frac{3-x}{2}=\frac{3-\frac{1}{3}}{2}=\frac{8}{6}')},
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v15-root',type=Path,default=Path('benchmark-output/v1_5-real'))
    p.add_argument('--output',type=Path,default=Path('benchmark-output/v2/real'))
    p.add_argument('--ocr-python',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=Path('models/v2'))
    p.add_argument('--cache-dir',type=Path,default=Path('benchmark-output/v2/recognition-cache'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config={'threads':2,'language':'en','det_dir':str((args.model_root/'PP-OCRv5_mobile_det_infer').resolve()),
            'rec_dir':str((args.model_root/'en_PP-OCRv5_mobile_rec_infer').resolve()),
            'formula_dir':str((args.model_root/'PP-FormulaNet-S_infer').resolve())}
    rows=[];runs=[]
    for name,spec in SAMPLES.items():
        regions=[{'id':kind,'type':kind.split('_',1)[0],'bbox':bbox} for kind,(bbox,reference) in spec.items()]
        source=args.v15_root/'prepared/1080p/image'/(name+'.png')
        folder=args.v15_root/'results/1080p'/name
        variants={'rgb':source,'b0':folder/'B0.png','b3':folder/'B3_w25_k0.2_c0.png'}
        start=time.perf_counter()
        docs=recognize(source,regions,args.output/name,args.ocr_python,config,variants,cache_dir=args.cache_dir)
        run={'image':name,'wall_seconds':time.perf_counter()-start,'stages':docs['rgb']['metadata']['stages']}
        runs.append(run)
        for variant,doc in docs.items():
            render_pdf(doc,args.output/name/(variant+'.pdf'))
            for r in doc['regions']:
                reference=spec[r['id']][1]
                prediction=r['content'] if r['type']=='text' else r['latex']
                if r['status']=='failed':
                    edit=cer=exact=None
                else:
                    norm=normalized if r['type']=='text' else lambda v:re.sub(r'\s+','',v)
                    expected,actual=norm(reference),norm(prediction or '')
                    edit=distance(expected,actual);cer=edit/len(expected);exact=expected==actual
                rows.append({'image':name,'variant':variant,'region_id':r['id'],'type':r['type'],'bbox':r['bbox'],'reference':reference,
                             'prediction':prediction,'status':r['status'],'model':r['model'],'confidence':r['confidence'],
                             'edit_distance':edit,'cer' if r['type']=='text' else 'normalized_edit_distance':cer,
                             'exact_match':exact,'seconds':r.get('seconds'),'cache_hit':r['cache_hit'],'error':r['error']})
        for kind,(bbox,reference) in spec.items():
            x1,y1,x2,y2=bbox;panels=[]
            for variant in ['rgb','b0','b3']:
                crop=cv2.imread(next(r for r in docs[variant]['regions'] if r['id']==kind)['crop_image'])
                panels.append((variant.upper(),crop))
            grid(panels,args.output/(name+'_'+kind+'.png'),f'REAL {name} | {kind} | same source crop {bbox} | nearest 2x')
        print('Completed',name,flush=True)
    report={'scope':'3 real frames, 3 text titles + 3 simple handwritten equations + 3 multi-line/fraction crops, original 1080p, manually specified regions',
            'references':SAMPLES,'normalization':{'text':'lowercase and collapse whitespace; punctuation retained','formula':'remove whitespace only; not mathematical equivalence'},
            'b3_settings':{'window':25,'k':.2,'radius':2,'clahe':False},'config':config,'runs':runs,'metrics':rows,
            'caveats':['PP-FormulaNet-S is officially aimed at simple printed formulas; handwriting is exploratory.',
                       'Tiny convenience sample; not held-out generalization.','No Korean/complex formula evaluation.',
                       'No mock predictions in this report. Model failures have null accuracy metrics.']}
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (args.output/'comparison.csv').open('w') as f:
        fields=['image','variant','region_id','type','reference','prediction','status','edit_distance','cer','normalized_edit_distance','exact_match','seconds','cache_hit']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(args.output/'comparison.json')


if __name__=='__main__': main()
