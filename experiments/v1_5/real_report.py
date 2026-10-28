"""Compare saved real-frame runs without re-running LectureNet."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from .evaluation import evaluate

MATH_CROPS = {'lecture_01_10423':(290,430,320,200),'lecture_02_44601':(1320,480,470,250),
              'nm_lecture_01_19899':(150,435,330,210)}
TEXT_CROPS = {'lecture_01_10423':((965,300,470,80),'Algebraic Interpretation.'),
              'lecture_02_44601':((275,270,470,70),'Multiplication by a'),
              'nm_lecture_01_19899':((225,315,600,95),'System of Linear Equations.')}


def read(path):
    image=cv2.imread(str(path))
    if image is None:
        raise ValueError(path)
    return image


def grid(panels, path, caption, size=None, zoom=2):
    rows=[]
    cells=[]
    for label,image in panels:
        w,h=size or (panels[0][1].shape[1]*zoom,panels[0][1].shape[0]*zoom)
        image=cv2.resize(image,(w,h),interpolation=cv2.INTER_NEAREST)
        cell=cv2.copyMakeBorder(image,35,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(cell,label,(7,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1,cv2.LINE_AA)
        cells.append(cell)
        if len(cells)==3:
            rows.append(np.hstack(cells));cells=[]
    if cells:
        cells += [np.full_like(cells[0],255) for _ in range(3-len(cells))]
        rows.append(np.hstack(cells))
    result=cv2.copyMakeBorder(np.vstack(rows),42,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
    cv2.putText(result,caption,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),result):
        raise OSError(path)


def distance(a,b):
    row=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        new=[i]
        for j,cb in enumerate(b,1):
            new.append(min(new[-1]+1,row[j]+1,row[j-1]+(ca!=cb)))
        row=new
    return row[-1]


def normalized(text):
    return re.sub(r'\s+',' ',text.lower()).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('benchmark-output/v1_5-real'))
    p.add_argument('--output',type=Path,default=Path('docs/assets/v1_5_real'))
    p.add_argument('--ocr',action='store_true',help='optional installed Tesseract, short English crops only')
    p.add_argument('--pdf',action='store_true',help='optional existing ReportLab; fixed image placement, no smoothing')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.input/'source_manifest.json').read_text())
    all_metrics=[];deltas=[];runs=[];ocr=[]
    tesseract=shutil.which('tesseract') if args.ocr else None
    for res in ['1080p','720p']:
        data=json.loads((args.input/'results'/res/'metrics.json').read_text())
        extra=json.loads((args.input/'clahe'/res/'metrics.json').read_text())
        runs.append({'resolution':res,'settings':data['settings'],'environment':data['environment'],'sources':data['sources']})
        methods={name:record for record in data['metrics']+extra['metrics'] for name in [(record['image'],record['method'])]}
        for sample in manifest['samples']:
            name=sample['id'];folder=args.input/'results'/res/name
            image=read(folder/'original.png');truth=read(folder/'ground_truth.png')[:,:,0]==0
            h,w=truth.shape;scale=w/1920
            x,y,rw,rh=[round(v*scale) for v in (150,160,1670,760)]
            roi=np.zeros((h,w),bool);roi[y:y+rh,x:x+rw]=True
            baseline=read(folder/'B0.png')[:,:,0]==0
            masks={}
            for (frame,method),record in methods.items():
                if frame!=name:
                    continue
                source=args.input/('clahe' if method.endswith('_c1') else 'results')/res/name/(method+'.png')
                pred=read(source)[:,:,0]==0
                masks[method]=pred
                if res=='720p':
                    source_truth=read(args.input/'source/gt'/(name+'.png'))[:,:,0]==0
                    projected=cv2.resize(pred.astype(np.uint8),(1920,1080),interpolation=cv2.INTER_NEAREST)>0
                    source_metrics=evaluate(projected,np.ones((1080,1920),bool),source_truth)
                else:
                    source_metrics={key:record[key] for key in ['precision','recall','f1','iou','tp','fp','fn']}
                all_metrics.append({'resolution':res,**record,'board_roi':evaluate(pred,roi,truth),'source_grid':source_metrics})
                if method.startswith('B3') or method.startswith('B4'):
                    added=pred & ~baseline;removed=baseline & ~pred
                    deltas.append({'resolution':res,'image':name,'method':method,
                        'added_true_pixels':int((added&truth).sum()),'added_false_pixels':int((added&~truth).sum()),
                        'removed_true_pixels':int((removed&truth).sum()),'removed_false_pixels':int((removed&~truth).sum())})
            if res=='1080p':
                labels=[('Original RGB',image),('Ground truth',read(folder/'ground_truth.png'))]
                choices=[('B0','B0'),('B1','B1_blur0'),('B2 w25 k.2','B2_w25_k0.2_c0'),
                         ('B3 w25 k.2 r2','B3_w25_k0.2_c0'),('B3 + CLAHE','B3_w25_k0.2_c1'),
                         ('B4 t=.35','B4_t0.35'),('B4 t=.65','B4_t0.65')]
                for label,method in choices:
                    img=np.where(masks[method],0,255).astype(np.uint8)
                    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                    labels.append((label,img))
                grid(labels,args.output/(name+'_overview.png'),f'REAL {name} | native 1920x1080 | overview nearest resize to 640x360',size=(640,360))
                for tag,box in [('math',MATH_CROPS[name]),('text',TEXT_CROPS[name][0])]:
                    x,y,cw,ch=box
                    cropped=[(label,img[y:y+ch,x:x+cw]) for label,img in labels]
                    grid(cropped,args.output/(name+'_'+tag+'.png'),f'REAL {name} | crop={box} | nearest 2x')
                refined=masks['B3_w25_k0.2_c1'];added=refined&~baseline
                color=np.full_like(image,255);color[baseline]=0
                color[added&truth]=(0,150,0);color[added&~truth]=(0,0,230)
                x,y,cw,ch=MATH_CROPS[name]
                panel=[('B0',labels[2][1][y:y+ch,x:x+cw]),('B3 + CLAHE',np.where(refined[y:y+ch,x:x+cw,None],0,255).astype(np.uint8).repeat(3,axis=2)),
                       ('Added: green=GT ink, red=not GT',color[y:y+ch,x:x+cw])]
                grid(panel,args.output/(name+'_changes.png'),f'REAL {name} | B3+CLAHE additions relative to B0 | nearest 2x')
            if tesseract:
                native,reference=TEXT_CROPS[name]
                x,y,cw,ch=[round(v*scale) for v in native]
                options=[('RGB',image)]+[(method,np.where(masks[method],0,255).astype(np.uint8)) for method in
                          ['B0','B1_blur0','B2_w25_k0.2_c0','B3_w25_k0.2_c0','B3_w25_k0.2_c1','B4_t0.35','B4_t0.65']]
                with tempfile.TemporaryDirectory() as temp:
                    for method,img in options:
                        path=Path(temp)/'crop.png';cv2.imwrite(str(path),img[y:y+ch,x:x+cw])
                        result=subprocess.run([tesseract,str(path),'stdout','--psm','7','-l','eng'],capture_output=True,text=True,check=True)
                        expected=normalized(reference);actual=normalized(result.stdout)
                        ocr.append({'resolution':res,'image':name,'method':method,'crop':[x,y,cw,ch],
                            'manual_reference':reference,'raw_prediction':result.stdout,'normalized_reference':expected,
                            'normalized_prediction':actual,'edit_distance':distance(expected,actual),'cer':distance(expected,actual)/len(expected)})
    for sample in manifest['samples']:
        name=sample['id'];x,y,cw,ch=MATH_CROPS[name]
        panels=[('Original 1080p',read(args.input/'results/1080p'/name/'original.png')),
                ('Ground truth 1080p',read(args.input/'results/1080p'/name/'ground_truth.png'))]
        for label,folder,res,method in [('B0 1080p','results','1080p','B0'),('B0 720p -> source grid','results','720p','B0'),
                                       ('B3+CLAHE 1080p','clahe','1080p','B3_w25_k0.2_c1'),
                                       ('B3+CLAHE 720p -> source grid','clahe','720p','B3_w25_k0.2_c1')]:
            img=read(args.input/folder/res/name/(method+'.png'))
            if res=='720p':
                img=cv2.resize(img,(1920,1080),interpolation=cv2.INTER_NEAREST)
            panels.append((label,img))
        grid([(label,img[y:y+ch,x:x+cw]) for label,img in panels],args.output/(name+'_resolutions.png'),
             f'REAL {name} | native 1080p vs derived 720p | common source crop | nearest 2x')
    aggregate=[]
    for res in ['1080p','720p']:
        for method in sorted({m['method'] for m in all_metrics}):
            selected=[m for m in all_metrics if m['resolution']==res and m['method']==method]
            tp=sum(m['tp'] for m in selected);fp=sum(m['fp'] for m in selected);fn=sum(m['fn'] for m in selected)
            aggregate.append({'resolution':res,'method':method,'macro_f1':sum(m['f1'] for m in selected)/len(selected),
                'micro_f1':2*tp/(2*tp+fp+fn),'precision':tp/(tp+fp),'recall':tp/(tp+fn),
                'macro_board_roi_f1':sum(m['board_roi']['f1'] for m in selected)/len(selected),
                'macro_source_grid_f1':sum(m['source_grid']['f1'] for m in selected)/len(selected)})
    summary={'source_manifest':manifest,'runs':runs,'metrics':all_metrics,'aggregate':aggregate,'pixel_changes':deltas,
             'ocr':{'engine':subprocess.run([tesseract,'--version'],capture_output=True,text=True).stdout.splitlines()[0] if tesseract else None,
                    'scope':'Three manually transcribed English title crops, same native-size inputs, PSM7 eng, lowercase+whitespace normalization. No math OCR evaluation.',
                    'results':ocr},
             'limits':['Only 3 different real frames; 720p is derived from native 1080p.','No blackboard/chalk or meaningful instructor occlusion coverage.',
                       'Unknown V34 training overlap. No held-out parameter selection.','CLAHE-on single setting is exploratory.','Process peak RSS is cumulative, not per-method memory.']}
    (args.output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['resolution','image','method','precision','recall','f1','iou','foreground_ratio','postprocess_seconds']
    with (args.output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(all_metrics)
    if args.pdf:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4,landscape
        pdf=canvas.Canvas(str(args.output/'comparison.pdf'),pagesize=landscape(A4))
        page_w,page_h=landscape(A4)
        for sample in manifest['samples']:
            name=sample['id']
            for label,folder in [('Original RGB','results'),('B0','results'),('B3_w25_k0.2_c0','results'),
                                 ('B3_w25_k0.2_c1','clahe'),('B4_t0.35','results'),('B4_t0.65','results')]:
                path=args.input/folder/'1080p'/name/(('original' if label=='Original RGB' else label)+'.png')
                pdf.setFont('Helvetica',12)
                pdf.drawString(20,page_h-25,f'REAL {name} | {label} | native 1920x1080, no smoothing')
                width=page_w-40;height=width*1080/1920
                pdf.drawImage(str(path),20,(page_h-height)/2,width=width,height=height)
                pdf.showPage()
        pdf.save()
    print(f'{len(all_metrics)} method/frame/resolution rows; {len(ocr)} English OCR crops; {args.output}')


if __name__=='__main__':
    main()
