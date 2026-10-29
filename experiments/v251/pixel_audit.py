"""Recorded local feature evidence, not calibrated segmentation ground truth."""
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    root=Path('docs/assets/v251');rows=[]
    prefix='real__lecture_08_53016__'
    rgb=cv2.imread(str(root/'inputs'/(prefix+'rgb__cardinality_factorial.png')))
    b0=cv2.imread(str(root/'inputs'/(prefix+'b0__cardinality_factorial.png')))
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    for name,box in [('factorial_dot',[153,71,172,93]),('factorial_stem',[154,18,171,71])]:
        x1,y1,x2,y2=box;mask=cv2.cvtColor(b0[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)<128
        record={'feature':name,'bbox_relative_to_original_roi':box,'b0_black_pixels':int(mask.sum()),'rgb_thresholds':{}}
        for threshold in [96,128,150,192]:
            dark=gray[y1:y2,x1:x2]<threshold
            record['rgb_thresholds'][str(threshold)]={'rgb_dark_pixels':int(dark.sum()),'shared_dark_pixels':int((dark&mask).sum()),
                'b0_over_rgb_dark_recall':float((dark&mask).sum()/max(1,dark.sum()))}
        for variant,image in [('rgb',rgb),('b0',b0)]:
            crop=image[y1:y2,x1:x2]
            out=root/'pixel_audit'/(name+'_'+variant+'.png');out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out),cv2.resize(crop,None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST))
        rows.append(record)
    cases=json.loads((root/'manifest.json').read_text())['cases'];boundary=[]
    for r in cases:
        if r['group']!='real_baseline' or r['variant']!='b0':continue
        image=cv2.imread(r['input'],0);ink=image<128
        boundary.append({'id':r['id'],'bbox':r['source_bbox'],'black_border_pixels':{'top':int(ink[0].sum()),'bottom':int(ink[-1].sum()),'left':int(ink[:,0].sum()),'right':int(ink[:,-1].sum())},
                         'notice':'Boundary contact flags possible clipping only; absence does not establish complete equation context.'})
    byid={r['id']:r for r in cases};context=[]
    for r in cases:
        if r['group']!='sensitivity' or r['condition'] not in {'pad10','pad20'}:continue
        base=byid[r['base_id']];image=cv2.imread(r['input'],0)
        x=base['source_bbox'][0]-r['source_bbox'][0];y=base['source_bbox'][1]-r['source_bbox'][1]
        outside=np.ones(image.shape,bool);outside[y:y+base['height'],x:x+base['width']]=False
        context.append({'id':r['id'],'added_dark_pixels_outside_original_bbox':{str(t):int(((image<t)&outside).sum()) for t in [128,150,200]},
                        'notice':'Added pixels may be neighboring writing, line extensions or background texture; not labeled noise.'})
    (root/'pixel-audit.json').write_text(json.dumps({'protocol':'Feature boxes fixed by visual inspection of source input before V2.51 prediction inspection. RGB thresholds are sensitivity diagnostics, not ground-truth masks.',
        'features':rows,'boundary_contact':boundary,'expanded_context':context},indent=2)+'\n')


if __name__=='__main__':main()
