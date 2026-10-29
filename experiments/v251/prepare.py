"""Generate and freeze all V2.51 conditions before model inference."""
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from boardnotes.v25.rendering import render_latex
from .design import CONTROLS,SENSITIVITY,ALTERNATE_PRINTS,SPLIT_ROWS
from .inputs import ink_box,condition_crop,printed_canvas

ROOT=Path('docs/assets/v251')


def main():
    if (ROOT/'manifest.json').exists():raise ValueError('manifest already frozen; use a new experiment path for changed conditions')
    ROOT.mkdir(parents=True,exist_ok=True)
    rows=[]
    def add(rid,image,reference,group,**details):
        path=ROOT/'inputs'/(rid+'.png');path.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(path),image)
        row={'id':rid,'input':str(path),'reference':reference,'group':group,'width':image.shape[1],'height':image.shape[0],
             'ink_box':ink_box(image),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**details}
        rows.append(row);return row
    def printed(ref,key):
        record=render_latex(ref,ROOT/'reference_renders'/(key+'.png'))
        if record['status']!='rendered':raise RuntimeError(record)
        return cv2.imread(record['path'])
    controls={}
    for key,family,level,ref in CONTROLS:
        source=printed(ref,'control_'+key)
        # A single TeX font/DPI for all structures; do not equalize total height across complexity.
        image=printed_canvas(source,(source.shape[1]+16,source.shape[0]+16))
        controls[key]=add('control__'+key,image,ref,'control',family=family,level=level,condition='native_tex_180dpi')
    old=json.loads(Path('docs/assets/v25/comparison.json').read_text())
    baseline=[r for r in old['predictions'] if r['model']=='PP-FormulaNet-S' and r.get('reference') is not None]
    full={}
    from experiments.v2.benchmark import SAMPLES
    for name,spec in SAMPLES.items():
        full[name]={'rgb':f'benchmark-output/v1_5-real/prepared/1080p/image/{name}.png','b0':f'benchmark-output/v1_5-real/results/1080p/{name}/B0.png'}
    for sample in json.loads(Path('experiments/v25/additional_regions.json').read_text())['samples']:
        full[sample['id']]={'rgb':sample['source_path'],'b0':f"benchmark-output/v25/b0/{sample['id']}/b0.png"}
    for item in baseline:
        rid=item['id'];image=cv2.imread(item['crop'])
        add('real__'+rid,image,item['reference'],'real_baseline',case=item['image']+'__'+item['region_id'],
            variant=item['variant'],condition='base',source_id=rid,source_bbox=item['bbox'],reuse_v25=True)
        if item['variant']=='b0':
            rendered=printed(item['reference'],rid)
            canvas=printed_canvas(rendered,(image.shape[1],image.shape[0]),ink_box(image))
            add('matched__'+rid,canvas,item['reference'],'matched_print',case=item['image']+'__'+item['region_id'],
                condition='matched_canvas_ink_bbox',matched_to='real__'+rid,source_bbox=item['bbox'])
    for name,rid in SENSITIVITY:
        for variant in ['rgb','b0']:
            source=cv2.imread(full[name][variant]);item=next(r for r in baseline if r['image']==name and r['region_id']==rid and r['variant']==variant)
            for condition in ['pad10','pad20','tight']+(['scale2'] if rid in {'symbolic_vector','cardinality_factorial'} else []):
                crop,box=condition_crop(source,item['bbox'],condition)
                add(f'sensitivity__{name}__{variant}__{rid}__{condition}',crop,item['reference'],'sensitivity',
                    case=name+'__'+rid,variant=variant,condition=condition,source_bbox=box,base_id='real__'+item['id'])
    for key in ALTERNATE_PRINTS:
        item=controls[key];source=cv2.imread(item['input']);a,b,c,d=ink_box(source)
        canvas=np.full((source.shape[0]*2,source.shape[1]*2,3),255,np.uint8)
        top=source.shape[0]//2;left=source.shape[1]//2
        canvas[top:top+source.shape[0],left:left+source.shape[1]]=source
        add('format__'+key+'__wide_canvas',canvas,item['reference'],'print_format',family=item['family'],condition='extra_blank_canvas',base_id=item['id'])
        if key in {'factorial','matrix33'}:
            add('format__'+key+'__inverted',255-source,item['reference'],'print_format',family=item['family'],condition='inverted',base_id=item['id'])
    # Fixed geometric row cuts; these are an oracle diagnostic, not automatic structure recovery.
    for key,refs in SPLIT_ROWS.items():
        item=controls[key];image=cv2.imread(item['input']);a,b,c,d=ink_box(image)
        for i,ref in enumerate(refs):
            top=b+round((d-b)*i/len(refs));bottom=b+round((d-b)*(i+1)/len(refs))
            add(f'split__{key}__row{i}',image[top:bottom,a:c].copy(),ref,'split_row',family=item['family'],
                parent_id=item['id'],row_index=i,condition='diagnostic_fixed_equal_row_crop',deployment_allowed=key.startswith('lines'))
    item=next(r for r in baseline if r['region_id']=='formula_system' and r['variant']=='b0')
    for i,ref in enumerate(['3x=1','x+2y=3']):
        path=Path('docs/assets/v25/crops')/(item['id']+f'__row{i}.png')
        add(f'split__hand_system__row{i}',cv2.imread(str(path)),ref,'split_row',family='multiline',
            parent_id='real__'+item['id'],row_index=i,condition='v25_geometric_split_retest',deployment_allowed=True)
    manifest={'design_version':1,'notice':'Diagnostic convenience sample. Fixed conditions, no GT in model requests. Existing V2.5 real predictions reused unchanged; new predictions are fresh.',
        'models':['PP-FormulaNet-S','PP-FormulaNet_plus-S','PP-FormulaNet-L','UniMERNet'],
        'margin_definition':'10/20 percent of original width/height on EACH side; clipped to source bounds',
        'matching_limit':'Same canvas and approximate aggregate ink footprint; font, stroke, local glyph size and line geometry are not perfectly matched.',
        'cases':rows}
    path=ROOT/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'manifest.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  manifest.json\n')
    print('Frozen',len(rows),'inputs;',sum(not r.get('reuse_v25') for r in rows),'fresh jobs/model')


if __name__=='__main__':main()
