"""Render all recorded outputs and produce condition-separated evidence."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import textwrap
from boardnotes.v25.rendering import render_latex
from boardnotes.v25.splitting import assemble_rows
from .evaluate import compare,visual_similarity


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();root=Path('docs/assets/v251')
    manifest=json.loads((root/'manifest.json').read_text());cases=manifest['cases'];byid={r['id']:r for r in cases}
    models=[m for m in manifest['models'] if (root/'predictions'/(m+'.json')).exists()]
    if not args.partial and len(models)!=4:raise ValueError('all four model runs must finish; use --partial for interim diagnostics')
    render_cache={}
    def render(value):
        key=hashlib.sha256((value or '').encode()).hexdigest()
        if key not in render_cache:
            record=render_latex(value,root/'rendered'/(key+'.png'))
            if record.get('path'):record['path']=str(Path(record['path']).relative_to(Path.cwd()))
            render_cache[key]=record
        return render_cache[key]
    rows=[];runs=[]
    for model in models:
        data=json.loads((root/'predictions'/(model+'.json')).read_text());runs.append({'model':model,**data['run']})
        for pred in data['predictions']:
            case=byid[pred['id']];ref=render(case['reference']);actual=render(pred['latex'])
            score=compare(case['reference'],pred['latex'],pred['status'])
            visual=visual_similarity(ref['path'],actual['path']) if ref['status']==actual['status']=='rendered' else None
            rows.append({**case,**pred,'model':model,**score,'reference_render':ref,'prediction_render':actual,'visual_similarity':visual})
        print('Rendered/evaluated',model,flush=True)
    indexed={(r['model'],r['id']):r for r in rows}
    probe=json.loads((root/'internal-preprocessing.json').read_text())
    tensors={(r['model'],r['id']):r['tensor_sha256'] for r in probe['records']}
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['group'],r.get('variant',''),r.get('condition',''))].append(r)
    summary=[]
    for (m,g,v,c),selected in groups.items():
        summary.append({'model':m,'group':g,'variant':v,'condition':c,'n':len(selected),
            'strict_exact_matches':sum(r['strict_exact_match'] for r in selected),'mean_edit_distance':sum(r['edit_distance'] for r in selected)/len(selected),
            'rendering':dict(Counter(r['prediction_render']['status'] for r in selected)),
            'matrix_cells_matches':sum(r['matrix_cells_match'] is True for r in selected),'matrix_denominator':sum(r['matrix_cells_match'] is not None for r in selected),
            'runtime_failures':sum(r['status']=='failed' for r in selected)})
    comparisons=[]
    for m in models:
        for c in cases:
            if c['group'] not in {'sensitivity','print_format','matched_print'}:continue
            base=c.get('base_id') or c.get('matched_to');a=indexed[m,base];b=indexed[m,c['id']]
            comparisons.append({'model':m,'base_id':base,'condition_id':c['id'],'group':c['group'],'condition':c['condition'],
                'internal_tensor_identical':tensors[m,base]==tensors[m,c['id']],
                'prediction_string_identical':a['latex']==b['latex'],
                'strict_before':a['strict_exact_match'],'strict_after':b['strict_exact_match'],
                'matrix_cells_before':a['matrix_cells_match'],'matrix_cells_after':b['matrix_cells_match'],
                'edit_distance_before':a['edit_distance'],'edit_distance_after':b['edit_distance']})
    assemblies=[]
    for m in models:
        parents=defaultdict(list)
        for c in cases:
            if c['group']=='split_row':parents[c['parent_id']].append(c)
        for parent,parts in parents.items():
            parts.sort(key=lambda c:c['row_index']);values=[indexed[m,c['id']]['latex'] or '' for c in parts]
            allowed=all(c['deployment_allowed'] for c in parts)
            assembly=assemble_rows(values,'aligned') if allowed else {'status':'diagnostic_only','latex':None,'error':'Rows are matrix cells; aligned recombination would destroy matrix structure.'}
            rendered=render(assembly['latex']) if assembly.get('latex') else None
            assemblies.append({'model':m,'parent_id':parent,'row_ids':[c['id'] for c in parts],'row_predictions':values,
                'all_rows_strict':all(indexed[m,c['id']]['strict_exact_match'] for c in parts),'deployment_allowed':allowed,
                'whole_prediction':indexed[m,parent]['latex'],'whole_strict':indexed[m,parent]['strict_exact_match'],
                'assembly':assembly,'rendering':rendered})
    result={'notice':'Diagnostic conditions, not optimized model selection or general semantic equivalence. See manual-review.json for reviewed cases.',
        'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'complete':len(models)==4,
        'summary':summary,'runs':runs,'condition_comparisons':comparisons,'assemblies':assemblies,'predictions':rows}
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fields=['model','id','group','family','level','variant','condition','reference','latex','status','evidence','strict_exact_match','edit_distance','matrix_shape_match','matrix_cells_match','factorial_count_match','scripts_lexical_match','multiline_rows_match','vector_as_fraction_suspected']
    with (root/'evaluation.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    make_pdfs(root,cases,models,indexed)
    print('Saved',len(rows),'predictions,',len(assemblies),'assemblies',flush=True)


def make_pdfs(root,cases,models,indexed):
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    def picture(c,path,x,y,w,h):
        if not path or not Path(path).exists():return
        with Image.open(path) as im:iw,ih=im.size
        scale=min(w/iw,h/ih);c.drawImage(ImageReader(str(path)),x+(w-iw*scale)/2,y+(h-ih*scale)/2,iw*scale,ih*scale,mask='auto')
    def literal(c,value,x,y,width=110,lines=2):
        wrapped=textwrap.wrap(value or '(empty)',width=width,replace_whitespace=False,drop_whitespace=False) or ['(empty)']
        c.setFont('Courier',6)
        for i,line in enumerate(wrapped[:lines]):c.drawString(x,y-i*8,line.encode('ascii','replace').decode())
        if len(wrapped)>lines:c.drawString(x,y-lines*8,'[Full literal source in evaluation.json]')
    def pages(file,selected):
        c=canvas.Canvas(str(root/file),pagesize=(842,595));c.setTitle('V2.51 fixed-condition formula diagnostics')
        for case in selected:
            key=case['id'];first=indexed[models[0],key]
            c.setFont('Helvetica-Bold',10);c.drawString(20,575,key)
            c.setFont('Helvetica',8);c.drawString(20,560,case['group']+' | '+case.get('condition','')+' | no GT-based routing | all models, same input')
            c.drawString(20,545,'ACTUAL INPUT');c.drawString(435,545,'REFERENCE RENDERING')
            picture(c,case['input'],20,435,390,100);picture(c,first['reference_render'].get('path'),435,435,385,100)
            literal(c,case['reference'],20,423,128,1)
            for i,m in enumerate(models):
                r=indexed[m,key];y=335-i*81
                c.setFont('Helvetica-Bold',10);c.drawString(20,y+55,m)
                c.setFont('Helvetica',8);c.drawString(20,y+42,f"Strict EM: {r['strict_exact_match']} | Render: {r['prediction_render']['status']} | Matrix cells: {r['matrix_cells_match']}")
                picture(c,r['prediction_render'].get('path'),340,y+20,480,45)
                literal(c,r['latex'],20,y+11,155,2)
            c.setFont('Helvetica',7);c.drawString(20,15,'Render success / matrix cell agreement / raster similarity are separate observations, not mathematical correctness verdicts.')
            c.showPage()
        c.save()
    pages('all-conditions.pdf',cases)
    pages('controls.pdf',[c for c in cases if c['group']=='control'])
    paired=canvas.Canvas(str(root/'comparison.pdf'),pagesize=(842,595))
    for case in [c for c in cases if c['group']=='matched_print']:
        real=next(c for c in cases if c['id']==case['matched_to']);key=case['id']
        paired.setFont('Helvetica-Bold',11);paired.drawString(20,575,case['case'])
        paired.setFont('Helvetica',9);paired.drawString(20,555,'ACTUAL V34 HANDWRITING');paired.drawString(435,555,'PRINTED REFERENCE: matched canvas / approximate ink footprint')
        picture(paired,real['input'],20,448,385,100);picture(paired,case['input'],435,448,385,100)
        literal(paired,case['reference'],20,435,130,1)
        for i,m in enumerate(models):
            y=335-i*81;a=indexed[m,real['id']];b=indexed[m,key]
            paired.setFont('Helvetica-Bold',9);paired.drawString(20,y+60,m)
            for r,x in [(a,20),(b,435)]:
                paired.setFont('Helvetica',7);paired.drawString(x,y+49,f"EM: {r['strict_exact_match']} | cells: {r['matrix_cells_match']} | render: {r['prediction_render']['status']}")
                picture(paired,r['prediction_render'].get('path'),x,y+16,385,32)
                literal(paired,r['latex'],x,y+7,85,2)
        paired.setFont('Helvetica',7);paired.drawString(20,15,'Same GT and model. Printed/handwritten rendering geometry is approximate, not a perfectly isolated font intervention.')
        paired.showPage()
    paired.save()


if __name__=='__main__':main()
