"""Package actual oracle predictions and render every formula without rewriting it."""
import csv
import json
from pathlib import Path
import shutil
from collections import Counter
from boardnotes.v25.rendering import render_latex


def main():
    root=Path('docs/assets/v25');root.mkdir(parents=True,exist_ok=True)
    rows=[];runs=[]
    for stage in ['stage01','stage2','stage3','heldout-formulas']:
        source=Path('benchmark-output/v25')/stage
        data=json.loads((source/'results.json').read_text())
        runs.extend({'stage':stage,**r} for r in data['runs'])
        for item in data['predictions']:
            record={'stage':stage,**item}
            rid=item['id'];name=item['model']
            crop=source/'crops'/(rid+'.png')
            if crop.exists():
                destination=root/'crops'/(rid+'.png');destination.parent.mkdir(exist_ok=True)
                shutil.copy2(crop,destination);record['crop']=str(destination)
            record['rendering']=render_latex(item.get('latex'),root/'rendered'/name/(rid+'.png'))
            if item.get('reference') is not None:
                record['reference_rendering']=render_latex(item['reference'],root/'references'/(rid+'.png'))
            rows.append(record)
    summary=[]
    for stage in ['development-regression','evaluation-unseen-lectures']:
        for name in dict.fromkeys(r['model'] for r in rows):
            for variant in ['rgb','b0']:
                selected=[r for r in rows if r.get('split')==stage and r['model']==name and r.get('variant')==variant and r.get('reference') is not None]
                summary.append({'split':stage,'model':name,'input':('RGB' if variant=='rgb' else 'FCN-LectureNet V34'),
                    'count':len(selected),'strict_exact_matches':sum(r.get('exact_match') is True for r in selected),
                    'mean_edit_distance':sum(r.get('edit_distance') or 0 for r in selected)/max(1,len(selected)),
                    'rendering':dict(Counter(r['rendering']['status'] for r in selected))})
    (root/'comparison.json').write_text(json.dumps({'notice':'Oracle ROI, strict whitespace-normalized strings; rendering is not semantic accuracy. All original predictions retained.',
        'summary':summary,'runs':runs,'predictions':rows},ensure_ascii=False,indent=2)+'\n')
    with (root/'comparison.csv').open('w',newline='') as f:
        fields=['stage','model','id','variant','reference','latex','exact_match','edit_distance']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    c=canvas.Canvas(str(root/'comparison.pdf'),pagesize=(842,595))
    def picture(path,x,y,w,h):
        if not path or not Path(path).exists():return
        with Image.open(path) as im:iw,ih=im.size
        scale=min(w/iw,h/ih)
        c.drawImage(ImageReader(str(path)),x,y+(h-ih*scale)/2,iw*scale,ih*scale,mask='auto')
    ids=list(dict.fromkeys(r['id'] for r in rows if r.get('variant')=='b0' and r.get('reference') is not None))
    for rid in ids:
        selected=[r for r in rows if r['id']==rid]
        first=selected[0]
        c.setFont('Helvetica-Bold',13);c.drawString(24,570,rid)
        c.setFont('Helvetica',9);c.drawString(24,550,'Input: FCN-LectureNet V34 | Oracle ROI | Full RGB predictions and source LaTeX in comparison.json')
        picture(first.get('crop'),24,452,790,85)
        for i,(label,item) in enumerate([('Reference',{'latex':first['reference'],'rendering':first['reference_rendering']})]+[(r['model'],r) for r in selected]):
            y=368-i*72
            c.setFont('Helvetica-Bold',10);c.drawString(24,y+55,label)
            status=item['rendering']['status']
            c.setFont('Helvetica',8);c.drawString(24,y+41,'Render: '+status+(' | Strict EM: '+str(item.get('exact_match')) if label!='Reference' else ''))
            picture(item['rendering'].get('path'),250,y+12,555,50)
            literal=item.get('latex') or '(empty)'
            c.setFont('Courier',6);c.drawString(24,y+1,literal[:185].encode('ascii','replace').decode()+(' ... [full JSON]' if len(literal)>185 else ''))
        c.setFont('Helvetica',8);c.drawString(24,22,'Rendering success does not establish correctness. String mismatch may reflect equivalent notation. No GT-based runtime routing.')
        c.showPage()
    c.save()
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
