"""Package the measured run, portable JSON samples, and representative PDFs."""
import argparse
import json
from pathlib import Path
import shutil
import textwrap
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from boardnotes.v2.schema import load_document
from boardnotes.v2.pdf import render_pdf


def annotated_contact(sample,region_id,records,output):
    docs={variant:load_document(sample/(variant+'.json')) for variant in ['rgb','b0','b3']}
    regions={variant:next(r for r in doc['regions'] if r['id']==region_id) for variant,doc in docs.items()}
    first=regions['rgb'];x1,y1,x2,y2=first['bbox']
    w,h=(x2-x1)*2,(y2-y1)*2
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    title_font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    wrap=max(20,(w-16)//9)
    captions={}
    names={'rgb':'Original RGB','b0':'FCN-LectureNet V34','b3':'LectureNet-guided Sauvola'}
    for variant,r in regions.items():
        record=next(row for row in records if row['variant']==variant and row['region_id']==region_id)
        prediction=r['content'] if r['type']=='text' else r['latex']
        captions[variant]=textwrap.wrap('Reference: '+record['reference'],wrap)+['']+textwrap.wrap('Prediction: '+str(prediction).replace('\n',' '),wrap)+[f"Status: {r['status']}; formula review required" if r['type']=='formula' else f"Status: {r['status']}"]
    caption_h=max(len(lines) for lines in captions.values())*22+15
    canvas=Image.new('RGB',(w*3,h+caption_h+100),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,5),f'REAL {sample.name} | {region_id} | same source crop | nearest 2x',font=title_font,fill='black')
    for col,(variant,r) in enumerate(regions.items()):
        left=col*w
        draw.text((left+8,38),names[variant],font=title_font,fill='black')
        with Image.open(r['crop_image']) as crop:
            crop=crop.convert('RGB').resize((w,h),Image.Resampling.NEAREST)
            canvas.paste(crop,(left,72))
        for line,text in enumerate(captions[variant]):
            draw.text((left+8,82+h+line*22),text,font=font,fill='black')
    canvas.save(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('benchmark-output/v2/real-final'))
    p.add_argument('--output',type=Path,default=Path('docs/assets/v2'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads((args.input/'comparison.json').read_text())
    for name in ['comparison.json','comparison.csv']:
        shutil.copy2(args.input/name,args.output/name)
    for source in args.input.glob('*.png'):
        shutil.copy2(source,args.output/source.name)
    for run in data['runs']:
        name=run['image'];dest=args.output/name;dest.mkdir(exist_ok=True)
        for variant in ['rgb','b0','b3']:
            doc=json.loads((args.input/name/(variant+'.json')).read_text())
            shutil.copy2(doc['source_image'],dest/'source.png')
            doc['source_image']='source.png'
            for i,region in enumerate(doc['regions']):
                target=dest/'crops'/variant/f'region_{i:04}.png'
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(region['crop_image'],target)
                region['crop_image']=str(target.relative_to(dest))
            (dest/(variant+'.json')).write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    for run in data['runs']:
        name=run['image']
        records=[r for r in data['metrics'] if r['image']==name]
        for region_id in {r['region_id'] for r in records}:
            annotated_contact(args.output/name,region_id,records,args.output/(name+'_'+region_id+'.png'))
    # One representative frame, three inputs. All other PDFs remain in the run directory.
    writer=PdfWriter()
    first=data['runs'][0]['image']
    for variant in ['rgb','b0','b3']:
        doc=load_document(args.output/first/(variant+'.json'))
        output=args.output/first/(variant+'.pdf')
        render_pdf(doc,output)
        writer.append(PdfReader(output))
    with (args.output/'comparison.pdf').open('wb') as stream: writer.write(stream)
    shutil.copy2('models/v2/models.json',args.output/'model_provenance.json')
    print(args.output)


if __name__=='__main__':main()
