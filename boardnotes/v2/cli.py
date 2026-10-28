"""V2 specified-region OCR, reviewable proposals, and JSON-to-PDF rendering."""
import argparse
import json
import os
from pathlib import Path
import sys

from .schema import load_document

ROOT=Path(__file__).resolve().parents[2]


def region(value):
    try:
        kind,box=value.split(':',1)
        return {'type':kind,'bbox':[int(v) for v in box.split(',')]}
    except ValueError as error:
        raise argparse.ArgumentTypeError('use text:x1,y1,x2,y2 or formula:x1,y1,x2,y2') from error


def model_options(parser):
    parser.add_argument('--ocr-python',type=Path,default=Path(os.getenv('BOARD_OCR_PYTHON',sys.executable)))
    parser.add_argument('--model-root',type=Path,default=ROOT/'models/v2')
    parser.add_argument('--language',choices=['en','ko'],default='en')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--timestamp',type=float,help='original video seconds; never milliseconds')


def config(args):
    if args.threads<1 or args.timeout<=0:
        raise ValueError('threads and timeout must be positive')
    return {'threads':args.threads,'language':args.language,
            'det_dir':str((args.model_root/'PP-OCRv5_mobile_det_infer').resolve()),
            'rec_dir':str((args.model_root/('en_PP-OCRv5_mobile_rec_infer' if args.language=='en' else 'korean_PP-OCRv5_mobile_rec_infer')).resolve()),
            'formula_dir':str((args.model_root/'PP-FormulaNet-S_infer').resolve())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    rec=sub.add_parser('recognize',help='Phase A: explicit source-coordinate regions')
    rec.add_argument('image',type=Path)
    choose=rec.add_mutually_exclusive_group(required=True)
    choose.add_argument('--regions',type=Path,help='regions list or document JSON; only bbox/type/id are consumed')
    choose.add_argument('--region',type=region,action='append')
    rec.add_argument('--output',type=Path,required=True)
    rec.add_argument('--page',type=int,default=0,help='Page index when image is a V1 v2_inputs manifest JSON')
    rec.add_argument('--source-variant',choices=['rgb','b0','b3'],default='rgb')
    rec.add_argument('--b0',type=Path,help='aligned black-writing binary PNG; never smoothed/cropped V1 PDF pages')
    rec.add_argument('--b3',type=Path,help='aligned black-writing binary PNG')
    rec.add_argument('--cache-dir',type=Path)
    rec.add_argument('--no-pdf',action='store_true')
    rec.add_argument('--font',type=Path)
    model_options(rec)
    det=sub.add_parser('detect',help='Phase B: OCR line proposals; all types require review')
    det.add_argument('image',type=Path);det.add_argument('--output',type=Path,required=True)
    model_options(det)
    pdf=sub.add_parser('render',help='regenerate PDF from edited JSON; no model imports/inference')
    pdf.add_argument('document',type=Path);pdf.add_argument('--output',type=Path,required=True)
    pdf.add_argument('--font',type=Path);pdf.add_argument('--source-image',type=Path,help='relocate preserved source image')
    args=parser.parse_args()
    try:
        if args.command=='render':
            from .pdf import render_pdf
            document=load_document(args.document)
            if args.source_image: document['source_image']=str(args.source_image.resolve())
            print(json.dumps({'pages':render_pdf(document,args.output,args.font),'output':str(args.output)}))
            return
        from .pipeline import recognize,detect
        if args.command=='detect':
            result=detect(args.image,args.output,args.ocr_python,config(args),args.timestamp,args.timeout)
            print(json.dumps({'output':str(args.output/'regions.json'),'candidates':len(result['regions']),
                              'status':'failed' if result['metadata']['worker']['items'][0]['status']=='failed' else 'needs_review'}))
            if result['metadata']['worker']['items'][0]['status']=='failed': raise SystemExit(1)
            return
        handoff=None
        if args.image.suffix.lower()=='.json':
            manifest=json.loads(args.image.read_text())
            if manifest.get('format')!='boardnotes-v2-inputs-1' or not 0<=args.page<len(manifest.get('pages',[])):
                raise ValueError('invalid V1 handoff manifest or page index')
            handoff=manifest['pages'][args.page]
            args.image=Path(handoff['image'])
            args.source_variant=handoff['variant']
            if args.timestamp is None: args.timestamp=handoff['timestamp']
        if args.regions:
            specs=json.loads(args.regions.read_text())
            regions=specs.get('regions') if isinstance(specs,dict) else specs
            if isinstance(specs,dict) and ('width' in specs or 'height' in specs):
                import cv2
                image=cv2.imread(str(args.image))
                if image is None or (specs.get('width'),specs.get('height'))!=(image.shape[1],image.shape[0]):
                    raise ValueError('region document dimensions do not match source; do not use resized/PDF coordinates')
        else:
            regions=[{'id':f'region_{i:04}',**r} for i,r in enumerate(args.region)]
        variants={args.source_variant:args.image}
        if args.b0: variants['b0']=args.b0
        if args.b3: variants['b3']=args.b3
        documents=recognize(args.image,regions,args.output,args.ocr_python,config(args),variants,args.timestamp,args.cache_dir,args.timeout)
        if handoff:
            from .schema import save_document
            for variant,document in documents.items():
                document['metadata']['v1_handoff']=handoff
                save_document(document,args.output/(variant+'.json'))
        summary={}
        for variant,document in documents.items():
            item={'json':str(args.output/(variant+'.json')),'failed':sum(r['status']=='failed' for r in document['regions'])}
            if not args.no_pdf:
                from .pdf import render_pdf
                try:
                    item['pages']=render_pdf(document,args.output/(variant+'.pdf'),args.font)
                except (ValueError,OSError) as error:
                    item['pdf_error']=str(error)
            summary[variant]=item
        print(json.dumps(summary,indent=2))
        if any(item.get('failed') or item.get('pdf_error') for item in summary.values()): raise SystemExit(1)
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))


if __name__=='__main__': main()
