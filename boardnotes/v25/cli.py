"""V2.5 automatic B0-first image/manifest processing and rendered mathematics."""
import argparse
import json
from pathlib import Path
import sys
from boardnotes.v2.schema import load_document
from .pipeline import configuration,process_image,process_manifest,process_video
from .pdf import render_pdf


def options(p):
    p.add_argument('--ocr-python',type=Path,required=True)
    p.add_argument('--v1-python',default=sys.executable)
    p.add_argument('--formula-model',choices=['PP-FormulaNet-S','PP-FormulaNet_plus-S','PP-FormulaNet-L','UniMERNet'],default='PP-FormulaNet-S')
    p.add_argument('--model-root',type=Path,default=Path('models/v2'))
    p.add_argument('--threads',type=int,default=2);p.add_argument('--max-rss-mib',type=float,default=6000)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--split-formulas',action='store_true')
    p.add_argument('--formula-context-margin',type=float,default=0.,help='experimental additional formula crop; fraction per side in [0,0.5], original prediction retained')
    p.add_argument('--hints',type=Path,help='JSON mapping region ids to explicit aligned/cases/protected structure hints')
    p.add_argument('--cache-dir',type=Path,default=Path('benchmark-output/v25/recognition-cache'))
    p.add_argument('--font',type=Path)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    image=sub.add_parser('image');image.add_argument('input',type=Path);image.add_argument('--output',type=Path,required=True)
    image.add_argument('--b0',type=Path);image.add_argument('--source-variant',choices=['rgb','b0'],default='rgb')
    image.add_argument('--regions',type=Path,help='reviewed JSON skips automatic detection; unchanged crops use cache')
    image.add_argument('--timestamp',type=float)
    options(image)
    manifest=sub.add_parser('manifest');manifest.add_argument('input',type=Path);manifest.add_argument('--output',type=Path,required=True);options(manifest)
    video=sub.add_parser('video');video.add_argument('input',type=Path);video.add_argument('--output',type=Path,required=True)
    video.add_argument('--max-width',type=int,default=1280);video.add_argument('--fps',type=float,default=1.);options(video)
    render=sub.add_parser('render');render.add_argument('document',type=Path);render.add_argument('--output',type=Path,required=True);render.add_argument('--font',type=Path)
    args=p.parse_args()
    try:
        if args.command=='render':
            print(json.dumps({'pages':render_pdf(load_document(args.document),args.output,args.font)}));return
        if min(args.threads,args.max_rss_mib,args.timeout)<=0:raise ValueError('limits must be positive')
        if not 0<=args.formula_context_margin<=.5:raise ValueError('formula context margin must be between 0 and 0.5 per side')
        config=configuration(args.model_root,args.formula_model,args.threads,args.max_rss_mib)
        hints=json.loads(args.hints.read_text()) if args.hints else None
        common={'v1_python':args.v1_python,'split_formulas':args.split_formulas,'structure_hints':hints,
                'cache_dir':args.cache_dir,'font':args.font,'timeout':args.timeout,'formula_context_margin':args.formula_context_margin}
        if args.command=='video':
            reports=process_video(args.input,args.output,args.ocr_python,config,max_width=args.max_width,fps=args.fps,**common)
            print(json.dumps({'output':str(args.output),'pages_processed':len(reports['v25_pages'])}))
            if any(r['failed_recognitions'] for r in reports['v25_pages']):raise SystemExit(1)
        elif args.command=='manifest':
            reports=process_manifest(args.input,args.output,args.ocr_python,config,**common)
            print(json.dumps({'pages_processed':len(reports),'output':str(args.output)}))
            if any(r['failed_recognitions'] for r in reports):raise SystemExit(1)
        else:
            specs=json.loads(args.regions.read_text()) if args.regions else None
            if isinstance(specs,dict):
                import cv2
                original=cv2.imread(str(args.input))
                if 'width' in specs and (specs['width'],specs['height'])!=(original.shape[1],original.shape[0]):
                    raise ValueError('reviewed regions must be in original input coordinates')
                specs=specs['regions']
            doc,report=process_image(args.input,args.output,args.ocr_python,config,b0=args.b0,regions=specs,
                                     timestamp=args.timestamp,source_variant=args.source_variant,**common)
            print(json.dumps(report,indent=2))
            if report['failed_recognitions']:raise SystemExit(1)
    except (ValueError,OSError,KeyError,RuntimeError) as error:
        p.error(str(error))


if __name__=='__main__':main()
