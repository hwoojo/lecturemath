"""Paddle-only subprocess protocol; never imported by the V1 pipeline.

Write results to a file so upstream progress messages cannot corrupt JSON IPC.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time


def versions():
    result={}
    for package in ['paddlepaddle','paddleocr','paddlex','numpy','opencv-contrib-python']:
        try:
            result[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package]=None
    return result


def as_dict(result):
    data=result.json if hasattr(result,'json') else dict(result)
    if callable(data):
        data=data()
    if isinstance(data,str):
        data=json.loads(data)
    return data.get('res',data)


def text_result(results):
    texts=[];scores=[];boxes=[]
    for result in results:
        data=as_dict(result)
        texts.extend(str(x) for x in data.get('rec_texts',[]))
        scores.extend(float(x) for x in data.get('rec_scores',[]))
        raw=data.get('rec_boxes',[])
        boxes.extend([[float(x) for x in box] for box in raw])
    raw={'texts':list(texts),'scores':list(scores),'boxes':list(boxes)}
    # Group vertically overlapping boxes before sorting left-to-right. Paddle's
    # raw detector order is not always reading order, even for a single title.
    entries=[(text.strip(),scores[i] if i<len(scores) else None,boxes[i] if i<len(boxes) else None)
             for i,text in enumerate(texts) if text.strip()]
    rows=[]
    if all(e[2] is not None for e in entries):
        for entry in sorted(entries,key=lambda e:(e[2][1],e[2][0])):
            box=entry[2];chosen=None
            for row in rows:
                top=min(e[2][1] for e in row);bottom=max(e[2][3] for e in row)
                overlap=max(0,min(bottom,box[3])-max(top,box[1]))
                if overlap>=.5*min(bottom-top,box[3]-box[1]):
                    chosen=row;break
            if chosen is None: rows.append([entry])
            else: chosen.append(entry)
        rows.sort(key=lambda row:min(e[2][1] for e in row))
        for row in rows: row.sort(key=lambda e:e[2][0])
    else:
        rows=[[entry] for entry in entries]
    ordered=[e for row in rows for e in row]
    scores=[e[1] for e in ordered if e[1] is not None]
    return {'content':'\n'.join(' '.join(e[0] for e in row) for row in rows),'latex':None,
            'confidence':sum(scores)/len(scores) if scores else None,
            'line_texts':[e[0] for e in ordered],'line_scores':scores,
            'line_boxes':[e[2] for e in ordered if e[2] is not None],
            'raw_engine_lines':raw,'reading_order':'vertical overlap >= 0.5; rows top-to-bottom, boxes left-to-right; heuristic',
            'confidence_definition':'arithmetic mean of nonempty engine line recognition scores; not calibrated document confidence'}


def load_model(kind, config):
    common={'device':'cpu','cpu_threads':config.get('threads',2),'enable_mkldnn':False}
    if kind in {'text','detect'}:
        from paddleocr import PaddleOCR
        kwargs={'text_detection_model_name':'PP-OCRv5_mobile_det',
                'text_recognition_model_name':'en_PP-OCRv5_mobile_rec' if config.get('language','en')=='en' else 'korean_PP-OCRv5_mobile_rec',
                'use_doc_orientation_classify':False,'use_doc_unwarping':False,'use_textline_orientation':False,**common}
        for source,target in [('det_dir','text_detection_model_dir'),('rec_dir','text_recognition_model_dir')]:
            if config.get(source): kwargs[target]=config[source]
        return PaddleOCR(**kwargs)
    if kind=='formula':
        from paddleocr import FormulaRecognition
        kwargs={'model_name':'PP-FormulaNet-S',**common}
        if config.get('formula_dir'): kwargs['model_dir']=config['formula_dir']
        return FormulaRecognition(**kwargs)
    raise ValueError('unknown worker kind')


def run(request):
    kind=request['kind'];config=request.get('config',{})
    started=time.perf_counter();items=[]
    try:
        model=load_model(kind,config)
        loaded=time.perf_counter();load_error=None
    except Exception as error:
        model=None;loaded=time.perf_counter();load_error=f'{type(error).__name__}: {error}'
    for job in request['jobs']:
        start=time.perf_counter()
        try:
            if model is None:
                raise RuntimeError(load_error)
            if kind=='formula':
                results=list(model.predict(input=job['image'],batch_size=1))
                values=[str(as_dict(r).get('rec_formula','')) for r in results]
                value={'content':None,'latex':'\n'.join(values),'confidence':None}
            else:
                results=list(model.predict(input=job['image']))
                value=text_result(results)
            nonempty=value.get('latex') if kind=='formula' else value.get('content')
            items.append({'id':job['id'],'status':'ok' if nonempty and nonempty.strip() else 'empty',
                          'error':None,'seconds':time.perf_counter()-start,**value})
        except Exception as error:
            items.append({'id':job['id'],'status':'failed','content':None,'latex':None,'confidence':None,
                          'error':f'{type(error).__name__}: {error}','seconds':time.perf_counter()-start})
    return {'items':items,'model_load_seconds':loaded-started,'model_load_error':load_error,
            'process_seconds':time.perf_counter()-started,'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'versions':versions(),'kind':kind,'config':config}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    if args.probe:
        response={'versions':versions(),'protocol':1}
    else:
        if not args.request:
            parser.error('--request is required')
        request=json.loads(args.request.read_text())
        response=run(request)
    args.response.write_text(json.dumps(response,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
