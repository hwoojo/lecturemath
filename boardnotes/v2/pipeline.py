"""Specified-region recognition, isolated model processes, editable JSON outputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np
from .schema import validate_regions, save_document

MODEL_NAMES={'text':'PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec','formula':'PP-FormulaNet-S'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def worker_call(python,kind,jobs,config,folder,timeout=300,probe=False):
    """Every call ends its process, so text and formula weights cannot coexist."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=folder/(kind+'_request.json');response=folder/(kind+'_response.json')
    log=folder/(kind+'.log')
    command=[str(python),'-m','boardnotes.v2.worker','--response',str(response.resolve())]
    if probe: command.append('--probe')
    else:
        request.write_text(json.dumps({'kind':kind,'config':config,'jobs':jobs},indent=2))
        command+=['--request',str(request.resolve())]
    env=os.environ.copy()
    env['OMP_NUM_THREADS']=str(config.get('threads',2))
    runtime_cache=Path(__file__).resolve().parents[2]/'benchmark-output/v2/runtime-cache'
    runtime_cache.mkdir(parents=True,exist_ok=True)
    env.setdefault('PADDLE_PDX_CACHE_HOME',str(runtime_cache/'paddlex'))
    env.setdefault('PADDLE_HOME',str(runtime_cache/'paddle'))
    env.setdefault('XDG_CACHE_HOME',str(runtime_cache))
    if (kind=='layout' and config.get('layout_dir')) or (kind=='formula' and config.get('formula_dir')) or (kind in {'text','detect'} and config.get('det_dir') and config.get('rec_dir')):
        env['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    # Local model directories are mandatory in documented offline/reproducible runs.
    started=time.perf_counter()
    try:
        with log.open('w') as stream:
            if config.get('max_rss_mib'):
                from ..v25.resources import run_bounded
                result=run_bounded(command,stdout=stream,stderr=subprocess.STDOUT,env=env,
                                   cwd=Path(__file__).resolve().parents[2],timeout=timeout,max_rss_mib=config['max_rss_mib'])
            else:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,
                                      cwd=Path(__file__).resolve().parents[2],timeout=timeout,check=False)
        if result.returncode:
            raise RuntimeError(f'OCR subprocess exit {result.returncode}; see {log.resolve()}')
        if not response.exists():
            raise RuntimeError('OCR subprocess returned no response file')
        response_data=json.loads(response.read_text())
        if not probe:
            expected={job['id'] for job in jobs}
            observed=[item.get('id') for item in response_data.get('items',[])]
            if len(observed)!=len(expected) or set(observed)!=expected:
                raise ValueError('OCR worker returned missing/duplicate/unexpected job ids')
        response_data['subprocess_wall_seconds']=time.perf_counter()-started
        if hasattr(result,'supervisor_peak_rss_mib'): response_data['supervisor_peak_rss_mib']=result.supervisor_peak_rss_mib
        return response_data
    except (OSError,subprocess.TimeoutExpired,ValueError,RuntimeError) as error:
        return {'versions':{},'error':f'{type(error).__name__}: {error}',
                'items':[{'id':job['id'],'status':'failed','content':None,'latex':None,'confidence':None,
                          'error':f'{type(error).__name__}: {error}'} for job in jobs],
                'subprocess_wall_seconds':time.perf_counter()-started}


def recognize(source_image,regions,output,ocr_python,config=None,variants=None,timestamp=None,cache_dir=None,timeout=300):
    config=dict(config or {})
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('V2 output directory must be empty; use a new directory or render existing JSON')
    source_image=Path(source_image)
    source=cv2.imread(str(source_image))
    if source is None: raise ValueError('cannot decode source RGB image')
    height,width=source.shape[:2]
    validate_regions(regions,width,height)
    if min(width,height)<1: raise ValueError('empty source')
    output.mkdir(parents=True,exist_ok=True)
    source_copy=output/'source.png'
    if not cv2.imwrite(str(source_copy),source): raise OSError(source_copy)
    variants=variants or {'rgb':source_image}
    if not variants or any(name not in {'rgb','b0','b3'} for name in variants):
        raise ValueError('variants must be rgb/b0/b3')
    cache=Path(cache_dir) if cache_dir else output/'cache'
    cache.mkdir(parents=True,exist_ok=True)
    probe=worker_call(ocr_python,'probe',[],config,output/'worker',timeout,probe=True)
    fingerprint={'versions':probe.get('versions',{}),'protocol':probe.get('protocol'),
                 'worker_sha256':sha(Path(__file__).with_name('worker.py'))}
    for key in ['det_dir','rec_dir','formula_dir']:
        if config.get(key):
            directory=Path(config[key])
            fingerprint[key]={p.name:sha(p) for p in sorted(directory.iterdir()) if p.is_file()} if directory.is_dir() else 'missing'
    documents={};jobs={'text':[],'formula':[]};mapping={};cached={}
    for vi,(variant,path) in enumerate(variants.items()):
        image=cv2.imread(str(path))
        if image is None or image.shape[:2]!=(height,width):
            raise ValueError(f'{variant}: input must match complete source image coordinates and resolution')
        if variant!='rgb' and (Path(path).suffix.lower()!='.png' or Path(path).read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n'):
            raise ValueError('b0/b3 must be PNG files, not JPEG or renamed JPEG')
        if variant!='rgb' and (not np.isin(image,[0,255]).all() or not np.array_equal(image[:,:,0],image[:,:,1]) or not np.array_equal(image[:,:,0],image[:,:,2])):
            raise ValueError('b0/b3 must be strictly binary PNG, black writing on white background')
        doc={'version':'2.0','source_image':str(source_copy.resolve()),'width':width,'height':height,'timestamp':timestamp,'regions':[],
             'metadata':{'source_original':str(source_image.resolve()),'source_sha256':sha(source_image),
                         'input_variant':variant,'recognition_image_sha256':sha(path),'preprocessing':'none; half-open bbox crop only',
                         'config':config,'worker_fingerprint':fingerprint,'probe_error':probe.get('error'),
                         'reading_order':'manual JSON array order; no semantic order inference'}}
        for i,region in enumerate(regions):
            item={'id':str(region['id']),'type':region['type'],'bbox':list(region['bbox']),
                  'status':'needs_review' if region['type']=='unknown' else 'preserved','content':None,'latex':None,
                  'confidence':None,'model':None,'error':None}
            x1,y1,x2,y2=item['bbox'];crop=image[y1:y2,x1:x2]
            crop_path=output/'crops'/variant/f'region_{i:04}.png'
            crop_path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(crop_path),crop): raise OSError(crop_path)
            item['crop_image']=str(crop_path.resolve())
            doc['regions'].append(item)
            if item['type'] not in jobs: continue
            kind=item['type'];job_id=f'v{vi}-r{i}'
            cache_key=hashlib.sha256(json.dumps({'crop_sha256':sha(crop_path),'kind':kind,'config':config,
                                               'fingerprint':fingerprint},sort_keys=True).encode()).hexdigest()
            cache_path=cache/(cache_key+'.json')
            mapping[job_id]=(item,cache_path)
            if cache_path.exists():
                result=json.loads(cache_path.read_text());result['id']=job_id
                cached[job_id]=result;item['cache_hit']=True
            else:
                jobs[kind].append({'id':job_id,'image':str(crop_path.resolve())});item['cache_hit']=False
            item['model']=(config.get('formula_model','PP-FormulaNet-S') if kind=='formula' else 'PP-OCRv5_mobile_det + korean_PP-OCRv5_mobile_rec' if config.get('language')=='ko' else MODEL_NAMES[kind])
        documents[variant]=doc
    stage_reports={};results=list(cached.values())
    for kind in ['text','formula']:
        if not jobs[kind]: continue
        report=worker_call(ocr_python,kind,jobs[kind],config,output/'worker',timeout)
        stage_reports[kind]={k:v for k,v in report.items() if k!='items'}
        results+=report['items']
    for result in results:
        item,cache_path=mapping[result['id']]
        if result['status'] not in {'ok','empty','failed'}:
            raise ValueError('invalid worker recognition status')
        item.update({k:v for k,v in result.items() if k!='id'})
        item['review_required']=item['type']=='formula' or item['status']!='ok' or (item.get('confidence') is not None and item['confidence']<.8)
        if result['status']!='failed' and not item['cache_hit']:
            cache_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        if 'line_boxes' in item:
            x,y,x2,y2=item['bbox'];cw,ch=x2-x,y2-y
            boxes=item['line_boxes']
            if any(len(box)!=4 or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in box) for box in boxes):
                raise ValueError('invalid OCR sub-boxes')
            item['line_boxes']=[[x+max(0,min(cw,int(np.floor(b[0])))),y+max(0,min(ch,int(np.floor(b[1])))),
                                 x+max(0,min(cw,int(np.ceil(b[2])))),y+max(0,min(ch,int(np.ceil(b[3]))))] for b in boxes]
    for variant,doc in documents.items():
        doc['metadata']['stages']=stage_reports
        save_document(doc,output/(variant+'.json'))
    return documents


def detect(source_image,output,ocr_python,config=None,timestamp=None,timeout=300):
    """OCR proposes text-line candidates only; every type needs human review."""
    config=config or {};output=Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('detection output directory must be empty')
    image=cv2.imread(str(source_image))
    if image is None: raise ValueError('cannot decode source')
    h,w=image.shape[:2];output.mkdir(parents=True,exist_ok=True)
    local=output/'source.png';cv2.imwrite(str(local),image)
    report=worker_call(ocr_python,'detect',[{'id':'page','image':str(local.resolve())}],config,output/'worker',timeout)
    value=report['items'][0];regions=[]
    boxes=value.get('line_boxes',[]);texts=value.get('line_texts',[]);scores=value.get('line_scores',[])
    if len(texts)!=len(boxes) or len(scores) not in (0,len(boxes)):
        value.update(status='failed',error='inconsistent OCR proposal boxes/texts/scores')
        boxes=[]
    for index,box in enumerate(boxes):
        x1,y1,x2,y2=[int(round(v)) for v in box]
        x1,x2=max(0,min(w,x1)),max(0,min(w,x2));y1,y2=max(0,min(h,y1)),max(0,min(h,y2))
        if x1>=x2 or y1>=y2: continue
        # Duplicate candidates must not be silently merged into guessed formulas.
        if any(all(abs(a-b)<=2 for a,b in zip([x1,y1,x2,y2],r['bbox'])) for r in regions): continue
        regions.append({'id':f'candidate_{index:04}','type':'unknown','bbox':[x1,y1,x2,y2],
                        'status':'needs_review','suggested_type':'text','candidate_text':value.get('line_texts',[])[index],
                        'confidence':value.get('line_scores',[])[index] if index<len(value.get('line_scores',[])) else None,
                        'model':'PP-OCRv5_mobile_det + recognition','content':None,'latex':None,'error':None})
    regions.sort(key=lambda r:(r['bbox'][1]//20,r['bbox'][0]))
    doc={'version':'2.0','source_image':str(local.resolve()),'width':w,'height':h,'timestamp':timestamp,'regions':regions,
         'metadata':{'automation':'OCR text-line proposals only; formula classification unavailable; unboxed content remains in source image',
                     'reading_order':'heuristic row bands, not mathematical layout understanding','worker':report}}
    save_document(doc,output/'regions.json')
    return doc
