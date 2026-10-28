"""Versioned, dependency-free contracts. Coordinates are original-image pixels."""
import json
import math
from pathlib import Path

TYPES = {'text', 'formula', 'other', 'unknown'}
STATUSES = {'ok', 'empty', 'failed', 'preserved', 'needs_review'}


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def validate_regions(regions, width, height):
    if not isinstance(regions,list):
        raise ValueError('regions must be a list')
    ids=set()
    for region in regions:
        if not isinstance(region,dict) or not isinstance(region.get('id'),(str,int)) or isinstance(region.get('id'),bool):
            raise ValueError('every region requires a string/integer id')
        key=str(region['id'])
        if not key or key in ids:
            raise ValueError('region ids must be nonempty and unique')
        ids.add(key)
        if region.get('type') not in TYPES:
            raise ValueError('region type must be text/formula/other/unknown')
        box=region.get('bbox')
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,int) and not isinstance(v,bool) for v in box):
            raise ValueError('bbox must be four integers [x1,y1,x2,y2]')
        x1,y1,x2,y2=box
        if not 0<=x1<x2<=width or not 0<=y1<y2<=height:
            raise ValueError(f'out-of-bounds/empty bbox for {key}: {box}')
        for field in ['content','latex','error']:
            if region.get(field) is not None and not isinstance(region[field],str):
                raise ValueError(f'{field} must be a string or null')
        if 'status' in region and region['status'] not in STATUSES:
            raise ValueError('invalid region status')
        if region.get('confidence') is not None and (not finite(region['confidence']) or not 0<=region['confidence']<=1):
            raise ValueError('confidence must be finite [0,1] or null')
    return regions


def validate_document(document):
    if document.get('version')!='2.0':
        raise ValueError('expected document version 2.0')
    width,height=document.get('width'),document.get('height')
    if not all(isinstance(v,int) and not isinstance(v,bool) and v>0 for v in [width,height]):
        raise ValueError('width/height must be positive integers')
    if not isinstance(document.get('source_image'),str) or not document['source_image']:
        raise ValueError('source_image is required')
    t=document.get('timestamp')
    if t is not None and (not finite(t) or t<0):
        raise ValueError('timestamp must be finite nonnegative seconds or null')
    validate_regions(document.get('regions'),width,height)
    return document


def save_document(document,path):
    validate_document(document)
    Path(path).write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_document(path):
    path=Path(path)
    document=validate_document(json.loads(path.read_text(encoding='utf-8')))
    source=Path(document['source_image'])
    if not source.is_absolute(): document['source_image']=str((path.parent/source).resolve())
    for region in document['regions']:
        if region.get('crop_image') and not Path(region['crop_image']).is_absolute():
            region['crop_image']=str((path.parent/region['crop_image']).resolve())
    return document
