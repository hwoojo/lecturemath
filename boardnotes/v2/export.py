"""Explicit V1 -> V2 handoff without pretending a temporal composite is RGB."""
import json
from pathlib import Path
import cv2
import numpy as np


def export_keyframes(keyframes,metadata,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    pages=[]
    for index,(image,start,end) in enumerate(keyframes):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        if gray.dtype!=np.uint8 or not np.isin(gray,[0,255]).all():
            raise ValueError('V2 export requires unchanged binary ST3D keyframes')
        if not (np.isfinite(start) and np.isfinite(end) and 0<=start<=end):
            raise ValueError('invalid keyframe interval in seconds')
        image_path=destination/f'keyframe_{index:04}.png'
        if not cv2.imwrite(str(image_path),gray): raise OSError(image_path)
        h,w=gray.shape
        pages.append({'id':index,'image':str(image_path.resolve()),'variant':'b0','width':w,'height':h,
                      'timestamp':float(end),'interval_seconds':[float(start),float(end)],
                      'coordinate_system':'uncropped_st3d_composite_pixels','original_rgb':None,
                      'source_video_dimensions':[metadata['width'],metadata['height']],
                      'scale_to_video':[metadata['width']/w,metadata['height']/h]})
    result={'format':'boardnotes-v2-inputs-1','pages':pages,
            'notice':'Temporal composites; no single matching RGB frame is guaranteed. Images are before PDF crop/smoothing. Scale is a geometric mapping, not proof of RGB alignment.'}
    (destination/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
