"""Explicit optional input geometry; no OCR, ground truth, or learned parameters."""
import cv2
import numpy as np
from boardnotes.v25.context import context_bbox


def ink_box(image,threshold=200):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    points=cv2.findNonZero((gray<threshold).astype(np.uint8))
    if points is None:return [0,0,image.shape[1],image.shape[0]]
    x,y,w,h=cv2.boundingRect(points)
    return [x,y,x+w,y+h]


def crop_box(box,width,height,margin=0.):
    return context_bbox(box,width,height,margin)


def condition_crop(image,box,condition):
    """Return pixels and source coordinates. margin is a fraction on each side."""
    h,w=image.shape[:2]
    if condition in {'pad10','pad20'}:box=crop_box(box,w,h,.1 if condition=='pad10' else .2)
    x1,y1,x2,y2=box;crop=image[y1:y2,x1:x2].copy()
    if condition=='tight':
        a,b,c,d=ink_box(crop);box=[x1+a,y1+b,x1+c,y1+d];crop=image[box[1]:box[3],box[0]:box[2]].copy()
    elif condition=='scale2':crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    elif condition not in {'base','pad10','pad20'}:raise ValueError(condition)
    return crop,box


def printed_canvas(rendered,canvas_size,target_box=None):
    """Fit tight rendered ink to a target bounding rectangle; preserve aspect ratio."""
    w,h=canvas_size
    box=target_box or [8,8,w-8,h-8]
    x1,y1,x2,y2=box
    a,b,c,d=ink_box(rendered);ink=rendered[b:d,a:c]
    scale=min(max(1,x2-x1)/ink.shape[1],max(1,y2-y1)/ink.shape[0])
    rw=max(1,round(ink.shape[1]*scale));rh=max(1,round(ink.shape[0]*scale))
    ink=cv2.resize(ink,(rw,rh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    canvas=np.full((h,w,3),255,np.uint8)
    left=x1+(x2-x1-rw)//2;top=y1+(y2-y1-rh)//2
    canvas[top:top+rh,left:left+rw]=ink
    return canvas
