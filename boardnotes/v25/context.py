"""Explicit context-crop geometry; never chooses a prediction or changes defaults."""
import math


def context_bbox(box,width,height,margin):
    if not 0<=margin<=.5:raise ValueError('formula context margin must be between 0 and 0.5 per side')
    x1,y1,x2,y2=box
    dx=math.ceil((x2-x1)*margin);dy=math.ceil((y2-y1)*margin)
    return [max(0,x1-dx),max(0,y1-dy),min(width,x2+dx),min(height,y2+dy)]
