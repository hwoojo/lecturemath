"""Execute installed PaddleX preprocessing only; no weights or inference."""
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault('PADDLE_PDX_CACHE_HOME',str(Path('benchmark-output/v251/sdk-cache').resolve()))
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
import cv2
import numpy as np
from paddlex.inference.models.formula_recognition.processors import (
    UniMERNetImgDecode,UniMERNetTestTransform,LatexImageFormat,UniMERNetImageFormat)


def main():
    root=Path('docs/assets/v251');data=json.loads((root/'manifest.json').read_text())
    sizes={'PP-FormulaNet-S':(384,384),'PP-FormulaNet_plus-S':(384,384),'PP-FormulaNet-L':(768,768),'UniMERNet':(192,672)}
    rows=[]
    for name,size in sizes.items():
        decode=UniMERNetImgDecode(size);normalize=UniMERNetTestTransform()
        fmt=UniMERNetImageFormat() if name=='UniMERNet' else LatexImageFormat()
        for case in data['cases']:
            image=cv2.cvtColor(cv2.imread(case['input']),cv2.COLOR_BGR2RGB)
            decoded=decode([image])[0];normalized=normalize([decoded])[0];tensor=fmt([normalized])[0]
            row={'model':name,'id':case['id'],'decode_shape':list(decoded.shape),'tensor_shape':list(tensor.shape),
                'tensor_sha256':hashlib.sha256(tensor.tobytes()).hexdigest(),'tensor_min':float(tensor.min()),'tensor_max':float(tensor.max())}
            if (case.get('family') in {'factorial','matrix'} and case['group'] in {'control','print_format'}) or any(key in case['id'] for key in ['cardinality_factorial','symbolic_vector']):
                out=root/'internal_previews'/name/(case['id']+'.png');out.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(out),cv2.cvtColor(decoded,cv2.COLOR_RGB2BGR));row['decoded_preview']=str(out)
            rows.append(row)
    import paddlex.inference.models.formula_recognition.processors as processors
    source=Path(processors.__file__)
    (root/'internal-preprocessing.json').write_text(json.dumps({'implementation':str(source),'implementation_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'notice':'Actual installed SDK preprocessing; no neural models loaded. Preview is decoded RGB before normalization. Same tensor hash establishes identical model input, not accuracy.',
        'sizes':sizes,'records':rows},indent=2)+'\n')
    print('Measured',len(rows),'internal tensors',flush=True)


if __name__=='__main__':main()
