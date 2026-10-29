"""Aspect-preserving visual sheets for a declared diagnostic review subset."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

REVIEW_IDS=[
 'control__matrix33','control__indexed_factorial','control__lines2','control__complex_scripts',
 'control__set_text','control__augmented',
 'real__lecture_08_53016__b0__cardinality_factorial','matched__lecture_08_53016__b0__cardinality_factorial',
 'real__lecture_07_67415__b0__symbolic_vector','matched__lecture_07_67415__b0__symbolic_vector',
 'real__lecture_08_53016__b0__permutation_matrix','matched__lecture_08_53016__b0__permutation_matrix',
 'real__lecture_08_53016__b0__input_vector',
 'real__lecture_07_67415__b0__noncommuting_product',
 'real__lecture_06_47799__b0__real_space',
 'real__lecture_01_10423__b0__formula_system',
 'control__nested_frac','control__long_fraction',
 'control__vec3','control__matrix22','control__matrix23',
 'sensitivity__lecture_08_53016__rgb__permutation_matrix__pad10',
 'sensitivity__lecture_08_53016__b0__permutation_matrix__pad10',
]


def main():
    root=Path('docs/assets/v251');data=json.loads((root/'evaluation.json').read_text())
    models=list(dict.fromkeys(r['model'] for r in data['predictions']));indexed={(r['model'],r['id']):r for r in data['predictions']}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    for start in range(0,len(REVIEW_IDS),3):
        sheet=Image.new('RGB',(1250,780),'white');draw=ImageDraw.Draw(sheet)
        for i,key in enumerate(REVIEW_IDS[start:start+3]):
            row=indexed[models[0],key];y=i*260;draw.text((5,y+2),key,fill='black',font=font)
            paths=[('INPUT',row['input']),('REFERENCE',row['reference_render'].get('path'))]+[(m,indexed[m,key]['prediction_render'].get('path')) for m in models]
            # Input/reference share the leftmost cell, model cells follow.
            for j,(label,path) in enumerate(paths):
                col=0 if j<2 else j-1;top=y+25+(105 if j==1 else 0)
                draw.text((col*250+5,top),label,fill='black',font=font)
                if path:
                    with Image.open(path) as image:
                        image=image.convert('RGB');image.thumbnail((238,80 if j<2 else 170))
                        sheet.paste(image,(col*250+(250-image.width)//2,top+22))
                else:draw.text((col*250+5,top+30),'RENDER FAILED / UNSUPPORTED',fill='red',font=font)
        target=root/'review_sheets'/f'sheet_{start//3+1}.png';target.parent.mkdir(exist_ok=True);sheet.save(target)
    print('Wrote', (len(REVIEW_IDS)+2)//3,'review sheets for',len(models),'models')


if __name__=='__main__':main()
