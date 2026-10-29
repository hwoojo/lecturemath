"""Execute V2.5 geometry on printed structures; oracle hints labeled separately."""
import json
from pathlib import Path
import cv2
from boardnotes.v25.splitting import propose_rows


def main():
    root=Path('docs/assets/v251');manifest=json.loads((root/'manifest.json').read_text());rows=[]
    for case in manifest['cases']:
        if case['group']!='control':continue
        image=cv2.imread(case['input']);h,w=image.shape[:2]
        family=case['family'];hint={'fraction':'fraction','matrix':'matrix','vector':'matrix','scripts':'script','multiline':'aligned'}.get(family)
        rows.append({'id':case['id'],'without_hint':propose_rows(image,[0,0,w,h]),
                     'with_oracle_structure_hint':propose_rows(image,[0,0,w,h],hint),
                     'notice':'Hints are known diagnostic labels, not auto-detected mathematical structure.'})
    (root/'split-geometry.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Checked actual V2.5 splitter on',len(rows),'printed controls')


if __name__=='__main__':main()
