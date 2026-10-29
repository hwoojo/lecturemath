"""Create B0 for frozen new frames in short-lived V34-only processes."""
import argparse
import json
from pathlib import Path
from boardnotes.v25.pipeline import obtain_b0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,default=Path('experiments/v25/additional_regions.json'))
    p.add_argument('--output',type=Path,default=Path('benchmark-output/v25/b0'))
    p.add_argument('--v1-python',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report=[]
    for sample in json.loads(args.annotations.read_text())['samples']:
        name=sample['id'];dest=args.output/name
        if (dest/'b0.png').exists():
            print('Reusing',name,flush=True);continue
        dest.mkdir(parents=True,exist_ok=True)
        mask,metadata=obtain_b0(sample['source_path'],None,dest,args.v1_python)
        report.append({'id':name,'path':str(mask),'report':metadata})
        print('Prepared',name,flush=True)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
