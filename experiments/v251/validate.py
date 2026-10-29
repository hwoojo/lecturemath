"""Verify the final evidence bundle, including completion and portable image paths."""
import hashlib
import json
from pathlib import Path
import re
from pypdf import PdfReader


def main():
    root=Path('docs/assets/v251');manifest=json.loads((root/'manifest.json').read_text())
    digest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    assert digest==(root/'manifest.sha256').read_text().split()[0]
    cases=manifest['cases'];assert len(cases)==129
    ids={c['id'] for c in cases};assert len(ids)==129
    for c in cases:assert hashlib.sha256(Path(c['input']).read_bytes()).hexdigest()==c['sha256'],c['id']
    for model in manifest['models']:
        run=json.loads((root/'predictions'/(model+'.json')).read_text())
        assert run['manifest_sha256']==digest
        assert {r['id'] for r in run['predictions']}==ids and len(run['predictions'])==129
        request=json.loads((Path('benchmark-output/v251')/model/'formula_request.json').read_text())
        assert len(request['jobs'])==93 and all(set(j)=={'id','image'} for j in request['jobs'])
    result=json.loads((root/'evaluation.json').read_text());assert result['complete'] is True
    assert len(result['predictions'])==516 and result['manifest_sha256']==digest
    for r in result['predictions']:
        for key in ['reference_render','prediction_render']:
            render=r[key]
            if render['status']=='rendered':
                assert Path(render['path']).is_file()
                assert hashlib.sha256(Path(render['path']).read_bytes()).hexdigest()==render['png_sha256']
    pages={name:len(PdfReader(root/name).pages) for name in ['comparison.pdf','controls.pdf','all-conditions.pdf']}
    assert pages=={'comparison.pdf':18,'controls.pdf':24,'all-conditions.pdf':129}
    reviewed=json.loads((root/'manual-review.json').read_text())
    assert len(reviewed['records'])==92
    assert len({(r['model'],r['id']) for r in reviewed['records']})==92
    for r in reviewed['records']:assert r['observations'] and Path(r['evidence_sheet']).is_file()
    for name in ['docs/v2_51.md','docs/v2_51_results.md']:
        path=Path(name)
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in link and not link.startswith('#'):assert (path.parent/link.split('#')[0]).exists(),link
    tests=Path('benchmark-output/v251/tests.log').read_text();assert '138 passed' in tests
    integration=json.loads((root/'optional-context.json').read_text())
    assert len(integration['regions'])==2 and integration['formula_context_margin']==.1
    assert all(r['prediction_matches_diagnostic_run'] for r in integration['regions'])
    report={'manifest_sha256':digest,'inputs_verified':129,'fresh_predictions':372,'reused_v25_predictions':144,
            'evaluation_records':516,'additional_integration_predictions':2,'review_records':92,'pdf_pages':pages,'existing_and_new_tests_passed':138,
            'scope':'Artifact integrity and software regression, separate from recognition quality.'}
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
