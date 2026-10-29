"""Generate descriptive, condition-separated tables from final measured outputs."""
from collections import defaultdict
import json
from pathlib import Path


def main():
    root=Path('docs/assets/v251');data=json.loads((root/'evaluation.json').read_text())
    if not data['complete']:raise ValueError('final tables require all four models')
    rows=data['predictions'];models=list(dict.fromkeys(r['model'] for r in rows))
    lines=['# V2.51 정량 부록','', '각 조건을 별도로 보고한다. 최적 조건 조합 점수나 의미 정확도는 산출하지 않는다.','',
           '| 모델 | 실제 RGB EM | 실제 V34 EM | 인쇄 대응본 EM | 인쇄 통제 EM | 통제 matrix cell 일치 |','|---|---:|---:|---:|---:|---:|']
    def ratio(selected,key='strict_exact_match'):
        return f"{sum(r[key] is True for r in selected)}/{len(selected)}"
    for m in models:
        r=[r for r in rows if r['model']==m]
        groups=[[x for x in r if x['group']=='real_baseline' and x.get('variant')==v] for v in ['rgb','b0']]
        groups += [[x for x in r if x['group']==g] for g in ['matched_print','control']]
        matrices=[x for x in r if x['group']=='control' and x['matrix_cells_match'] is not None]
        lines.append('| '+m+' | '+' | '.join([ratio(x) for x in groups]+[ratio(matrices,'matrix_cells_match')])+' |')
    lines+=['','Matrix cell 일치는 행·열·원소 문자열의 제한적 관측이며 괄호/분할선/전체 수학 의미의 정답 판정이 아니다.','',
            '## Crop 민감도','', '| 모델 | 입력 | 기존5 | pad10/5 | pad20/5 | tight/5 | scale2/2 |','|---|---|---:|---:|---:|---:|---:|']
    from .design import SENSITIVITY
    keys={a+'__'+b for a,b in SENSITIVITY}
    for m in models:
        for v in ['rgb','b0']:
            r=[r for r in rows if r['model']==m and r.get('variant')==v]
            base=[x for x in r if x['group']=='real_baseline' and x['case'] in keys]
            parts=[base]+[[x for x in r if x['group']=='sensitivity' and x['condition']==c] for c in ['pad10','pad20','tight','scale2']]
            lines.append('| '+m+' | '+v+' | '+' | '.join(ratio(x) for x in parts)+' |')
    lines += ['','## 통제식별 관측','', '| 식 | 모델 | Strict | matrix cells | factorial count | script tokens | multiline rows | render |','|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['group']=='control':
            values=[r['id'],r['model'],r['strict_exact_match'],r['matrix_cells_match'],r['factorial_count_match'],r['scripts_lexical_match'],r['multiline_rows_match'],r['prediction_render']['status']]
            lines.append('| '+' | '.join(str(v) if v is not None else '—' for v in values)+' |')
    lines += ['','## 새 추론의 자원 사용','', '| 모델 | 새 crop 수 | 프로세스 초 | 로딩 초 | peak RSS MiB |','|---|---:|---:|---:|---:|']
    for r in data['runs']:
        count=sum(x['model']==r['model'] and x['evidence']=='fresh_v251_inference' for x in rows)
        lines.append(f"| {r['model']} | {count} | {r['process_seconds']:.2f} | {r['model_load_seconds']:.2f} | {r['process_peak_rss_mib']:.1f} |")
    (root/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
