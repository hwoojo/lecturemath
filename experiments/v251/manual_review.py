"""Serialize explicit, finite-case visual decisions after inspecting review sheets.

These decisions are authored for the recorded predictions, not a classifier or
an equivalence normalization applied to other expressions.
"""
from collections import Counter
import json
from pathlib import Path
from .review_sheets import REVIEW_IDS

S='PP-FormulaNet-S';P='PP-FormulaNet_plus-S';L='PP-FormulaNet-L';U='UniMERNet'
MODELS=[S,P,L,U]


def decisions():
    out={}
    def all_same(key,note):out[key]={m:('visual_match',note) for m in MODELS}
    def set_case(key,entries):out[key]=entries
    all_same('control__matrix33','All nine values and the 3x3 square-bracket layout are visually preserved; environment names differ.')
    all_same('control__indexed_factorial','Absolute bars, S subscript 3, factorial !, and final 6 are preserved. Braces around subscript 3 explain the strict mismatch.')
    all_same('control__lines2','Both equations and their two-row order are preserved; array replaces aligned.')
    out['control__lines2'][P]=('content_error','First equation loses coefficient 3: x=1 instead of 3x=1. Added punctuation also changes appearance.')
    all_same('control__complex_scripts','x carries subscript i+1 and superscript n-1; a carries subscript ij and superscript (k). Both attachment positions and values were visually checked.')
    all_same('control__set_text','Set identifier, matrix dimensions and the English words is a permutation matrix are present; text/mathrm font and word spacing differ.')
    out['control__set_text'][P]=('content_error','Literal letters from malformed mathrm/text output appear in the English phrase; this is not merely a text-vs-mathrm wrapper difference.')
    set_case('control__augmented',{
        S:('latex_syntax_error','Digits are largely retained but braces/environment syntax is malformed and cannot render. Do not infer semantic correctness from digit presence.'),
        P:('layout_difference','All six values and their rows/columns are present, but one vertical partition rule becomes two. Visual layout differs; no automatic claim that the mathematics is different.'),
        L:('visual_match','Two rows, three columns and the single partition rule after column 2 are preserved.'),
        U:('presentation_difference_parser_limit','Two adjacent array blocks joined by middle| render as the reference augmented matrix. The limited flat-matrix parser cannot combine this representation; its false cell-match flag is not a demonstrated recognition error.')})
    set_case('real__lecture_08_53016__b0__cardinality_factorial',{
        S:('content_and_syntax_error','Factorial ! is not preserved; malformed vert/subscript output also blocks rendering. The source dot is visibly present.'),
        P:('content_error','3! becomes 31 and final 6 becomes G. Source dot survives.'),
        L:('content_error','3! becomes 31 although the source dot survives; final 6 is preserved.'),
        U:('content_error','S becomes lowercase s and ! becomes a vertical delimiter; the factorial dot survives in the source.')})
    all_same('matched__lecture_08_53016__b0__cardinality_factorial','The matched printed counterpart preserves S_3, bars, ! and 6; the handwritten counterpart fails. This is an appearance contrast, not proof of a particular training-data cause.')
    set_case('real__lecture_07_67415__b0__symbolic_vector',{
        S:('structure_error','Three-row p/q/r column vector becomes an integral-like expression; rows and symbols are lost.'),
        P:('structure_error','Integral/fraction-like expression with wrong letters replaces the three-element vector.'),
        L:('structure_error','Column vector becomes p/q fraction and loses r.'),
        U:('structure_error','Column vector becomes p/r fraction and loses q.')})
    all_same('matched__lecture_07_67415__b0__symbolic_vector','All p,q,r rows and square brackets are preserved in the printed counterpart.')
    out['matched__lecture_07_67415__b0__symbolic_vector'][U]=('structure_error','Even the printed counterpart becomes binom(p,q), drops r and changes brackets. This failure is not specific to handwriting.')
    set_case('real__lecture_08_53016__b0__permutation_matrix',{
        S:('content_and_syntax_error','Malformed matrix tokens; rows/columns and entries cannot be recovered from the output as the reference 3x3 matrix.'),
        P:('structure_error','Output is 3x2, omitting a column and changing the arrangement of entries.'),
        L:('structure_error','Rows are flattened into tokens 001/1000/010 rather than three cells per row, with an extra zero.'),
        U:('structure_and_syntax_error','First row has two entries while subsequent rows have three; array declares two columns. Ragged structure also fails rendering.')})
    all_same('matched__lecture_08_53016__b0__permutation_matrix','All nine 0/1 values and the 3x3 square-bracket layout are preserved in the matched print.')
    all_same('real__lecture_08_53016__b0__input_vector','All x,y,z rows and tall square brackets are preserved; the matrix/array environment differs.')
    out['real__lecture_08_53016__b0__input_vector'][S]=('structure_and_syntax_error','Missing/misplaced x,y,z structure and unsupported accent output; not just a matrix environment difference.')
    out['real__lecture_08_53016__b0__input_vector'][U]=('layout_difference','x,y,z and three rows are present, but plain [ ] are short and attach around the middle y row rather than enclosing the whole vector. Cell agreement alone misses this layout issue.')
    all_same('real__lecture_07_67415__b0__noncommuting_product','AB neq BA and punctuation preserve the reference relation; ne/neq and spacing are presentation differences.')
    out['real__lecture_07_67415__b0__noncommuting_product'][S]=('content_error','B becomes beta in the right-hand product: AB neq beta A.')
    set_case('real__lecture_06_47799__b0__real_space',{
        S:('unresolved_typography','R^2 versus reference mathbb R^2. Double-stroke typography differs; do not automatically infer semantic equivalence or count as a proven content error.'),
        P:('content_error','P^z differs from both the visible R and superscript 2.'),
        L:('unresolved_typography','R^2 versus mathbb R^2; symbol-style interpretation requires contextual adjudication.'),
        U:('unresolved_typography','R^2 versus mathbb R^2; symbol-style interpretation requires contextual adjudication.')})
    set_case('real__lecture_01_10423__b0__formula_system',{
        S:('structure_and_syntax_error','Whole output is malformed and omits the two-equation layout; separately recognized rows are preserved in the assembly evidence.'),
        P:('content_and_structure_error','Only 3x=3 is produced; RHS is wrong and second equation is missing.'),
        L:('layout_difference','Both equations are present, but they are separate inline brace groups rather than the original two-row system with a right brace.'),
        U:('structure_error','Only the second equation x+2y=3 remains; the first row is absent.')})
    all_same('control__nested_frac','Both fraction levels, denominator 1+1/x and numerator 1 are visually preserved; redundant grouping braces may differ.')
    all_same('control__long_fraction','Both fractions, inner 1/3, equals signs and arithmetic terms are preserved; redundant outer groups differ.')
    all_same('control__vec3','All three 1/2/3 rows and square brackets are preserved.')
    out['control__vec3'][U]=('structure_error','Printed three-element vector becomes binom(1,2), dropping 3 and changing brackets; counterexample to handwriting-only failure.')
    for key,values in [('control__matrix22','1,2 / 3,4'),('control__matrix23','1,2,3 / 4,5,6')]:
        all_same(key,'All row/column positions and values '+values+' are visually preserved with square brackets.')
        out[key][U]=('layout_error','All numeric values '+values+' remain, but the first row closes its own bracket and the second row lacks a matching left bracket. Row-array output distorts the matrix delimiter structure.')
    set_case('sensitivity__lecture_08_53016__rgb__permutation_matrix__pad10',{
        S:('structure_error','Matrix is flattened and most values/rows are lost.'),
        P:('structure_error','Matrix is flattened into a short numeric token, losing its rows/columns.'),
        L:('structure_error','The output does not retain the full reference 3x3 matrix; inspect the paired literal for lost cells.'),
        U:('partial_recovery_with_syntax_error','All nine cells and their row/column order recover, but array declares two columns while rows have three. Rendering fails. Cell-match success is not full-formula recovery.')})
    set_case('sensitivity__lecture_08_53016__b0__permutation_matrix__pad10',{
        S:('structure_and_syntax_error','Malformed and incorrect matrix rows/cells remain after padding.'),
        P:('structure_and_syntax_error','Padding yields extra/malformed rows; full 3x3 structure is not recovered.'),
        L:('structure_error','Output is 3x2 rather than 3x3; one column is still missing.'),
        U:('partial_recovery_with_extra_symbol','All nine matrix cells recover and render, but an extra 1 is appended outside the closing bracket. This is partial cell recovery, not a correct complete prediction.')})
    return out


def main():
    root=Path('docs/assets/v251');data=json.loads((root/'evaluation.json').read_text());assert data['complete']
    indexed={(r['model'],r['id']):r for r in data['predictions']};notes=decisions();records=[]
    for i,key in enumerate(REVIEW_IDS):
        for m in MODELS:
            judgement,note=notes[key][m];row=indexed[m,key]
            if judgement=='visual_match':judgement='literal_and_visual_match' if row['strict_exact_match'] else 'presentation_difference'
            records.append({'id':key,'model':m,'judgement':judgement,'observations':note,
                'strict_exact_match':row['strict_exact_match'],'reference':row['reference'],'prediction':row['latex'],
                'rendering_status':row['prediction_render']['status'],'evidence_sheet':str(root/'review_sheets'/f'sheet_{i//3+1}.png'),
                'scope':'Finite visual structure/symbol inspection; not a general mathematical equivalence verdict.'})
    (root/'manual-review.json').write_text(json.dumps({'reviewer':'Coding assistant: direct visual inspection of eight sheets, plus literal prediction inspection.',
        'independent_human_review':False,'selection':'23 purpose-selected diagnostic inputs after viewing outputs; not an unbiased accuracy sample.',
        'protocol':'Judgements explicitly authored per recorded case/model. No automatic semantic classifier or GT-based runtime selection.',
        'counts':dict(Counter(r['judgement'] for r in records)),'records':records},indent=2)+'\n')
    print('Stored',len(records),'finite-case visual decisions')


if __name__=='__main__':main()
