# V2.51 정량 부록

각 조건을 별도로 보고한다. 최적 조건 조합 점수나 의미 정확도는 산출하지 않는다.

| 모델 | 실제 RGB EM | 실제 V34 EM | 인쇄 대응본 EM | 인쇄 통제 EM | 통제 matrix cell 일치 |
|---|---:|---:|---:|---:|---:|
| PP-FormulaNet-S | 4/18 | 5/18 | 4/18 | 8/24 | 6/7 |
| PP-FormulaNet_plus-S | 5/18 | 2/18 | 6/18 | 9/24 | 7/7 |
| PP-FormulaNet-L | 5/18 | 4/18 | 4/18 | 8/24 | 7/7 |
| UniMERNet | 5/18 | 5/18 | 5/18 | 7/24 | 1/7 |

Matrix cell 일치는 행·열·원소 문자열의 제한적 관측이며 괄호/분할선/전체 수학 의미의 정답 판정이 아니다.

## Crop 민감도

| 모델 | 입력 | 기존5 | pad10/5 | pad20/5 | tight/5 | scale2/2 |
|---|---|---:|---:|---:|---:|---:|
| PP-FormulaNet-S | rgb | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 |
| PP-FormulaNet-S | b0 | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 |
| PP-FormulaNet_plus-S | rgb | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 |
| PP-FormulaNet_plus-S | b0 | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 |
| PP-FormulaNet-L | rgb | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 |
| PP-FormulaNet-L | b0 | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 |
| UniMERNet | rgb | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 |
| UniMERNet | b0 | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 |

## 통제식별 관측

| 식 | 모델 | Strict | matrix cells | factorial count | script tokens | multiline rows | render |
|---|---|---|---|---|---|---|---|
| control__eq | PP-FormulaNet-S | True | — | — | — | — | rendered |
| control__factorial | PP-FormulaNet-S | True | — | True | — | — | rendered |
| control__indexed_factorial | PP-FormulaNet-S | False | — | True | True | — | rendered |
| control__general_factorial | PP-FormulaNet-S | False | — | True | True | — | rendered |
| control__frac | PP-FormulaNet-S | True | — | — | — | — | rendered |
| control__frac_sum | PP-FormulaNet-S | True | — | — | — | — | rendered |
| control__nested_frac | PP-FormulaNet-S | True | — | — | — | — | rendered |
| control__long_fraction | PP-FormulaNet-S | False | — | — | — | — | rendered |
| control__vec2 | PP-FormulaNet-S | False | True | — | — | — | rendered |
| control__vec3 | PP-FormulaNet-S | False | True | — | — | — | rendered |
| control__vec_symbolic | PP-FormulaNet-S | False | True | — | — | — | rendered |
| control__matrix22 | PP-FormulaNet-S | False | True | — | — | — | rendered |
| control__matrix23 | PP-FormulaNet-S | False | True | — | — | — | rendered |
| control__matrix33 | PP-FormulaNet-S | False | True | — | — | — | rendered |
| control__augmented | PP-FormulaNet-S | False | False | — | — | — | unsupported |
| control__line1 | PP-FormulaNet-S | True | — | — | — | — | rendered |
| control__lines2 | PP-FormulaNet-S | False | — | — | — | True | rendered |
| control__lines3 | PP-FormulaNet-S | False | — | — | — | True | rendered |
| control__sup | PP-FormulaNet-S | False | — | — | True | — | rendered |
| control__sub | PP-FormulaNet-S | False | — | — | True | — | rendered |
| control__sub_sup | PP-FormulaNet-S | False | — | — | True | — | rendered |
| control__complex_scripts | PP-FormulaNet-S | True | — | — | True | — | rendered |
| control__set | PP-FormulaNet-S | True | — | — | — | — | rendered |
| control__set_text | PP-FormulaNet-S | False | — | — | True | — | rendered |
| control__eq | PP-FormulaNet_plus-S | True | — | — | — | — | rendered |
| control__factorial | PP-FormulaNet_plus-S | True | — | True | — | — | rendered |
| control__indexed_factorial | PP-FormulaNet_plus-S | False | — | True | True | — | rendered |
| control__general_factorial | PP-FormulaNet_plus-S | False | — | True | True | — | rendered |
| control__frac | PP-FormulaNet_plus-S | True | — | — | — | — | rendered |
| control__frac_sum | PP-FormulaNet_plus-S | True | — | — | — | — | rendered |
| control__nested_frac | PP-FormulaNet_plus-S | True | — | — | — | — | rendered |
| control__long_fraction | PP-FormulaNet_plus-S | True | — | — | — | — | rendered |
| control__vec2 | PP-FormulaNet_plus-S | False | True | — | — | — | rendered |
| control__vec3 | PP-FormulaNet_plus-S | False | True | — | — | — | rendered |
| control__vec_symbolic | PP-FormulaNet_plus-S | False | True | — | — | — | rendered |
| control__matrix22 | PP-FormulaNet_plus-S | False | True | — | — | — | rendered |
| control__matrix23 | PP-FormulaNet_plus-S | False | True | — | — | — | rendered |
| control__matrix33 | PP-FormulaNet_plus-S | False | True | — | — | — | rendered |
| control__augmented | PP-FormulaNet_plus-S | False | True | — | — | — | rendered |
| control__line1 | PP-FormulaNet_plus-S | True | — | — | — | — | rendered |
| control__lines2 | PP-FormulaNet_plus-S | False | — | — | — | False | rendered |
| control__lines3 | PP-FormulaNet_plus-S | False | — | — | — | False | rendered |
| control__sup | PP-FormulaNet_plus-S | False | — | — | True | — | rendered |
| control__sub | PP-FormulaNet_plus-S | False | — | — | True | — | rendered |
| control__sub_sup | PP-FormulaNet_plus-S | False | — | — | True | — | rendered |
| control__complex_scripts | PP-FormulaNet_plus-S | True | — | — | True | — | rendered |
| control__set | PP-FormulaNet_plus-S | True | — | — | — | — | rendered |
| control__set_text | PP-FormulaNet_plus-S | False | — | — | True | — | rendered |
| control__eq | PP-FormulaNet-L | True | — | — | — | — | rendered |
| control__factorial | PP-FormulaNet-L | True | — | True | — | — | rendered |
| control__indexed_factorial | PP-FormulaNet-L | False | — | True | True | — | rendered |
| control__general_factorial | PP-FormulaNet-L | False | — | True | True | — | rendered |
| control__frac | PP-FormulaNet-L | True | — | — | — | — | rendered |
| control__frac_sum | PP-FormulaNet-L | True | — | — | — | — | rendered |
| control__nested_frac | PP-FormulaNet-L | True | — | — | — | — | rendered |
| control__long_fraction | PP-FormulaNet-L | False | — | — | — | — | rendered |
| control__vec2 | PP-FormulaNet-L | False | True | — | — | — | rendered |
| control__vec3 | PP-FormulaNet-L | False | True | — | — | — | rendered |
| control__vec_symbolic | PP-FormulaNet-L | False | True | — | — | — | rendered |
| control__matrix22 | PP-FormulaNet-L | False | True | — | — | — | rendered |
| control__matrix23 | PP-FormulaNet-L | False | True | — | — | — | rendered |
| control__matrix33 | PP-FormulaNet-L | False | True | — | — | — | rendered |
| control__augmented | PP-FormulaNet-L | False | True | — | — | — | rendered |
| control__line1 | PP-FormulaNet-L | True | — | — | — | — | rendered |
| control__lines2 | PP-FormulaNet-L | False | — | — | — | True | rendered |
| control__lines3 | PP-FormulaNet-L | False | — | — | — | True | rendered |
| control__sup | PP-FormulaNet-L | False | — | — | True | — | rendered |
| control__sub | PP-FormulaNet-L | False | — | — | True | — | rendered |
| control__sub_sup | PP-FormulaNet-L | False | — | — | True | — | rendered |
| control__complex_scripts | PP-FormulaNet-L | True | — | — | True | — | rendered |
| control__set | PP-FormulaNet-L | True | — | — | — | — | rendered |
| control__set_text | PP-FormulaNet-L | False | — | — | True | — | rendered |
| control__eq | UniMERNet | True | — | — | — | — | rendered |
| control__factorial | UniMERNet | True | — | True | — | — | rendered |
| control__indexed_factorial | UniMERNet | False | — | True | True | — | rendered |
| control__general_factorial | UniMERNet | False | — | True | True | — | rendered |
| control__frac | UniMERNet | True | — | — | — | — | rendered |
| control__frac_sum | UniMERNet | True | — | — | — | — | rendered |
| control__nested_frac | UniMERNet | False | — | — | — | — | rendered |
| control__long_fraction | UniMERNet | False | — | — | — | — | rendered |
| control__vec2 | UniMERNet | False | False | — | — | — | rendered |
| control__vec3 | UniMERNet | False | False | — | — | — | rendered |
| control__vec_symbolic | UniMERNet | False | False | — | — | — | rendered |
| control__matrix22 | UniMERNet | False | False | — | — | — | rendered |
| control__matrix23 | UniMERNet | False | False | — | — | — | rendered |
| control__matrix33 | UniMERNet | False | True | — | — | — | rendered |
| control__augmented | UniMERNet | False | False | — | — | — | rendered |
| control__line1 | UniMERNet | True | — | — | — | — | rendered |
| control__lines2 | UniMERNet | False | — | — | — | True | rendered |
| control__lines3 | UniMERNet | False | — | — | — | True | rendered |
| control__sup | UniMERNet | False | — | — | True | — | rendered |
| control__sub | UniMERNet | False | — | — | True | — | rendered |
| control__sub_sup | UniMERNet | False | — | — | True | — | rendered |
| control__complex_scripts | UniMERNet | True | — | — | True | — | rendered |
| control__set | UniMERNet | True | — | — | — | — | rendered |
| control__set_text | UniMERNet | False | — | — | True | — | rendered |

## 새 추론의 자원 사용

| 모델 | 새 crop 수 | 프로세스 초 | 로딩 초 | peak RSS MiB |
|---|---:|---:|---:|---:|
| PP-FormulaNet-S | 93 | 90.76 | 12.75 | 799.8 |
| PP-FormulaNet_plus-S | 93 | 65.49 | 7.63 | 823.6 |
| PP-FormulaNet-L | 93 | 1216.44 | 9.08 | 1631.6 |
| UniMERNet | 93 | 676.55 | 12.49 | 2286.3 |
