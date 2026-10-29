# V2.51 진단 근거

실제 새 진단 추론372건 + V2.5 재사용144건 =516건이다. 별도 선택 옵션 통합 검증2건도 포함한다.
129개 이미지는 파생 조건을 포함하며 독립된129개 수식 표본이 아니다. 학습/신규 모델은 없다.

- `manifest.json` / `manifest.sha256`: 추론 전에 고정한 입력/조건/정답/이미지 해시. 입력129개는 `inputs/`에 있다.
- `predictions/`: 네 모델의 실제 원문, 시간/RSS/버전/가중치 provenance. 새 추론과 기존 예측 재사용을 구분한다.
- `evaluation.json` / `evaluation.csv`: 전체516개 Strict EM/edit distance, 제한적 구조 관측, 렌더링, raster 유사도.
- `comparison.pdf`: 실제 V34 판서와 인쇄 대응본18개를 나란히 비교한다.
- `controls.pdf`: 인쇄 구조 통제식24개. `all-conditions.pdf`: 전체129개 입력의 모델별 결과.
- `reference_renders/` / `rendered/`: 정답 및 모델 예측의 실제 TeX 렌더링. 실패/미지원은 평가 JSON에 남는다.
- `manual-review.json` / `review_sheets/`:23개 입력×4모델,92건의 직접 시각 검토. 코딩 에이전트의 목적 선택 사례이며 독립 인간 판정이나 전체 의미 정확도가 아니다.
- `internal-preprocessing.json` / `internal_previews/`: 설치된 SDK의516개 내부 tensor shape/hash와 일부 decode 이미지. 신경망을 새로 실행한 결과와 구분한다.
- `pixel-audit.json` / `pixel_audit/`: 느낌표 점/획의 원픽셀 및 경계/주변 영역 감사. RGB threshold는 GT mask가 아니다.
- `split-geometry.json`: 기존 V2.5 분할기를24개 인쇄식에 실제 적용한 결과. oracle hint를 명시한다.
- `optional-context.json` / `integration/`: 기본값0의 확장 후보 옵션을 실제 UniMERNet으로 실행한2개 예측. 원래 예측을 유지하며 자동 선택하지 않는다.
- `tables.md`, `weights-verification.json`, `validation.json`: 정량 부록, 기존 공식 가중치12파일 SHA 확인, 최종 근거 묶음 검증.

렌더링 성공이나 matrix cell 일치는 전체 수식의 정답 판정이 아니다. 특히 여러 array로 분해된
표현은 제한적 parser가 비교하지 못하고, 올바른 cell 뒤의 불필요한 기호도 cell 지표가 놓칠 수 있다.
시각 검토와 전체 원문을 함께 확인한다. PDF에서 긴 모델 출력은 일부만 표시하고 전체 문자열은 JSON/CSV에 보존한다.

이미지/렌더링 경로는 주로 저장소 루트 기준이다. 모델 디렉터리/SDK 소스/일부 로그의 절대경로는
당시 실행 provenance이며 다른 환경에서 실행할 때 로컬 모델 경로를 다시 준비한다.
원시 IPC/worker 로그와 추가 통합 PDF는 `benchmark-output/v251`에만 있다.

실제 판서 원출처는 bhargavaurala/accessmath-icfhr2018의 공개 AccessMath 키프레임이다.
기존 자료의 출처/해시는 V2/V2.5 provenance와 `experiments/v25/additional_regions.json`에 보존되어 있다.
이 묶음은 그 판서의 crop/파생 이미지와 자체 TeX 통제식을 포함한다. 저장소 코드의 MIT 라이선스가
외부 연구 데이터의 권리/조건을 대체하지 않는다. 모델 가중치는 포함하지 않는다.
