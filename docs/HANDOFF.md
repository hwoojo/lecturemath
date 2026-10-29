# 후임 작업자 인수인계

기준일: 2026-10-28 (Asia/Seoul)

## V2.51 실패 원인 진단 (2026-10-29)

[설계](v2_51.md)와 [결과](v2_51_results.md)를 먼저 읽는다. V2.5 데이터와 기본 동작을 보존하고
`experiments/v251`에서 기존4모델의 인쇄/판서·구조·crop·분할 조건을 비교했다.
새 모델/학습은 없다. 입력129개와 SHA를 고정하고 기존36개 baseline/모델과 새93개 추론/모델을 구분한다.
Strict EM과 행렬 cell 일치/렌더링/선택 사례 시각 판정을 같은 정확도로 취급하지 않는다.
추가 테스트를 포함한138개 통과. 공유 근거는 `docs/assets/v251`, 원시 IPC/로그는 ignored `benchmark-output/v251`에 둔다.
`--formula-context-margin`은 기본0의 검토용 추가 후보 옵션이다. 원래 예측을 보존하며 자동 선택하지 않는다.

## V2.5 후속 구현 (2026-10-29)

현재 후속 작업은 [v2_5.md](v2_5.md), [v2_5_results.md](v2_5_results.md)를 먼저 읽는다.
아래 V2 설명은 당시의 기록이다. V1/V2 명령/API를 보존하고 `boardnotes.v25`를 추가했다.
네 수식 모델을 실제 CPU에서 비교했으며 기본 S는 유지한다. 자동 영역은 검토가 필요하다.
기존52개를 포함해131테스트 통과. MIT 실제12초 영상의 영상→키프레임→자동인식→PDF도 실행했다.
가중치/원본 clip/실행 캐시는 Git 제외이고 비교 자료는 `docs/assets/v25`에 있다.
추가 주석22영역은 추론 전에 고정했다. 행렬의 문자열 불일치와 의미 오류를 구분한다.
V2.5 구현과 실험 근거를 work 브랜치에 함께 보존한다.

## V2 지정 영역 구현 및 실제 검증 (후속 작업)

V2 Phase A 구현과 실제 CPU 비교를 추가했다. 먼저 [v2.md](v2.md)와
[v2_results.md](v2_results.md)를 읽는다. 기존 V1 기본 경로는 그대로다.

- `boardnotes/v2/cli.py`: recognize/detect/render. JSON bbox는 원본 입력 half-open 좌표,
  timestamp는 초. text/formula/other/unknown을 구분한다.
- 모델은 별도 `/workspace/lecturemath-v2-env`에서 PP-OCRv5 Mobile/PP-FormulaNet-S를 순차 실행한다.
  `requirements/v2-ocr.txt` 및 실제 전체 lock을 보존했다. V1 Torch/OpenCV는 변경하지 않았다.
- 사용자 확인 후 인쇄체 중심 S로 손글씨의 한계를 먼저 평가했다. 더 큰 모델로 대체하지 않았다.
- 실제 3프레임의 제목 3개/수식 6영역 × RGB/B0/B3, 총27개 예측을 실행했다.
  CER은 13.61/4.84/4.84%, 수식 LaTeX 엄격 일치는 4/6·5/6·5/6이다.
  이는 작은 탐색 표본으로 일반 성능이 아니다. 연립방정식이 틀렸으며 모든 수식에 사람 검토가 필요하다.
- 자동 검출은 실제 후보15개를 unknown/needs_review JSON으로 내보낸 수준이다.
  text/formula 자동 분류 완료로 보지 않는다. 수정 JSON을 지정 영역 명령에 다시 넣을 수 있다.
- 원본 전체 이미지 + 검색/복사 가능한 Unicode/LaTeX 결과 페이지 PDF, JSON-only 재생성,
  crop/model/config/버전/가중치 fingerprint 캐시를 지원한다.
- 전체52테스트 통과. fixture 테스트와 실제 Paddle 결과는 문서에서 분리했다.
- `--export-v2-inputs`는 기본False이며 기존 V1 출력은 바꾸지 않는다. ST3D 합성 키프레임의
  untrimmed 이미지/시간/좌표 manifest를 내보낸다. 단일 RGB와 정렬된다고 가정하지 않는다.
- `docs/assets/v2`에 재배치 가능한 JSON/원본/crop, 비교PNG/CSV/JSON, 대표PDF를 보존했다.
  실제 모델 가중치/실행 캐시는 `models/v2`, `benchmark-output/v2`에만 있다.

다음에는 자동 수식 영역 분류, 복잡한 손글씨 모델/분할 전략, 한국어+글꼴,
수식 시각 렌더링과 실제 연속 영상 연계를 별도로 결정·검증해야 한다.
문서의 literal LaTeX 검색과 수식이 시각적으로 정확하게 조판된다는 주장을 구분한다.

## V1.5 실험 완료 및 V2 시작 기준 (2026-10-28)

이후 V1.5 실험을 추가했다. V2에서는 **B0(V34 기준선)와 B3(선택적 refinement 후보)**를
유지하고 원본 RGB도 함께 보존한다는 방향으로 사용자와 합의했다.
기존 V1 영상→PDF의 기본 동작은 바꾸지 않았으며 B3를 자동 적용하지 않았다.

- 구현/명령: [v1_5.md](v1_5.md), 독립 모듈 `experiments/v1_5/`.
- 실제 결과/한계: [v1_5_results.md](v1_5_results.md).
- 후속 평가 지시문 검토: [v1_5_evaluation_review.md](v1_5_evaluation_review.md).
- 최초 합성 결과는 [v1_5_synthetic_results.md](v1_5_synthetic_results.md)에 보존.
- 실제 AccessMath2015 3프레임, native1080p/derived720p, 기존 후보와 소수 CLAHE 조건 실행.
  96개 결과, 영어 제목 Tesseract 48회, 비교 PNG/CSV/JSON 및 18페이지 PDF를 보존했다.
- 기존 V1 14개 + V1.5 7개 테스트 통과. 실제 V34에서 B0와 기존 V1의 픽셀 동일성도 확인했다.

V2 연결 시 주의할 계약:

1. 원본 RGB, V1.5 출력(검정 판서/흰 배경), 원본 좌표/해상도/ROI/파라미터를 함께 보존한다.
   `boardnotes.model.LectureNet.extract()` 출력은 ST3D용 흰 판서/검정 배경이므로 극성을 구분한다.
2. B3는 `B0 OR (dilate(B0) AND Sauvola)`이며 기존 전경을 제거하지 않는다.
   완전히 놓친 독립 영역을 복원하지 못하고 획을 두껍게 하거나 오검출을 늘릴 수 있다.
   기본 window25/k.2/radius2/CLAHE off는 기준 후보이고, CLAHE on은 탐색 조건이다.
3. 1080p 평균 픽셀 F1은 B0 88.91%, 기본 B3 88.93%, B3+CLAHE 89.27%였다.
   작은 픽셀 개선이 영어 OCR 개선으로 이어지지 않았다. B3가 OCR에 더 좋다고 확정하지 않는다.
4. Tesseract 영문 제목 3영역은 수식 인식 검증이 아니다. 수식 OCR/LaTeX 구조화는 V2의 작업이다.
   인식 engine과 전사 정답을 고정해 RGB/B0/B3를 같은 crop에서 비교한다.
5. 실제 독립 프레임은 3장뿐이며 학습 포함 여부는 미확인이다.
   12~20장 확대 평가, 검은 칠판/교수자 가림/모션 블러/연속 영상의 시간적 안정성은 미완료다.
   파라미터 선택용과 평가용 자료를 강의 단위로 분리한다.

재현용 원본/GT URL과 해시는 `docs/assets/v1_5_real/metrics.json`에 보존했다.
대용량 PNG/확률 캐시는 `benchmark-output/v1_5-real/`에만 있으며 clone에 포함되지 않는다.
아래의 기존 V1 상태/검증 기록도 함께 참고한다.

## 현재 상태

FCN-LectureNet V34와 원본 ST3D를 사용하는 V1 구현이 완료되어 있다.
CLI·FastAPI·정적 웹 UI가 있으며, 합성 영상 전체 흐름과 실제 1080p 강의 프레임 한 장을 검증했다.
**실제 고해상도 연속 영상의 시간적 통합은 다음 우선 검증 항목이다.**

저장소: `https://github.com/hwoojo/lecturemath`
이번 작업은 `work` 브랜치에 정리했다. 기존 연구 코드의 기준 커밋은 `245d58a20393166addacb9ece75326ef3455b16f`다.
기본 브랜치 `master`에 병합하거나 실제 서비스를 배포하는 작업은 포함하지 않았다.

먼저 읽을 문서:

1. [README](../README.md): 설치·모델·CLI·API·웹 실행과 설정
2. [작업 기록 및 고해상도 품질 비교](WORK_LOG_HD_QUALITY.md): 지금까지의 작업과 재현 명령
3. [검증 보고](VALIDATION.md): 실제 검증/합성 테스트/미검증 구분
4. [원본 분석](UPSTREAM_ANALYSIS.md): ST3D 입출력·의존성·원본 수정 근거
5. [배포 및 Jetson](DEPLOYMENT.md): Pages/API 분리 배포와 ARM64 이식

## 새 환경에서 바로 시작하기

```bash
git clone --branch work https://github.com/hwoojo/lecturemath.git
cd lecturemath
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu
python -m pip install -r requirements-dev.txt
bash scripts/build_native.sh
python -m pip install -e .
python -m pytest -q
```

검증한 환경은 Linux x86_64, Python 3.12.14, torch 2.5.1+cpu/torchvision 0.20.1+cpu다.
프로덕션 의존성은 `requirements.txt`에 있다. torch/torchvision은 CPU/CUDA/Jetson에 맞춰 별도 설치한다.
저장소 체크아웃에서 실행하는 프로젝트이며 독립 wheel만 복사해서 실행하는 구성은 아니다.

원본 저장소의 `accessmath_lib.so` 대신 빌드 스크립트가 생성하는 `accessmath_lib.native.so`를 사용한다.
네이티브 바이너리는 커밋하지 않으므로 새 머신에서 반드시 빌드한다.

현재 세션에 남은 가상환경 `/workspace/lecturemath-env`는 개발 편의용이며 Git clone에 포함되지 않는다.
`.env.example`은 예시이고 자동으로 읽지 않는다. 셸 `export`나 서비스 설정을 사용한다.

## 체크포인트

사용자가 제공한 실제 V34를 **사용자의 후속 요청에 따라 Git에 포함**했다.

- 경로: `models/LectureNet_model_BIN_V34_final.dat`
- 크기: 63,319,863바이트
- SHA-256: `d3721339db6cc2d2de2a29816bfcaca38f2d4c4bc8a1eb75a7402a194604b144`
- Git LFS 포인터가 아닌 실제 파일이다.
- `BOARD_MODEL_PATH` 또는 CLI `--model`로 별도 경로를 지정할 수 있다.
- 모델 파일이 없으면 명시적으로 실패한다. 학습되지 않은 `synthetic.dat`를 대체 모델로 사용하지 않는다.

## 주요 코드 위치

| 파일 | 역할 |
|---|---|
| `boardnotes/config.py` | 장치·모델·샘플링·영상 제한·출력 스타일 설정 |
| `boardnotes/model.py` | 원본 모델 생성, 안전한 state_dict 로딩, CPU/CUDA 추론 |
| `boardnotes/video.py` | 일반 영상 메타데이터·순차 디코딩·샘플링 |
| `boardnotes/upstream.py` | 원본 모듈 경로와 연구 설정 어댑터 |
| `boardnotes/temporal.py` | 원본 ST3D 단계 연결 및 짧은 삭제 충돌 보완 분할 |
| `boardnotes/pdf.py` | 빈/완전 중복 제거, 여백 crop, 종횡비 보존, PDF 표시 |
| `boardnotes/pipeline.py`, `cli.py` | 공통 영상→PDF 파이프라인과 CLI |
| `boardnotes/api.py`, `worker.py` | 작업 접수·상태·다운로드, 추론 subprocess와 정리 |
| `frontend/` | 빌드 없이 배포하는 정적 UI; API 주소는 `config.js` |
| `scripts/evaluate_keyframe.py` | 실제 프레임 한 장의 해상도별 모델/PDF 평가 |
| `scripts/make_demo_video.py`, `verify_web.py` | 합성 테스트 영상 및 선택적 실제 브라우저 검증 |

원본 연구 코드에서 변경한 파일은 모델의 장치 이동, C 라이브러리 경로 3곳, NumPy bool 호환성이다.
원본 CC 추적·그룹화·삭제 이벤트 함수는 어댑터가 재사용한다. 학습이나 fine-tuning을 하지 않았다.

## 실행과 운영 제약

```bash
python scripts/make_demo_video.py /tmp/board-demo.mp4
python -m boardnotes.cli /tmp/board-demo.mp4 --output-dir /tmp/board-demo-result --keep-intermediates
```

CLI 출력 디렉터리는 비어 있어야 한다. `--keep-intermediates`에서 마스크·ST3D pickle·최종 페이지를 보존한다.
pickle은 자체 디버그 산출물이며 외부 업로드 pickle을 입력으로 받지 않는다.

API와 UI는 별도 터미널에서 실행한다.

```bash
python -m uvicorn boardnotes.api:app --host 0.0.0.0 --port 8000 --workers 1
```

```bash
python -m http.server 8080 --directory frontend
```

- API `POST /jobs`는 **multipart가 아닌 원문 영상 body**를 받는다.
- 전체 추론은 subprocess에서 실행한다. Redis/Celery는 도입하지 않았다.
- 동시에 업로드 또는 추론 한 작업만 허용한다. Uvicorn worker는 반드시 1개로 실행한다.
- CORS는 설정된 origin만 허용한다. 공개 서비스용 로그인·인증은 아직 없다.
- 업로드 원본은 성공/실패 후 삭제하고 결과 PDF/로그/리포트는 기본 24시간 후 정리한다.
- Pages는 정적 UI만 제공한다. 실제 추론은 별도 HTTPS FastAPI 서버가 필요하다.

## 확인한 결과

- 자동 테스트 14개 통과. API 단위 테스트에는 명시적인 fake runner를 사용한다.
- 실제 V34로 생성한 6·24·48초 합성 영상의 CLI/PDF 생성 성공.
- 실제 브라우저에서 합성 영상 업로드→상태 조회→실제 모델→PDF 다운로드 성공.
- 실제 AccessMath2015 1920×1080 프레임에서 모델 추론 및 PDF 생성 성공.
- 같은 실제 프레임의 픽셀 F1: 1080p 90.77%, 720p 82.22%, 448×252 40.89%.
- 1080p 단일 프레임 CPU 추론 11.169초, 프로세스 최대 RSS 약 3 GiB.

실데이터 점수는 한 프레임의 정확 픽셀 일치이며 OCR 정확도·영상 전체 성능이 아니다.
이 프레임이 V34 학습에 포함됐는지는 확인하지 못했으므로 독립 평가셋 성능으로 주장하지 않는다.
저해상도는 획 손실의 큰 원인이었다. 기본 폭 상한은 1280이며, 비교할 때 `--max-width 1920`을 지정한다.
`smooth` PDF는 보간에 의한 경계 표시 개선이고, 이미 손실된 획을 복원하지 않는다.

## 다음 작업 우선순위

1. 실제 1080p 강의의 짧은 연속 구간을 확보한다. 실제 영상 메타데이터와 원본 출처를 기록한다.
2. 같은 구간을 폭 1280/1920으로 처리해 원본·FCN 이진화·PDF 직전 이미지 및 삭제/가림 보존을 비교한다.
3. 잡음과 미세 획 손실을 분리해 분석한다. 작은 점·첨자·소수점 보존 검증 없이 성분 삭제/강한 평활화를 추가하지 않는다.
4. 고해상도 영상에서 최대 RSS·CPU 시간·디스크 사용량을 측정한 뒤 기본 해상도/길이/샘플 제한을 결정한다.
5. 품질과 자원이 확인되면 외부 API+Pages 배포, 이후 Jetson 실기 검증을 진행한다.

## 막힌 사항과 미검증 범위

- 이 환경은 GitHub raw와 패키지 저장소에는 연결되지만 YouTube/Dropbox/Box는 프록시 CONNECT 403이었다.
  네트워크 정책이 변경되었는지 실제 요청과 환경 상태로 다시 확인한다. 차단을 우회하지 않는다.
- 실제 고해상도 연속 영상, 카메라 이동, 실제 교수자 가림의 정량 평가가 남아 있다.
- CUDA·Jetson, 30분 고해상도 영상 부하, VFR의 정밀 타임스탬프는 미검증이다.
- 항상 가려지거나 안정 CC 조건을 만족하지 못한 판서는 보존하지 못할 수 있다.
- PDF는 완전 중복만 제거한다. 부분 중복은 문맥 보존을 위해 남는다.

## Git에 보존한 것과 로컬 산출물

소스·설정·테스트·문서·요약 JSON·비교 그림·V34 가중치를 저장소에 포함했다.
다운로드한 전체 이미지/영상, 생성 PDF, 작업 디렉터리, 빌드한 `.so`, 가상환경은 Git ignore 대상이다.
`benchmark-output/`는 현재 머신에만 남은 검증 산출물이며 새 clone에는 없을 수 있다.
실데이터 원본 URL과 해시 및 다시 받는 명령은 작업 기록에 있어 재현할 수 있다.
비교 그림은 `docs/assets/real-hd-comparison.png`라 GitHub에서도 확인할 수 있다.

현재 브랜치와 최근 커밋은 `git status -sb`, `git log -3 --oneline`으로 확인한다.
