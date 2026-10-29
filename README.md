# 강의 영상 자동 판서 문서화 V1

일반 강의 영상 → **사전 학습 FCN-LectureNet 판서 추출** → 원본 ST3D CC 분석·그룹화 → 삭제 이벤트 분할 → 흑백 판서 PDF.
CLI, FastAPI 비동기 작업 API, GitHub Pages용 정적 프론트엔드를 포함한다. OCR·음성 전사·LLM·학습·대체 모델은 사용하지 않는다.

V2 지정 영역 OCR/수식 인식의 로컬 구현과 실제 CPU 실험을 추가했다.
[설치·CLI·JSON 계약](docs/v2.md), [실제 결과와 미완료 범위](docs/v2_results.md)를 참고한다.
기존 영상→PDF 기본 경로는 유지하며, 자동 텍스트·수식 영역 분류는 아직 검토 단계다.

V2.5는 자동 영역 제안, 네 수식 모델 교체, 여러 줄 분할, LaTeX 렌더링과
실제 MIT 강의 영상 E2E를 추가했다. [실행 방법](docs/v2_5.md),
[실제 결과·한계](docs/v2_5_results.md), [모델 비교 PDF](docs/assets/v25/comparison.pdf)를 참고한다.

V2.51은 기존 네 모델의 실패 조건을 인쇄/판서, 구조, crop·스케일, 행 분할로 나누어 진단한다.
[실험 설계](docs/v2_51.md), [원인 분석](docs/v2_51_results.md),
[판서·인쇄 비교 PDF](docs/assets/v251/comparison.pdf)를 참고한다. 기본 파이프라인은 유지한다.

연구 코드 분석과 변경 근거는 [UPSTREAM_ANALYSIS](docs/UPSTREAM_ANALYSIS.md), 실제 검증 범위는 [VALIDATION](docs/VALIDATION.md), 배포/Jetson 안내는 [DEPLOYMENT](docs/DEPLOYMENT.md)를 참고한다.
V1 작업 기록과 실제 1080p 데이터의 해상도별 품질 비교는 [고해상도 품질 검증 작업 기록](docs/WORK_LOG_HD_QUALITY.md)에 정리했다.
후속 작업자는 [인수인계 문서](docs/HANDOFF.md)에서 현재 상태, 실행 순서, 다음 작업과 알려진 한계를 먼저 확인한다.
원본 연구 코드는 ACCESS2021_release에 유지하며 [MIT 라이선스](LICENSE)와 출처를 보존했다.

## 설치 (Linux CPU, Python 3.10~3.12)

```bash
cd /path/to/lecturemath
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu
python -m pip install -r requirements-dev.txt
bash scripts/build_native.sh
python -m pip install -e .
```

C 컴파일러(`cc`, GCC/Clang), Python 개발 환경이 필요하다. FFmpeg 지원 OpenCV로 디코딩하며 시스템 ffmpeg도 디버깅에 유용하다.
CUDA/Jetson은 먼저 플랫폼에 맞는 torch/torchvision을 설치한다. PyTorch 휠은 requirements에서 고정하지 않았다.
editable 설치는 선택 사항이며 저장소 루트에서 `python -m boardnotes.cli`로 실행할 수 있다.
연구 소스를 함께 참조하므로 이 프로젝트는 저장소 체크아웃에서 실행한다(독립 wheel 배포용이 아님).

현재 준비된 개발 환경은 `/workspace/lecturemath-env/bin/python`이다. V34 체크포인트는 `models/LectureNet_model_BIN_V34_final.dat`로 저장소에 포함한다. 빌드한 C 라이브러리와 다른 모델 파일은 Git에서 제외한다.

## 체크포인트 준비

V34 체크포인트가 `models/LectureNet_model_BIN_V34_final.dat`에 포함되어 있어 기본 설정으로 사용할 수 있다.
파일이 없는 별도 배포 환경에서는 공식 [V34 다운로드](https://www.dropbox.com/s/ea0266hm1vkcjwc/LectureNet_model_BIN_V34_final.dat?dl=0) 파일을 준비한다.

```bash
mkdir -p models
# 파일을 이 위치에 복사하거나 아래 환경변수로 다른 경로를 지정
export BOARD_MODEL_PATH=/absolute/path/LectureNet_model_BIN_V34_final.dat
```

다운로드 403이 발생하면 직접 받은 파일을 지정한다. **가중치가 없으면 CLI는 실패, 업로드 API는 503을 반환**한다.
학습되지 않은 synthetic.dat 또는 다른 모델로 대체하지 않는다. torch state_dict를 weights_only=True와 strict=True로 읽는다.
이번 세션에서는 사용자가 제공한 분할 체크포인트를 합쳐 `models/LectureNet_model_BIN_V34_final.dat`에 준비하고 실제 추론을 확인했다.
SHA-256: `d3721339db6cc2d2de2a29816bfcaca38f2d4c4bc8a1eb75a7402a194604b144` (63,319,863바이트).

첨부 전송 한도를 피하려고 파일을 나눴다면 다음처럼 명시적인 순서로 합친다.

```bash
cat checkpoint.part-aa checkpoint.part-ab checkpoint.part-ac checkpoint.part-ad > models/LectureNet_model_BIN_V34_final.dat
sha256sum models/LectureNet_model_BIN_V34_final.dat
```

## CLI

```bash
python -m boardnotes.cli /path/to/lecture.mp4 --output-dir /path/to/new-output
# 자세한 중간 결과 보존, 장치/샘플링/해상도 선택
python -m boardnotes.cli /path/to/lecture.mp4 --output-dir /path/to/debug-output \
  --model /path/to/V34.dat --device cpu --fps 1 --max-width 1280 --keep-intermediates
```

출력 폴더는 비어 있어야 한다. `result.pdf`, `report.json`, `pipeline.log`, 분할 분석 PNG가 생성된다.
`--keep-intermediates`에서는 `masks/`, `pages/`, `temporal/*.pickle`도 보존한다.
중간 마스크는 검정 바탕/흰 판서이며 PDF/페이지 이미지는 흰 바탕/검정 판서다.
`--no-timestamps`로 시간 표기를 끈다. `--research-config <사본.conf>`로 CC/삭제 임계값을 조정한다.
PDF는 기본 `--pdf-style smooth`로 출력한다. 이진 판서를 3배 bicubic 보간해 경계에 회색 픽셀을 추가하므로
확대 시 픽셀 테두리가 조금 부드러워진다. 추론·시간적 통합·중복 검사는 원래 이진 마스크로 수행한다.
순수 0/255 출력을 유지하려면 `--pdf-style binary` 또는 `BOARD_PDF_STYLE=binary`를 지정한다.
이는 표시 방식의 개선이며 이미 사라지거나 뭉개진 획을 복구하지 않는다. 원본 영상이 저해상도이면 효과가 제한적이다.
`--keep-intermediates`의 pages 이미지는 PDF에 실제 삽입한 최종 이미지다. smooth에서는 폭·높이가 각각 3배이며,
PDF 내 물리적 크기와 종횡비는 binary와 같다. 고해상도 판서 품질 확인에는 원본 1080p 이상 영상으로 비교하는 것이 좋다.
가독성에 영향을 줄 수 있는 축소가 싫다면 max-width를 영상 폭 이상으로 지정하되 CPU 메모리를 먼저 확인한다.
모델 자체의 2.5MP 내부 축소는 원본 동작이다.

## 웹 실행

터미널 1:

```bash
python -m uvicorn boardnotes.api:app --host 0.0.0.0 --port 8000 --workers 1
```

터미널 2:

```bash
python -m http.server 8080 --directory frontend
```

브라우저에서 `http://localhost:8080`을 열어 영상을 선택한다. 영상은 서버에 스트리밍 업로드되고 작업 ID를 반환한다.
프론트엔드는 2초마다 상태를 확인해 완료 후 PDF 다운로드 링크를 제공한다. 새로고침 시 같은 브라우저의 미완료 작업을 다시 조회한다.
API 주소는 `frontend/config.js`, 허용 origin은 `BOARD_CORS_ORIGINS`에서 설정한다.

| API | 동작 |
|---|---|
| `GET /health` | 서버 상태, 모델 파일 존재 여부(유효성/품질 보증 아님), 업로드 제한 |
| `POST /jobs?fps=1&timestamps=true` | **multipart가 아닌 영상 파일 원문 body**를 업로드. 202와 작업 ID 반환 |
| `GET /jobs/{id}` | queued/running/complete/failed, stage, progress, 오류, 완료 시 다운로드 경로 |
| `GET /jobs/{id}/pdf` | 완료한 PDF 다운로드 |

```bash
curl -X POST 'http://localhost:8000/jobs?fps=1&timestamps=true' \
  -H 'Content-Type: application/octet-stream' --data-binary @lecture.mp4
curl http://localhost:8000/jobs/JOB_ID
curl -o notes.pdf http://localhost:8000/jobs/JOB_ID/pdf
```

오류: 모델 미준비 503, 진행 중 작업 있음 409, 업로드 크기 초과 413, 잘못된 영상/길이/샘플 초과 422, 결과 미준비 409, 만료/알 수 없는 ID 404.
영상 유효성/메타데이터 검사는 작업 접수 전에 수행하고 전체 추론은 subprocess에서 실행한다. 별도 Redis/Celery는 사용하지 않는다.
진행률은 단계 기반 추정치이며 시간 예측이 아니다.

## 설정

`.env.example`은 예시이며 자동 로딩하지 않는다. 셸 export 또는 서비스 환경변수로 설정한다.

| 환경변수 | 기본값 |
|---|---|
| `BOARD_MODEL_PATH` | `models/LectureNet_model_BIN_V34_final.dat` (저장소 기준) |
| `BOARD_DEVICE` | `cpu` (`cuda`, `cuda:N` 선택 가능) |
| `BOARD_SAMPLING_FPS` | 1 |
| `BOARD_MAX_WIDTH` | 1280 |
| `BOARD_PDF_STYLE` | `smooth` (`binary` 선택 가능) |
| `BOARD_MAX_UPLOAD_MB` | 500 MiB |
| `BOARD_MAX_DURATION` | 1800초 |
| `BOARD_MAX_SAMPLES` | 1800 |
| `BOARD_THREADS` | 2 (PyTorch CPU threads) |
| `BOARD_JOBS_DIR` | 저장소 아래 `jobs/` |
| `BOARD_RETENTION_SECONDS` | 86400초 |
| `BOARD_CORS_ORIGINS` | `http://localhost:8080,http://127.0.0.1:8080` |
| `BOARD_RESEARCH_CONFIG` | `ACCESS2021_release/configs/FCN_LectureNet.conf` |
| `ACCESSMATH_LIB` | 빌드한 `ACCESS2021_release/accessmath_lib.native.so` |

원본 업로드는 성공/실패 후 삭제한다. API 중간 이미지는 자동 정리하고 PDF/리포트/로그는 TTL까지 보존한다.
CLI 결과는 자동 삭제하지 않는다. 고정 카메라의 칠판 강의를 우선 대상으로 하며, 완전히 가려진 판서나 모든 작은 획의 보존을 보장하지 않는다.
완전히 동일한 페이지는 제거하지만 서로 다른 페이지의 공통 판서는 문맥 보존을 위해 남는다.

## 테스트

```bash
python -m pytest -q
# 실제 V34를 쓰는 합성 영상 CLI smoke test
python scripts/make_demo_video.py /tmp/board-demo.mp4
python -m boardnotes.cli /tmp/board-demo.mp4 --output-dir /tmp/board-demo-result --keep-intermediates
# 선택: 실제 브라우저 + 실제 추론 API
python -m pip install playwright==1.49.1
python scripts/verify_web.py /tmp/board-demo.mp4
```

마지막 스크립트는 `/usr/bin/chromium`과 비어 있는 8000/8080 포트가 필요하다. 테스트 서버를 직접 시작/종료한다.
Codex sandbox에서 TestClient가 스레드 이벤트 루프 진입 시 대기한다면 로컬 소켓을 허용한 실행 권한으로 실행한다.
이 환경에서는 네트워크 권한을 부여한 테스트 실행으로 해결했다.

자동 테스트는 모델을 흉내 낸 API 실행기와 합성 이진 판서를 사용한다. **실제 모델을 사용하는 테스트와 구분**되어 있으며 자세한 결과와 제한은 VALIDATION에 기록했다.

---

## 원본 저장소 안내

## Welcome to LectureMath

The files in "[ACCESS2021_release](https://github.com/kdavila/lecturemath/tree/master/ACCESS2021_release)" are for the paper *FCN-LectureNet: Extractive Summarization of Whiteboard and Chalkboard Lecture Videos* (IEEE ACCESS 2021)

The files in "[DLHAU2020_release](https://github.com/kdavila/lecturemath/tree/master/DLHAU2020_release)" are for the paper 
*Skeleton-based Methods for Speaker Action Classification on Lecture Videos* (ICPRW DL-HAU 2020)

The latest release of our video annotation tool can be found [here](https://github.com/adaniefei/vid_annotation_tool)



## V1.5 CPU binarization experiments

Independent, training-free comparisons preserving the original release:
[commands and implementation](docs/v1_5.md), [results and limitations](docs/v1_5_results.md),
[module README](experiments/v1_5/README.md).
