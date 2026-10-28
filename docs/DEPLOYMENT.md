# 배포와 Jetson 이식

## 정적 프론트엔드 / GitHub Pages

`frontend/`는 빌드 도구 없이 index.html, style.css, app.js, config.js만 제공한다.
`config.js`의 `window.BOARDNOTES_API_URL`을 실제 HTTPS API 주소로 변경한 뒤 이 네 파일을 Pages 배포 경로에 올린다.
GitHub Pages에서 **추론 Python 서버는 실행되지 않는다**. FastAPI는 별도 서버에서 실행한다.
GitHub Actions를 사용한다면 `frontend/` 디렉터리를 Pages artifact로 배포하면 된다.
이 작업에서 실제 Pages 게시나 외부 추론 서버 배포는 수행하지 않았다.

HTTPS Pages에서는 API도 HTTPS여야 브라우저의 mixed content 차단을 피할 수 있다.
서버의 BOARD_CORS_ORIGINS에는 `https://<owner>.github.io`처럼 origin만 지정한다. 저장소 하위 경로는 넣지 않는다.
역방향 프록시 업로드 크기/시간 제한도 앱의 500 MB 등 설정에 맞춰야 한다.

## 백엔드 실행

```bash
export BOARD_MODEL_PATH=/srv/lecturemath/models/LectureNet_model_BIN_V34_final.dat
export BOARD_CORS_ORIGINS=https://example.github.io
export BOARD_JOBS_DIR=/srv/boardnotes-jobs
python -m uvicorn boardnotes.api:app --host 0.0.0.0 --port 8000 --workers 1
```

**단일 worker 프로세스**로 실행해야 작업 슬롯 1개가 서버 전체에 적용된다. `--reload`는 개발용이다.
각 추론은 별도 Python subprocess에서 실행되고 종료 시 모델 메모리를 반납한다.
작업 상태는 각 디렉터리 status.json에 저장한다. 재시작 시 미완료 작업은 실패로 표시하고 입력을 삭제한다.
API는 원본 업로드를 완료/실패 시 삭제하며 기본적으로 중간 이미지도 삭제한다.
결과 PDF·로그·리포트는 기본 24시간 보관 후, 실행 중 최대 60초 간격 및 재시작 시 정리한다.
CLI의 산출물은 사용자가 지정한 폴더에 보존하고 자동 삭제하지 않는다.

V1에는 사용자 로그인, 작업 취소, 분산 큐가 없다. ID를 아는 사람이 결과를 받을 수 있다.
외부 공개 서비스로 운영할 때는 배포 계층에서 인증/속도 제한을 설정해야 한다. CORS는 인증을 대신하지 않는다.
정해진 디코딩·샘플 상한 외에 OS/cgroup 메모리 한도를 두면 실제 강의의 자원 사용량을 관리하기 쉽다.

## Jetson Orin NX 8GB (ARM64)

1. JetPack/L4T와 CUDA 버전을 먼저 확인하고 이에 맞는 NVIDIA 배포 PyTorch/torchvision 조합을 설치한다.
   x86 CPU 개발용 `torch==2.5.1+cpu` 휠은 Jetson에 복사하지 않는다. Python 버전도 JetPack과 맞춘다.
2. `requirements.txt`의 NumPy/SciPy/scikit-learn/OpenCV 등 ARM64 휠과 Python 버전 호환성을 확인한다.
   wheel이 없으면 시스템 패키지 또는 ARM 빌드를 사용한다. 시스템 OpenCV를 쓴다면 pip의 opencv-python-headless와 중복 설치하지 않는다.
   PyTorch가 요구하는 NumPy ABI에 맞게 버전을 조정한다. 현재 핀은 x86_64 Python 3.12에서 검증했다.
3. Jetson에서 `bash scripts/build_native.sh`를 실행해 ARM64 네이티브 라이브러리를 새로 만든다.
   이 C 코드는 SIMD/x86 전용 명령을 추가하지 않았다.
4. `BOARD_DEVICE=cuda` 또는 `cuda:0`을 설정한다. 장치를 지정하지 않으면 CPU다.
5. 8GB는 CPU/GPU 공유 메모리다. 우선 BOARD_MAX_WIDTH=640, 짧은 입력, 낮은 max_samples로 시작해 측정한 뒤 확대한다.
   1 FPS도 영상당 추론 시간이 충분히 길 수 있다. 가중치는 CPU에서 먼저 로딩해 설정한 장치로 옮긴다.
6. 같은 CLI/API 테스트와 실제 강의 품질 테스트를 Jetson에서 다시 수행한다. TensorRT, FP16, 모델 재학습은 구현하지 않았다.

Jetson 실기, CUDA 추론, 장시간 부하 테스트는 이번 CPU 환경에서 검증하지 않았다.
