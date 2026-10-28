# V1 검증 보고

검증일: 2026-10-27 (Asia/Seoul). 구현 위치: `/workspace/lecturemath`.

## 환경과 모델

- Linux x86_64, Python 3.12.14, torch 2.5.1+cpu, torchvision 0.20.1+cpu, OpenCV 4.10.0, FFmpeg 설치 확인.
- CUDA 사용 가능 여부: False. PyTorch CPU thread 수: 2.
- 공식 저장소 HEAD와 로컬 기준 커밋 `245d58a20393166addacb9ece75326ef3455b16f` 일치 확인.
- V34 공식 Dropbox 다운로드는 프록시 CONNECT 단계 HTTP 403. 이후 사용자가 제공한 aa/ab/ac/ad 4개 파일을 순서대로 연결.
- 복원 크기 63,319,863바이트, SHA-256 `d3721339db6cc2d2de2a29816bfcaca38f2d4c4bc8a1eb75a7402a194604b144`.
- 실제 모델의 strict state_dict 로딩, eval, CPU 추론 성공. 256×448 테스트 프레임에서 0/255 이진 마스크, 판서 foreground 1,546픽셀 확인.
- 기존 작업공간의 `lecturemath-local/artifacts/synthetic.dat`는 학습되지 않은 별도 파일이며 이번 모델 검증에서 사용하지 않았다.

## 실제 V34 + 생성 영상 통합 테스트

`scripts/make_demo_video.py`는 448×256, 10 FPS MP4를 생성한다.
두 식을 영상 중간에 지우고 같은 위치에 다른 식을 표시한다. 다른 행의 식은 유지한다.
24/48초 영상에는 4~6초 동안 색 사각형으로 일부 판서를 가린다.
**이 영상은 프로그램으로 그린 판서이며 실제 강의나 실제 교수자 영상이 아니다.**
모델 추론은 실제 사용자가 제공한 V34로 수행한다. 영상의 정답 이진 마스크를 모델 입력 대신 넣지 않는다.

최종 코드에서 실행한 CLI 결과 (독립 CLI 프로세스, 1 FPS 샘플링, 해상도 축소 없음):

| 영상 길이 | 샘플 | PDF 페이지 | 소요 시간 | 최대 RSS (MiB) | 원본 삭제 구간 / 보완 분할 |
|---|---:|---:|---:|---:|---|
| 6초 | 6 | 2 | 7.503초 | 514.30 | `[0,5]` → `[0,2]`, `[3,5]` (추가 분할 1회) |
| 24초 | 24 | 2 | 15.962초 | 509.95 | `[0,10]`, `[12,23]`, 추가 분할 0 |
| 48초 | 48 | 2 | 32.594초 | 547.94 | `[0,22]`, `[24,47]`, 추가 분할 0 |

RSS는 Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss / 1024`로 측정한 프로세스 high-water mark다.
CLI는 한 프로세스에서 한 작업만 실행했다. API는 추론마다 새 subprocess를 실행하므로 작업별 메모리를 해제한다.
디코더/모델/그룹화/리포트 작업을 포함한 시간이며, 호스트 부하와 파일 캐시의 영향을 받는다.
긴 실제 강의나 고해상도 영상에 이 수치를 그대로 외삽할 수 없다.

24초 영상에서 원시 CC 관측 306개 → 고유 CC 18개 → 그룹 16개, 최종 요약 2페이지를 확인했다.
이는 추적·통합이 수행됐다는 계수이며 정확도/재현율 또는 중복 제거율을 뜻하지 않는다.
PDF 내부 이미지가 원본 RGB 프레임이 아닌 흰 배경의 이진 판서임을 페이지 PNG와 PDF 이미지로 확인했다.
화면에 남은 동일 행은 두 요약 페이지에 문맥으로 재등장한다. 완전 동일한 페이지의 제거는 별도의 테스트로 검증했다.

초기 6초 실행에서는 원본 최소 분할 길이 때문에 삭제 이벤트 분할이 `[0,5]` 하나를 반환했고,
원본 keyframe의 충돌 처리로 오래된 판서가 탈락했다. 원본 CC 충돌 정보 중 수명이 분리된 그룹에 한해서
보완 분할을 추가한 뒤 같은 영상을 재실행해 이전/이후 판서가 각각 별도 페이지에 포함되는 것을 확인했다.
24/48초는 원본 삭제 이벤트만으로 이미 분리되어 보완 분할이 필요하지 않았다.

## 실제 브라우저 + 실제 추론 API

`scripts/verify_web.py`로 로컬 정적 서버와 FastAPI를 시작하고 시스템 Chromium을 Playwright로 조작했다.
24초 영상에서 실제 업로드→202 작업 ID→반복 상태 조회→실제 V34 subprocess→2페이지 PDF 다운로드를 확인했다.
보완 분할 수정 후에는 6초 영상으로 다시 전체 브라우저 경로를 실행해 2페이지 PDF 다운로드를 확인했다.
마지막 API worker 실행: 7.361초, 최대 RSS 542.95 MiB.

- 데스크톱 1100×1000 / 모바일 390×844 화면 확인.
- JavaScript pageerror 0건, 모바일 가로 overflow 없음.
- CORS가 적용된 별도 origin(8080 frontend / 8000 API) 간 요청 성공.
- 다운로드한 실제 PDF의 헤더와 페이지 수 확인.
- 완료 후 작업 디렉터리에서 원본 업로드 파일 및 비보존 중간 디렉터리 삭제 확인.
- 상태/로그/리포트/PDF는 보관, TTL 만료 정리는 별도의 제어된 자동 테스트로 확인.
- 실제 GitHub Pages와 외부 HTTPS 서버에는 배포하지 않았다.

## 모델 없이 실행하는 자동 테스트

최종 `python -m pytest -q`: **13 passed, 59 warnings, 2.95s**.
이 테스트들의 API runner는 합성 PDF를 만드는 명시적인 fake runner이며 학습 모델을 가장하지 않는다.

| 테스트 | 검증 내용 |
|---|---|
| `test_upstream_deletion_occlusion_and_pdf` | 이상적인 이진 판서에서 원본 CC 분석/그룹화/삭제 분할, 두 프레임 가림 후 판서 보존, 지워진 내용 보존, PDF 이미지·타임스탬프 |
| `test_short_erase_rewrite_preserves_both_conflicting_shapes` | 최소 길이로 원본이 분리하지 못한 6샘플 충돌을 보완하여 양쪽 판서를 보존 |
| `test_pdf_omits_blank_and_exact_duplicate_preserves_ratio` | 빈/완전 중복 페이지 제외, crop 이후 판서 픽셀 수 보존, 이미지 종횡비 배치 |
| `test_empty_binary_sequence_returns_clear_no_content_error` | 빈 영상 마스크 처리 후 의미 있는 오류, 빈 PDF 미생성 |
| `test_metadata_sampling_and_limits` | 메타데이터, 1 FPS 샘플 인덱스/시간, 폭 축소, 길이/샘플 제한, 깨진 파일 |
| `test_no_checkpoint_never_uses_fallback` | 모델 없을 때 명시적 중단 |
| `test_invalid_sample_fps` (3개) | 0, 음수, 상한 초과 FPS 거절 |
| `test_upload_status_download_cleanup` | 접수/상태/다운로드/입력 정리, 모르는 작업, 경로 탐색 요청 거절, CORS preflight |
| `test_missing_model_invalid_upload_and_size` | 503/400/422/413, Content-Length와 스트리밍 누적 크기 제한, 거절된 입력 삭제 |
| `test_busy_and_error_state` | 단일 실행 슬롯, 동시 요청 409, 실패 전파, 미완료 다운로드 409, 실패 입력 정리 |
| `test_restart_and_expiration` | 재시작 시 미완료 작업 실패 처리·입력 정리, TTL 삭제 |

경고는 원본 SciPy measurements 네임스페이스와 matplotlib/pyparsing, ReportLab, Starlette/AnyIO의 deprecation 경고다.
현재 핀한 환경에서는 테스트가 통과한다. API TestClient를 네트워크가 완전히 격리된 sandbox에서 실행하면
스레드 이벤트 루프의 로컬 소켓 wakeup이 대기하는 현상을 관측했다. 로컬 네트워크 권한을 허용한 실행으로 해결했다.

추가 확인: `pip check` 정상, `git diff --check` 정상, Python compileall 정상, `node --check frontend/app.js` 정상.
네이티브 빌드·대용량 영상·작업 산출물과 다른 모델 파일은 Git ignore 확인.
사용자의 후속 요청에 따라 제공된 V34 체크포인트 한 파일은 ignore 예외로 지정해 저장소에 포함한다.

## 산출물과 재현

실행 결과는 Git에 포함하지 않는 `benchmark-output/`에 남겼다.

- `final-24s/result.pdf`, `pages/`, `masks/`, `temporal/*.pickle`, `pipeline.log`, `report.json`
- `final-48s/result.pdf`, `report.json`, 분할 신호 이미지와 로그
- `cli-6s-preserved/`: 짧은 삭제 보완 후 실제 모델 실행 결과
- `web/browser-result.pdf`, `desktop.png`, `complete.png`, `mobile.png`, `browser-report.json`, 서버/작업 로그
- 커밋 가능한 요약 수치는 `docs/validation-results.json`에 보존

재현 명령은 README의 테스트 절을 따른다. 체크포인트가 없는 환경에서도 13개 자동 테스트는 실행 가능하지만,
실제 V34 CLI/브라우저 검증에는 사용자 제공 가중치가 필요하다.

## 미검증 / 남은 한계

1. 실제 강의·실제 교수자 가림·다양한 칠판 색·카메라 움직임·미세한 수식 획의 정확도는 정량 검증하지 않았다.
2. MP4 테스트를 수행했다. MOV/MKV/AVI/WebM은 OpenCV 빌드의 디코더에 의존하며 각각의 실제 파일을 검증하지 않았다.
3. 30분·고해상도·많은 CC의 최대 메모리/CPU 부하, 디스크 가득 참, 강제 종료 중 디코딩은 부하 검증하지 않았다.
4. CUDA, cuda:N, Jetson ARM64 실행은 미검증. 설정/이식 안내만 제공한다.
5. 영상에서 한 번도 보이지 않거나 안정 조건을 충족하지 못한 판서, 시간 수명이 겹치는 충돌 그룹의 모든 원문 보존은 보장하지 않는다.
6. 완전 중복은 제거하지만 부분 중복/근사 중복은 남는다. VFR의 정밀 timestamp, 공개 서비스 인증/분산 작업/작업 취소는 구현하지 않았다.

## 후속 변경: PDF 경계 표시 개선

품질 피드백을 반영해 `smooth` 출력 스타일을 추가하고 기본값으로 지정했다. 기존 이진 결과를 3배 bicubic 보간해
표시 경계에 회색 계조를 추가한다. 모델·CC 추적·분할·중복 판정은 이진 결과를 계속 사용한다.
`binary` 옵션으로 기존 출력을 선택할 수 있다. 이미지 픽셀 수만 늘리는 처리이며 원본 해상도/모델 정확도를 높이지 않는다.

- 기존 실제 V34 24초 결과 두 페이지에 모두 적용해 전후 PDF를 생성하고 같은 크기로 렌더링해 확인했다.
- 모든 원본 픽셀이 3배 이미지의 대응 픽셀 중심에 그대로 존재하는지 두 페이지 모두 확인했다.
- 회귀 테스트에 작은 점, 1픽셀 획, 원 내부 여백 보존, PDF 배치 변환 일치, 저장 PNG와 삽입 이미지 일치를 추가했다.
- 새 출력 설정을 연결한 실제 V34 CLI도 생성한 6초 영상으로 실행했다. smooth PDF 2페이지 생성, 7.132초,
  프로세스 최대 RSS 534.66 MiB (`quality-improved/cli-smoke/report.json`).
- 테스트: **14 passed, 59 warnings, 2.92s**. 최초 테스트 비교는 PDF 이미지 리소스 식별자까지 비교해 실패했으며,
  실제 검증 대상인 배치 변환 행렬 비교로 수정했다.
- 산출물: `benchmark-output/quality-improved/{binary,smooth}.pdf`, 각 스타일의 `page_0001.png`/`page_0002.png`,
  동일 크기 PDF 렌더링 `binary-preview.png`/`smooth-preview.png`, 원본 픽셀 보존 검사 `comparison.json`.

시각적 개선은 경계 표시 수준에 그친다. 기존 영상이 448×256이므로 계단 모양 자체와 모델이 뭉개뜨린 `x`는 남는다.
1080p 이상 실제 원본에서 판서 영역이 차지하는 픽셀 크기와 이진화 결과를 비교하는 작업은 아직 수행하지 않았다.

## 후속 검증: 실제 AccessMath2015 1080p 프레임 1장

위의 고해상도 미검증 범위를 일부 해소하기 위해 GitHub 공개 데이터셋에서 실제 강의 키프레임과 정답 마스크를 내려받았다.
출처: [bhargavaurala/accessmath-icfhr2018](https://github.com/bhargavaurala/accessmath-icfhr2018),
commit `b0278437b977679cf57001d7765227ef9e99038f`,
`output/annotations/AccessMath2015_lecture_01/{keyframes,binary}/10423.png`.
실제 파일의 크기는 양쪽 모두 **1920×1080**이다. 업스케일한 합성 영상이 아니다.
외부 연구 코드를 가져와 실행하지 않았으며 이미지/정답/원본 출처 설명은 Git에서 제외한 benchmark-output에 보존했다.

현재 환경은 인터넷이 완전히 개방된 상태가 아니었다. YouTube와 데이터셋 원본 Box URL은 프록시 CONNECT HTTP 403을 반환했다.
GitHub raw는 접근 가능했다. 따라서 **이 검증은 실제 프레임 한 장의 추출·PDF 테스트이며 연속 영상·ST3D 검증이 아니다.**
한 프레임을 반복해 가짜 동영상으로 만들거나 실제 시간적 통합처럼 보고하지 않았다.

동일 프레임을 원본/720p/폭 448픽셀로 입력하고 각 조건을 독립 CPU 프로세스로 실행했다.
원본 1080p는 2.5MP보다 작아 원본 FCN 내부 축소가 발생하지 않는다. 정답 마스크는 추론에 사용하지 않고 평가에만 사용했다.
저해상도 출력은 평가와 동일 위치 시각 비교에 한해서 nearest-neighbor로 원본 크기에 투영했다.

| 추론 입력 | 정확 픽셀 Precision | Recall | F1 | 추론 시간 (로딩 제외) | 프로세스 최대 RSS |
|---|---:|---:|---:|---:|---:|
| 1920×1080 | 92.44% | 89.17% | **90.77%** | 11.169초 | 3058.38 MiB |
| 1280×720 | 77.04% | 88.14% | 82.22% | 4.958초 | 1580.33 MiB |
| 448×252 | 28.81% | 70.39% | 40.89% | 0.595초 | 526.25 MiB |

1080p 입력에서 수식 `3x=1`, `x+2y=3`, 부등호와 본문 획이 저해상도 입력보다 잘 보존됐다.
일부 희미한 글자의 획 손실과 보드 위 배경의 소량 오검출은 남았다.
F1은 흑백 판서 픽셀의 정확 겹침 점수이며 OCR 정확도·강의 전체 정확도가 아니다.
V34가 이 프레임을 학습에 사용했는지는 확인하지 못했으므로 독립 평가셋에 대한 일반화 성능으로 해석하지 않는다.

생성한 PDF 3개가 각각 한 페이지이며, 저장된 최종 PNG와 PDF에 실제 삽입된 이미지가 동일함을 확인했다.
고해상도 PDF도 이번에는 한 프레임만 정리한 결과로, 삭제 이벤트/가림 복원/중복 통합을 거친 결과가 아니다.
V1의 기본 `max_width=1280`은 유지했다. 원본 1080p를 보존하려면 `--max-width 1920` 또는 `BOARD_MAX_WIDTH=1920`을 지정한다.
전체 영상에서는 고해상도 CC 처리 메모리도 추가되므로 위 단일 프레임 메모리를 영상 전체 상한으로 보면 안 된다.

재현:

```bash
python scripts/evaluate_keyframe.py benchmark-output/real-hd-accessmath/source/10423.png \
  --ground-truth benchmark-output/real-hd-accessmath/source/10423-gt.png \
  --width 1920 --output-dir /tmp/accessmath-1080p-new
```

출처·SHA-256·해상도별 수치: `docs/real-hd-keyframe-results.json`.
다운로드 및 결과 묶음: `benchmark-output/real-hd-accessmath.zip`.
비교 그림: `benchmark-output/real-hd-accessmath/comparison-crop.png`.
입력/원본 V34 이진 출력/PDF 직전 이미지/PDF는 `width-1920/`에 있다.
