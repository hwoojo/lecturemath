# FCN-LectureNet / ST3D 분석

분석 기준: [공식 저장소](https://github.com/kdavila/lecturemath), 커밋 `245d58a20393166addacb9ece75326ef3455b16f`.
현재 저장소 origin은 `hwoojo/lecturemath`이며 공식 원격 HEAD와 커밋이 같음을 `git ls-remote`로 확인했다.
원본 README: [ACCESS2021_release/README.md](../ACCESS2021_release/README.md).
MIT 라이선스와 Kenny Davila / Fei Xu 저작권을 원본 그대로 유지한다.

## 공통 실행 조건

원본 명령은 `python <script> configs/FCN_LectureNet.conf [options]`이다.
`ConsoleUIProcess`가 VIDEO_DATABASE_PATH의 LectureMath XML을 읽고 VIDEO_FILES_PATH에 상대 비디오 경로를 붙인다.
`-l <lecture title>` 또는 `-d <dataset>` 선택이 가능하다. 전체 데이터셋 영상은 이 저장소에 포함되지 않았다.
XML의 output_temporal 디렉터리와 설정 OUTPUT_PATH 아래에 `<prefix><lecture.id>.dat` 형태의 Python pickle을 저장한다.
**확장자는 .dat이지만 모델 가중치와 중간 pickle은 서로 다른 형식이다.**
시간은 밀리초, frame_indices는 원본 영상의 프레임 번호, ST3D의 interval은 샘플 배열 내의 인덱스다.
V1 timestamp는 프레임 번호/FPS 기반이다. 가변 프레임률(VFR) 영상의 정확한 presentation timestamp 정렬은 검증하지 않았다.

| 단계 / 실제 파일명 | 역할과 실행 조건 | 입력 | 출력 (설정 기본 prefix) |
|---|---|---|---|
| `pre_ST3D_v3.0_01_binarize.py` | 모델 생성·가중치 로딩, 기본 1 FPS 추출. 원본은 CUDA 기본값 True이며 torch.load의 CPU map_location이 없다. | XML, 비디오, 설정, 모델 state_dict | `tempo_binary_`: `(frame_times, frame_indices, compressed_frames)`; 각 마스크는 PNG 인코딩, 검정 바탕/흰 판서 |
| `pre_ST3D_v3.0_02_cc_analaysis.py` | 프레임마다 CC를 추출하고 공간적 recall/precision으로 대응시켜 안정성과 지속 시간을 추적. **실제 파일명은 analaysis 오타가 있다.** | 01 tuple | `tempo_stability_`: `(times, indices, CCStabilityEstimator)` |
| `pre_ST3D_v3.0_03_cc_grouping.py` | 긴 간격 CC 분리, 안정 CC 선택, 공간·시간 겹침 그룹화, 충돌 계산, 시간 구간별 그룹 영상 합성 | 02 tuple | `tempo_bin_reconstructed_`: 안정 CC로 재구성한 압축 마스크; `tempo_cc_conflicts_`: `(group_ages, conflicts)`; `tempo_cc_ST3D_`: `SpaceTimeStruct` |
| `pre_ST3D_v3.0_04_vid_segmentation.py` | method 1 SUMS, 2 CC conflicts, **3 deletion events**. 기본 설정은 3. 그룹 경계 상자 면적을 영상 면적으로 정규화해 추가/삭제 신호와 누적 삭제 신호를 계산하고 피크 기반 재귀 분할 | 3일 때 03의 세 출력 모두; 2일 때 reconstructed/conflicts; 1일 때 reconstructed | `tempo_intervals_`: inclusive 샘플 인덱스 `(start,end)` 목록. 원본은 debug_mode=True 고정으로 분할 신호 PNG도 저장 |
| `pre_ST3D_v3.0_05_generate_summary.py` | 구간에 존재한 그룹들을 모아 키프레임 합성. 충돌하면 최근 그룹을 선택. 단순 원본 프레임 캡처가 아님 | ST3D, intervals, XML 메타데이터 | `tempo_segments_`: `((summary_indices, summary_times, keyframes),)`; summaries 아래 PNG/XML/GUI 정보 |

## 주요 의존성

- 01: torch, torchvision, Pillow, NumPy, OpenCV. 모델은 3개 분기(배경 추정, 텍스트 마스크, 이진화) 구조다.
- 02/03/05: SciPy labeling, NumPy, OpenCV, 원본 `accessmath_lib.c` 네이티브 라이브러리, AM_CommonTools의 ConnectedComponent/IntervalIndex.
- 04: matplotlib와 scikit-learn을 모듈 import 단계부터 요구한다. deletion events에서도 설치해야 한다.
- 학습·평가·GUI의 pygame, shapely, munkres 등은 V1의 실행 경로에서 사용하지 않으므로 추가하지 않았다.
- 저장소에 포함된 `accessmath_lib.so`는 Linux에서 그대로 사용할 수 없는 바이너리다. C 소스로 호스트 아키텍처에 맞게 새로 빌드한다.

## V1에서 재사용한 범위

`boardnotes/model.py`는 원본 `FCN_LectureNet.CreateFromConfig`와 `binarize`를 사용한다.
원본 worker처럼 RGB 변환 후 추론하고 결과를 반전해 ST3D에 넘긴다. 모델을 재학습하지 않는다.
`boardnotes/video.py`는 일반 파일에 대해 메타데이터 검사와 순차 OpenCV 디코딩을 수행한다.
샘플 간 seek 대신 grab/retrieve를 사용하며 0초부터 샘플링한다. 기본 폭 상한 1280은 CPU 메모리 절약을 위한 V1 선택이다.
원본 모델도 2.5 MP 초과 이미지를 내부적으로 절반씩 축소하므로 해상도 설정은 미세한 획의 가독성에 영향을 줄 수 있다.

`boardnotes/temporal.py`는 02, 03, 04의 **process_input 원본 함수를 그대로 호출**한다.
ConsoleUIProcess가 기대하는 설정·이미지 출력 경로·lecture title만 SimpleNamespace로 제공한다.
데이터셋 XML/가짜 데이터셋을 새로 생성하지 않고 입출력 tuple을 직접 연결한다.
05는 `KeyframeExtractor.GenerateFromST3DForIntervals`를 그대로 사용하고 데이터셋 전용 XML/GUI exporter를 PDF 모듈로 대체한다.
설정의 segmentation method는 항상 3으로 지정한다. 원본 04 분할 후, 원본 03의 충돌 그래프에서
화소가 겹치고 수명이 완전히 분리된 안정 그룹들이 같은 구간에 남았을 때만 그 사이를 추가 분할한다.
이는 원본 05의 최근 그룹 우선 선택으로 짧은 삭제·다시쓰기의 이전 내용이 누락되는 문제를 방지한다.
원본 CC 추적이나 삭제 신호 알고리즘을 재작성하지 않으며, report에 원본 deletion_intervals와 추가 conflict_preservation_splits를 함께 기록한다.
판서 수명/최소 관찰 횟수/시간 창/최소 분할 길이는
1 FPS 기본값에 샘플링 FPS를 곱해 시간 단위를 가능한 유지한다. 임계값과 threshold 세부 조정은 연구 config 사본으로 가능하다.

## 원본에 가한 최소 수정

1. labeler, binarizer, adaptive_equalizer의 C 라이브러리 경로를 CWD 상대 Windows 바이너리에서 소스 위치 기반 `accessmath_lib.native.so`로 변경. `ACCESSMATH_LIB` 환경변수로 별도 경로도 지원한다.
2. keyframe_extractor의 제거된 `np.bool` 두 곳을 `np.bool_`로 변경. 겹치는 CC 충돌 분기에서도 최신 NumPy 실행을 허용한다.
3. 모델 유틸의 `batch_img.cuda(0)`를 모델 파라미터와 같은 장치로 이동하도록 변경하여 `cuda:N`을 지원한다.
4. 어댑터가 `map_location='cpu'`, `weights_only=True`, strict state_dict 로딩과 eval을 적용한다. 원본 01의 GPU 기본값/로딩 방식은 어댑터에서 사용하지 않는다.

네트워크: 2026-10-27 공식 Dropbox V34 링크의 직접 다운로드는 HTTP 403으로 실패했다.
응답은 `CONNECT tunnel failed, response 403`, `server: envoy`였으며, 환경의 제한된 허용 호스트 목록에 Dropbox가 없다.
이 환경에서는 프록시 정책에서 차단된 것으로 관측되며 Dropbox 원서버의 허용 여부는 확인하지 못했다.
우회 다운로드는 하지 않았다. 이후 사용자가 제공한 4개 분할 파일을 연결하여 실제 가중치를 확보했다.
단일 파일 첨부는 executor transfer limit 32 MiB를 초과해 받을 수 없었다.

## 품질과 자원 한계

- 고정된 카메라/판서 좌표에 맞는 파이프라인이다. 카메라 이동·줌·슬라이드 전환에 대한 정합이나 모델을 추가하지 않았다.
- 다른 프레임에서 충분히 관측된 안정 CC는 일시적 가림을 보완한다. 끝까지 가려진 내용은 복구하지 못한다.
- 원본은 기본 최소 20픽셀 CC와 3회 이상 관찰된 안정 CC를 사용한다. 작거나 빠르게 쓰고 지운 획, 부분 가림으로 형태가 크게 달라진 수식은 누락될 수 있다.
- 삭제 기반 분할의 최소 길이와 삭제 면적 임계값 때문에 매우 짧은 장면/작은 지움이 항상 분리되지는 않는다.
  수명이 분리된 안정 그룹의 충돌에는 추가 분할을 적용했다. 수명이 겹치는 그룹이나 같은 그룹 내부의
  변화는 여전히 원본 keyframe 생성기의 최근 그룹 우선 선택에 따라 내용이 누락될 수 있다.
- 전체 프레임 압축 목록을 유지하고 CC 처리에서 모든 마스크를 한 번 해제한다. 메모리는 길이·해상도·CC 복잡도에 따라 증가한다.
  기본 길이/샘플 상한은 자원 검사를 대체하지 않으며, 30분 고해상도 실제 영상의 메모리는 아직 검증하지 않았다.
- PDF는 빈 페이지/화소까지 동일한 페이지를 제거한다. 부분 중복은 문맥 보존을 위해 남는다. 근사 중복 제거는 수식 손실 위험 때문에 넣지 않았다.
- PDF timestamp는 샘플 구간을 나타내며 각 획이 동시에 존재했던 한 순간을 뜻하지 않는다.
