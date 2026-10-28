# V1.5 CPU binarization experiments

기존 V1을 수정하지 않는 독립 실험 모듈. 학습/다운로드 없음.
실행 위치는 저장소 루트이며 NumPy/OpenCV가 필요하다.
체크포인트 추론만 기존 Torch/torchvision/Pillow가 추가로 필요하다.

```sh
python -m experiments.v1_5.cli /data/frames --output /tmp/v15 --polarity dark --methods B1 B2 --limit 3
python -m experiments.v1_5.cli /data/frames --output /tmp/v15_cache --polarity dark --baseline-dir /data/binary --methods B0 B1 B2 B3
python -m experiments.v1_5.cli /data/frames --output /tmp/v15_cpu --polarity dark --checkpoint /models/V34.dat --config ACCESS2021_release/configs/FCN_LectureNet.conf --cache-dir /tmp/v34_cache
python -m experiments.v1_5.cli --help
python -m unittest discover -s experiments/v1_5/tests -v
python -m experiments.v1_5.synthetic_demo
```

`--polarity light`는 밝은 글씨. `--baseline-polarity light`는 V1 영상 worker의 흰 전경 PNG.
마스크와 입력 파일의 stem/해상도/좌표를 일치시킨다. `--resolution 1080p`는 검증만 하며 축소하지 않는다.
`--roi x,y,w,h`는 평가 영역, `--crop x,y,w,h --zoom 4`는 nearest-neighbor 확대 영역.
`--gt-dir`가 없으면 F1을 생성하지 않는다. 누락된 B0/B3/B4 의존성은 skip으로 기록한다.

상세 설명: [구현·명령](../../docs/v1_5.md), [결과·한계](../../docs/v1_5_results.md).
저장된 `artifacts/synthetic/`는 합성 기능 예제이며 **B0가 실제 LectureNet 결과가 아니다**.

## 실제 프레임 평가와 V2 인수인계

```sh
python -m experiments.v1_5.real_benchmark
python -m experiments.v1_5.real_report --ocr --pdf
```

첫 명령은 공개 출처의 실제 프레임/GT 3쌍만 내려받고 1080p/720p를 비교한다.
두 번째 명령의 선택 옵션은 기존 tesseract(eng)/reportlab이 준비된 환경에서 사용한다.
원본 RGB와 B0/B3를 V2의 비교 입력으로 보존한다. B3를 OCR 개선으로 확정한 것은 아니다.
자세한 상태와 미완료 항목은 [HANDOFF](../../docs/HANDOFF.md)를 참고한다.
