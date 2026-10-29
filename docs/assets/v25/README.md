# V2.5 실험 근거

- `comparison.pdf`: 기존6/추가12 수식의 V34 입력 비교, 모델 기술명 사용. `comparison.json/csv`에는 RGB/V34 전 예측과 행 분할 후보가 있다.
- `crops/`, `references/`, `rendered/`: 원본 crop, 고정 정답 렌더링, 모델 예측 렌더링. 실패/미지원은 JSON에 기록한다.
- `text-results.json`: 추가10개 텍스트 영역의 실제 OCR 예측과 CER.
- `automatic-results.json`: 추가3프레임 자동 영역/인식 결과. 부분 ROI 주석으로 검출 정확도를 계산하지 않았다.
- `real-video/result.pdf`: MIT 실제 강의 clip의 통합 결과, 출처 표지 포함. `real-video/page_*/result.json`은 이미지 경로를 상대 경로로 보존해 재렌더링할 수 있다.
- `video-preview.png`, `video-provenance.json`, `video-end-to-end-report.json`, `video-discovery.json`: 실제 영상/조사/실행 근거.
- `model-provenance.json`: 공식 모델 URL/체크포인트 SHA256. 가중치는 포함하지 않는다.

AccessMath 프레임의 원출처/해시는 `experiments/v25/additional_regions.json` 및 기존 V2 provenance에 있다.
MIT 파생자료의 원출처는 MIT OpenCourseWare/Prof. Gilbert Strang의 18.06 강의다.
OCW 페이지는 CC BY-NC-SA 4.0, 연결된 Archive 메타데이터는 CC BY-NC-SA 3.0 US를 표시한다.
해당 출처와 조건을 보존한다. 이 저장소 코드의 MIT 라이선스가 외부 자료의 라이선스를 대체하지 않는다.

일부 실행 로그/프로파일 JSON에는 당시 환경의 절대경로가 출처 정보로 남는다.
원본 clip/모델/캐시는 `benchmark-output`, `models/v2`에만 두며 이 묶음에 포함하지 않는다.
공유 JSON의 review flag는 최종 정책에 맞게 전파했다는 metadata를 남겼으며,
실제 모델 예측/측정 시간은 변경하지 않았다. 렌더링 성공은 인식 정확도의 증거가 아니다.
