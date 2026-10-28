# V1.5 병합 검토

2026-10-28. 원격 `work` 기준 `e1944c4cfc07c1290e3d8cc8db09ed7d98ef5269`에
로컬 `245d58a` 기준의 미커밋 V1.5 변경을 별도 worktree에서 3-way 적용했다.
사전 검토 당시 원래 작업 위치 `/workspace/lecturemath`의 파일/인덱스/HEAD는 유지했다.
원격 추적 `origin/work`만 fetch로 추가했으며, 그 시점에는 커밋/실제 work 병합/push를 수행하지 않았다.

## 판정

병합 가능. README 및 신규 실험/문서 경로는 충돌 없이 적용됐다.
유일한 텍스트 충돌은 양쪽이 새로 만든 `.gitignore`다.
원격의 전체 ignore 목록/V34 예외를 보존하고 로컬의 `*.py[cod]`를 추가해 해결했다.
V1.5는 독립 모듈이므로 V1의 boardnotes 및 원본 모델 변경과 직접 충돌하지 않는다.

내용 통합은 다음과 같이 시험 반영했다.

- `pyproject.toml`의 pytest testpaths에 `experiments/v1_5/tests`를 추가해 기본 명령에서 함께 검증.
- V1.5 문서 두 곳에 최초 조사 기준이 오래된 체크아웃임을 명시하고, 실제 V1 래퍼/문서/V34 존재를 정정.
- V1 소스 자체를 수정하거나 가중치를 변경하지 않았다.

## 검증

- 기존 V1 14개 + V1.5 7개: **21 passed**, 9.40초. Deprecation warning만 발생.
- 원격 스크립트로 native library 빌드 성공.
- `git diff --cached --check` 통과, 미해결 충돌 없음.
- 실제 V34 파일 SHA256 확인:
  `d3721339db6cc2d2de2a29816bfcaca38f2d4c4bc8a1eb75a7402a194604b144`.
- 실제 V34 + 합성 160×96 정지 프레임에서 새 CLI의 B0~B4 실행: PNG 14개, skip 0.
- 같은 입력에서 기존 `boardnotes.model.LectureNet.extract()`의 극성을 맞춘 결과와
  새 B0는 **차이 픽셀 0개**. GT를 제공하지 않아 F1 생성 없음.
- 실제 강의 품질 향상/고해상도 성능/전체 영상 재실험은 이번 병합 검토에서 수행하지 않았다.

환경에 V1 웹/테스트 의존성이 없어 `/tmp/lecturemath-merge-testdeps`에만 설치했다.
기존 Torch/OpenCV 및 가상환경 패키지는 변경하지 않았다.

```sh
# 시험 병합 worktree에서
bash scripts/build_native.sh
PYTHONPATH=/tmp/lecturemath-merge-testdeps /workspace/lecturemath-env/bin/python -m pytest -q
```

호환성 실행 결과: `/tmp/lecturemath-merge-real-check/compatibility.json`,
전체 CLI 출력: `/tmp/lecturemath-merge-real-check/out/`.
이 파일들은 현재 환경의 검증 산출물이며 Git에는 포함하지 않았다.

시험 병합 작업 위치: `/workspace/lecturemath-merge-preview` (detached HEAD e1944c4).
원래 미커밋 변경의 binary patch: `/tmp/lecturemath-v15-merge.patch`.

## 실제 work 반영 (후속 사용자 요청)

사용자 승인 후 로컬 `work`를 `e1944c4`로 fast-forward하고, 위에서 검증한
V1.5 변경과 충돌 해결을 반영했다. V1 코드/체크포인트는 원격 work 상태를 그대로 유지한다.
원래 미커밋 변경은 통합 전에 stash로 백업했다. 본 변경은 로컬 커밋으로 보존하며 push는 하지 않는다.
