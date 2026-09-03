# project.md — TransNote: OMR 추론 백엔드(RunPod Serverless/Docker)

> 세부 문서: 저장소 구조/명령어 [`CLAUDE.md`](CLAUDE.md). 모델 학습 코드·히스토리·정확도
> 근거 및 모바일 온디바이스 양자화(2026-09-03 이관)는 별도 저장소
> [Model_TransNote](https://github.com/braeden-hue/Model_TransNote) 참고. 이 문서는
> 요약/현황판 역할만 한다.

## 지난 공식 프로젝트 (완료, 2026-08-17 제출)

삼성생명 라이프놀로지랩 3기 산학 프로젝트로 시작 — 악보 이미지를 촬영/업로드하면 자체 학습한
OMR 모델이 음표를 인식해 사용자 정의 표기법으로 변환, 화면 가상 피아노 또는 전자 피아노로
연주 연습하는 웹 앱. 데모 웹페이지(`trans-note.vercel.app`) + 포스터 + 상세기획안 PPT + 영상
제출 완료. 당시 기준 OMR 모델은 r15 체크포인트, 뉴에이지 실사 테스트 기준 약 96% 인식 정확도
(상세 근거는 Model_TransNote 참고).

## 현재 (2026-09-03 이후)

공식 제출은 끝났지만 프로젝트는 여기서 계속된다. 웹 프론트엔드는 별도 프로젝트(myweb, 새
Vercel 배포)로 완전히 옮겨갔고, 이 저장소는 그 프론트엔드가 호출하는 **RunPod Docker GPU
추론 백엔드**만 유지한다. 모바일 온디바이스 양자화(모델 학습 저장소와 합치는 게 자연스럽다고
판단해 2026-09-03에 이관)는 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote)의
`train/tools/`에서 계속한다.

**타겟**: 악보를 처음 접하거나 읽기 어려운 사람
**형태**: RunPod Serverless(Docker, GPU) 추론 백엔드
**진행 상황**: 아래 "현재 상태" 참고. 모바일 양자화 진행 상황은 Model_TransNote의
`train/tools/QUANTIZATION_MOBILE.md`.

---

## 현재 상태

- **RunPod Docker 파이프라인 유지 중**: `.github/workflows/build-runpod-image.yml`이
  `runpod_serverless/`·`train/*.py`·`server/token_to_notes.py` 변경마다 이미지 자동 빌드,
  Docker Hub → RunPod Custom Image로 배포. myweb 프론트엔드가 이 엔드포인트를 그대로 호출.
- **웹 프론트엔드(`webpage/`+`api/`+`server.py`) 저장소에서 제거(2026-08-24)** — myweb으로
  이전, `trans-note.vercel.app`는 마지막 배포본만 남기고 더 이상 갱신 안 함.
- **모바일 양자화는 Model_TransNote로 이관(2026-09-03)**: `train/export_tflite.py`를 r15
  체크포인트에 맞게 고치는 것부터 시작해 KV캐시 고정 버퍼·버킷팅까지 진행했던 작업 전체가
  이제 그쪽 `train/tools/`에 있다 — 상세는 그쪽 README·`QUANTIZATION_MOBILE.md` 참고.
- **OMR 모델**: `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt` 그대로 채택 유지,
  주 타겟 장르인 뉴에이지 실사 테스트 기준 약 96% 인식 정확도(상세 근거는 Model_TransNote
  참고).

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| 추론 서버 | RunPod Serverless(Docker, GPU) — `runpod_serverless/handler.py` |
| 배포 | Docker Hub + `.github/workflows/build-runpod-image.yml` 자동 빌드 |
| OMR 모델 | PyTorch(CNN 인코더 + Transformer 디코더) — 학습 코드/히스토리는 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote) |

---

## 과거 결정 (요약)

- **2026-08-05**: r15를 프로덕션 체크포인트로 확정, r16/r17 시도 후 기각(상세는 Model_TransNote).
- **2026-08-09**: Flutter 앱 트랙 전면 폐기, 웹 단일 구조로 전환.
- **2026-08-12**: 저장소를 서버/배포 중심으로 정리 — 학습 코드·문서·평가 스크립트
  (`train/experiments/`, `train/docs/`, `test/`)를
  [Model_TransNote](https://github.com/braeden-hue/Model_TransNote)로 이관하고 이 저장소에서
  제거. Flutter 시절 기획 문서(`docs/archive/`, `docs/PLAN_booth_companion_page.md`)·옛
  서브에이전트 메모리(`.claude/agent-memory/`)도 함께 제거 — 단 `docs/music-notation-rule-designer.md`는
  당시엔 `webpage/js/notation.js` 등 코드가 실제로 참조하는 활성 문서라 유지했음(2026-08-24에
  webpage 제거와 함께 결국 Model_TransNote로 이전, 아래 참고).
- **2026-08-17**: 삼성생명 라이프놀로지랩 3기 공식 제출 완료(데모+포스터+PPT+영상).
- **2026-08-24**: 공식 제출 이후 방향 전환 — 웹 프론트엔드(`webpage/`·`api/`·`vercel.json`·
  `server/server.py`)를 저장소에서 제거(myweb으로 완전 이전, `trans-note.vercel.app`는
  방치). `docs/music-notation-rule-designer.md`를 Model_TransNote(`train/docs/`)로 이전.
  `FEATURES.md`(webpage 전용 기능 문서) 삭제.
- **2026-09-03**: `train/tools/`(모바일 양자화 전체)·`train/QUANTIZATION_MOBILE.md`를
  Model_TransNote로 이관 — 학습·평가·양자화를 한 저장소에 모으는 역할 분담 정리(팀원 피드백
  반영). 이 저장소는 이제 RunPod Serverless/Docker 프로덕션 배포만 담당한다.
