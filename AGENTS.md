# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Working Guidelines & Tri-Agent Protocol

Full rules live in `C:\Users\kyutae\agent-management\` (`WORKING_GUIDELINES.md`,
`tri_agent_command_protocol.md`) — read those before starting work, don't rely on a copy here.
Project-specific facts and decisions for TransNote are the SSOT at
`agent-management/projects/TransNote/{PROJECT_TRUTH.md,DECISIONS.md}`, not this file or any
README — cite or update those instead of restating numbers here.

## Project Overview

TransNote — 악보 이미지를 촬영/업로드하면 자체 학습한 OMR(광학 악보 인식) 모델이 음표를 인식해
**사용자 정의 표기법**으로 변환해주는 서비스의 **추론 백엔드** 저장소.

**2026-08-24부터 이 저장소의 역할이 바뀌었다.** 예전엔 `server.py`(FastAPI)+`webpage/`(정적
프론트)+`api/`(Vercel 프록시)로 웹앱까지 이 저장소 하나에서 서빙했으나(`trans-note.vercel.app`),
웹 프론트엔드가 별도 프로젝트(myweb, 새 Vercel 배포)로 완전히 이전되면서 그 부분을 전부
제거했다. `trans-note.vercel.app`는 마지막 배포본만 남아있고 더 이상 갱신하지 않는다.

**이 저장소가 지금 담당하는 것**: **RunPod Serverless Docker 이미지 파이프라인** —
myweb(외부 프론트엔드)이 호출하는 GPU 추론 백엔드를 계속 유지·배포한다.

모델 학습 코드·히스토리·정확도 근거는 별도 저장소
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote) 참고(2026-08-12 분리,
`train/docs/`·`train/experiments/`·`test/`를 이 저장소에서 제거하고 이관 — 단
`music-notation-rule-designer.md`는 당시엔 webpage 코드가 참조해서 남겨뒀다가 2026-08-24
webpage 제거와 함께 결국 Model_TransNote로 옮겼다). `train/` 최상위 `.py`/`tokenizer258.json`은
RunPod Docker 이미지가 실제로 빌드 시 복사해가는 런타임 의존 파일이라 이 저장소에도 그대로
남아있다(아래 "train/ 내부 구조" 참고) — 지우면 배포가 깨진다.

**2026-08-09 이전에는 Flutter 앱 + 자체 C++ TFLite 엔진 구조였으나 폐기됨** — 다만 폐기된 건
"Flutter+C++로 만든 그 구현체"이지 "온디바이스 추론"이라는 목표 자체가 아니다. 이후 웹(FastAPI)
단일 구조로 재구성했었고, 그마저 2026-08-24에 이 저장소에서 분리했다. Flutter/C++ 시절 코드
잔재는 저장소에 없다. **모바일 온디바이스 양자화(`export_tflite.py` 기반)는 2026-09-03에
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote)로 옮겼다** — 이 저장소에는
더 이상 없음, 그쪽 `train/tools/`·`train/tools/QUANTIZATION_MOBILE.md` 참고.

## Directory Layout

```
runpod_serverless/    # RunPod Serverless GPU 추론 워커(Docker) — .github/workflows가 자동 빌드
train/                # 추론 런타임 4개(model/dataset/inference/real_texture_augment) — 평면 유지
server/token_to_notes.py  # Docker 이미지가 복사해가는 런타임 의존 파일 (server/의 나머지는 삭제됨)
realImage/            # 실사 촬영 이미지(로컬 전용, .gitignore로 git 미포함)
secrets/              # API 키 등(.gitignore로 git 미포함, 절대 커밋 금지)
```

**중요**: `Dockerfile`의 `COPY train/*.py`와 CI 트리거 경로 `train/*.py`는 **평면 glob이라
하위 디렉토리를 안 가져간다.** 그래서 추론에 필요한 모듈은 반드시 `train/` 최상위에 둬야 한다 —
런타임 모듈을 하위 디렉토리로 옮기면 배포가 조용히 깨진다.

## Build & Run Commands

### 추론 동작 확인 (`train/`)
```bash
python train/inference.py --seq2seq <ckpt.pt> --tokenizer train/tokenizer258.json --analyze <dir>
```
학습(재학습·데이터 생성·라운드별 정확도)은 이 저장소 범위 밖 —
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote)에서 진행한다.
모바일 export/검증(`export_tflite.py` 등)도 2026-09-03부터 그 저장소의 `train/tools/`에서
진행한다 — 상세는 이 문서 상단 "Project Overview" 참고.

## Architecture

### 추론 경로 (실제 서비스, RunPod)
```
(외부 프론트엔드, myweb) → RunPod Serverless 엔드포인트
  → runpod_serverless/handler.py
      → train/inference.py: run_image()
          1. dataset.py: detect_staffs() — OpenCV 고전 알고리즘으로 오선 검출(학습된 모델 아님)
          2. dataset.py: extract_staff_canvas() / extract_system_canvas() — 오선 크롭·정규화
          3. model.py: OmrSeq2Seq — CNN 인코더 + Transformer 디코더, autoregressive 토큰 생성
      → token_to_notes.py: tokens_to_score() — 토큰 시퀀스 → 커스텀 표기법 JSON
  ← JSON 응답 (SVG 렌더링 등 시각화는 프론트엔드인 myweb 쪽 코드가 담당, 이 저장소 범위 밖)
```

**SegNet은 현재 추론 경로에 없다.** `inference.py`에 segnet 관련 코드가 전혀 없음 — 오선 검출은
순수 OpenCV(`detect_staffs()`)로만 한다. SegNet은 삭제된 옛 C++ 모바일 엔진 전용이었고
`train/checkpoints_legacy/segnet_best.pt`에 참고용으로만 남아있다(재도입 논의 시에만 필요) —
단, Model_TransNote로 옮긴 `export_tflite.py`는 이 SegNet도 같이 export하는 옛 스크립트라
모바일 파이프라인에서 SegNet을 다시 쓸지는 아직 미결정(그쪽 `QUANTIZATION_MOBILE.md` TODO 참고).

### 프로덕션 체크포인트
```
train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt   # 유일한 채택 체크포인트(로컬/gitignore, RunPod 빌드는 GitHub Release에서 받음)
train/tokenizer258.json                                    # DeepScore 토큰 vocabulary
```
아키텍처(in_ch/backbone 깊이/pool_h)는 `model.py`의 `infer_arch_from_state_dict()`가 체크포인트
텐서 shape에서 자동 역산 — 별도 config 파일 불필요(r15는 `in_ch=2`, CoordConv). r15 채택
근거·r16/r17/r18 기각 이유 등 학습 히스토리는
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote)의
`train/docs/TRAINING_REPORT.md` 참고(이 저장소엔 없음).

**주의**: 이 저장소의 export/학습 보조 스크립트를 건드릴 때 `OmrSeq2Seq(vocab_size=...)`를
생성자 기본값으로만 호출하면 r15처럼 논디폴트 아키텍처인 체크포인트에서 shape mismatch가
난다 — 반드시 `infer_arch_from_state_dict()`로 역산한 값을 넘길 것(2026-08-24,
`export_tflite.py`에서 실제로 겪은 버그).

### `train/` 내부 구조 (2026-09-03 정리)
| 위치 | 역할 |
|---|---|
| `train/*.py` (4개) | **Docker가 `COPY train/*.py`로 가져가는 추론 런타임** — `model.py`(아키텍처), `dataset.py`(전처리·오선검출), `inference.py`(디코딩·후처리), `real_texture_augment.py`(dataset이 import). 여기에 파일을 추가하면 Docker 이미지에 그대로 들어간다 |
| `train/checkpoints/` | r15(채택) — `.gitignore` 처리, 로컬에만 존재 |
| `train/checkpoints_legacy/` | 옛 Flutter/C++ 엔진 시절 체크포인트(segnet 포함) — 참고용, `.gitignore` 처리 |
| `train/real_texture_bank/` | `real_texture_augment.py`가 참조하는 노이즈 텍스처 png(304KB) |

`train/tools/`(TFLite export/추론/벤치마크)와 `train/QUANTIZATION_MOBILE.md`는 2026-09-03에
Model_TransNote로 이동했다 — 이 저장소엔 더 이상 없음.

학습 코드 전반(`train/experiments/`, `train/docs/`, 평가 스크립트 `test/`)은 2026-08-12에
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote)로 이관했고, 2026-08-26에
그때 이 저장소에 중복으로 남아있던 `generate_scores.py`·`train.py`·`mscz_to_tokens.py`·
`render_custom_notation.py`·`render_sample10_comparison.py`·`dump_canvas.py`(합 3,698줄)도
제거했다(내용이 Model_TransNote 사본과 동일한 걸 diff로 확인 — 그쪽 `train/core/`,
`train/data_gen/`, `train/visualize/`에 있음). 학습/재현 작업은 그 저장소에서 진행할 것.

## Platform Notes

- Python 전용 저장소(RunPod Docker) — Flutter/Dart/Android/iOS 네이티브 툴체인은 안 씀.
- 프론트엔드(myweb)의 Web MIDI/카메라 관련 보안 컨텍스트 제약은 이 저장소 범위 밖 —
  myweb 쪽 문서 참고.

## Known Gaps / Follow-ups

- 2026-08-12: `test/`, `train/experiments/`, `train/docs/`, `.claude/agent-memory/`,
  `docs/archive/`·`docs/PLAN_booth_companion_page.md`를 저장소에서 제거(학습 히스토리는
  Model_TransNote로 이관, 나머지는 Flutter 시절 등 stale 문서).
- 2026-08-24: 웹 프론트엔드(`webpage/`·`api/`·`vercel.json`·`server/server.py`·
  `server/requirements.txt`)를 저장소에서 제거, myweb으로 완전 이전.
  `docs/music-notation-rule-designer.md`를 Model_TransNote(`train/docs/`)로 이전.
  `FEATURES.md`(webpage 전용 기능 문서) 삭제. `server/token_to_notes.py`는 Docker 런타임
  의존성이라 그대로 유지.
- 2026-09-03: `train/tools/`(export_tflite.py, tflite_infer.py, benchmark.py,
  onnx_wrappers.py, eval_exactpicture.py, EXPORT_NOTES.md)와 `train/QUANTIZATION_MOBILE.md`를
  Model_TransNote로 이동 — 학습·평가·양자화를 한 저장소에 모으는 역할 분담 정리(팀원 피드백
  반영). 이 저장소는 이제 RunPod Serverless/Docker 프로덕션 배포만 담당한다.
