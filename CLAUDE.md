# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Overview

TransNote — 악보 이미지를 촬영/업로드하면 자체 학습한 OMR(광학 악보 인식) 모델이 음표를 인식해
**사용자 정의 표기법**으로 변환해주는 서비스의 **추론 백엔드** 저장소.

**2026-08-24부터 이 저장소의 역할이 바뀌었다.** 예전엔 `server.py`(FastAPI)+`webpage/`(정적
프론트)+`api/`(Vercel 프록시)로 웹앱까지 이 저장소 하나에서 서빙했으나(`trans-note.vercel.app`),
웹 프론트엔드가 별도 프로젝트(myweb, 새 Vercel 배포)로 완전히 이전되면서 그 부분을 전부
제거했다. `trans-note.vercel.app`는 마지막 배포본만 남아있고 더 이상 갱신하지 않는다.

**이 저장소가 지금 담당하는 것 두 가지**:
1. **RunPod Serverless Docker 이미지 파이프라인** — myweb(외부 프론트엔드)이 호출하는 GPU
   추론 백엔드를 계속 유지·배포한다.
2. **모바일 온디바이스 양자화** — 같은 r15 체크포인트를 양자화해 모바일에서 빠른 속도로
   추론하는 작업(`train/export_tflite.py`, 진행 상황은 `train/QUANTIZATION_MOBILE.md`).

모델 학습 코드·히스토리·정확도 근거는 별도 저장소
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote) 참고(2026-08-12 분리,
`train/docs/`·`train/experiments/`·`test/`를 이 저장소에서 제거하고 이관 — 단
`music-notation-rule-designer.md`는 당시엔 webpage 코드가 참조해서 남겨뒀다가 2026-08-24
webpage 제거와 함께 결국 Model_TransNote로 옮겼다). `train/` 최상위 `.py`/`tokenizer258.json`은
RunPod Docker 이미지가 실제로 빌드 시 복사해가는 런타임 의존 파일이라 이 저장소에도 그대로
남아있다(아래 "train/ 내부 구조" 참고) — 지우면 배포가 깨진다.

**2026-08-09 이전에는 Flutter 앱 + 자체 C++ TFLite 엔진 구조였으나 전면 폐기됨** — 이후 웹
단일 구조로 재구성했었고, 그마저 2026-08-24에 이 저장소에서 분리했다. Flutter 시절 잔재는
저장소에 없지만, `train/export_tflite.py`(모바일 export 스크립트)는 그 시절 유물을 다시
꺼내 쓰는 중이라는 점에 주의 — 지금 아키텍처(r15, CoordConv)에 맞게 일부 수정했다.

## Directory Layout

```
runpod_serverless/    # RunPod Serverless GPU 추론 워커(Docker) — .github/workflows가 자동 빌드
train/                # OMR 추론 런타임 코드(PyTorch) + 모바일 양자화 작업(export_tflite.py, QUANTIZATION_MOBILE.md)
server/token_to_notes.py  # Docker 이미지가 복사해가는 런타임 의존 파일 (server/의 나머지는 삭제됨)
realImage/            # 실사 촬영 이미지(로컬 전용, .gitignore로 git 미포함)
secrets/              # API 키 등(.gitignore로 git 미포함, 절대 커밋 금지)
```

## Build & Run Commands

### 추론 동작 확인 (`train/`)
```bash
python train/inference.py --seq2seq <ckpt.pt> --tokenizer train/tokenizer258.json --analyze <dir>
```
학습(재학습·데이터 생성·라운드별 정확도)은 이 저장소 범위 밖 —
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote)에서 진행한다.

### 모바일 export 시도
```bash
python train/export_tflite.py --segnet <legacy_segnet.pt> --seq2seq <ckpt.pt> \
    --tokenizer train/tokenizer258.json --out_dir train/tflite_export --version v1 --no_quantize
```
진행 상황·TODO는 `train/QUANTIZATION_MOBILE.md` 참고.

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
단, `export_tflite.py`는 이 SegNet도 같이 export하는 옛 스크립트라 모바일 파이프라인에서
SegNet을 다시 쓸지는 아직 미결정(`QUANTIZATION_MOBILE.md` TODO 참고).

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

### `train/` 내부 구조
| 위치 | 역할 |
|---|---|
| `train/*.py` (top-level, ~14개) | `runpod_serverless/Dockerfile`이 `COPY train/*.py`로 통째로 복사해가는 런타임 세트 — `dataset.py`/`model.py`/`inference.py`가 실제 추론 경로에서 쓰이고, 나머지(`train.py`, `generate_scores.py`, `export_tflite.py` 등)는 학습/모바일 export 코드지만 sibling import 안전을 위해 같이 유지 |
| `train/checkpoints/` | r15(채택) — `.gitignore` 처리, 로컬에만 존재 |
| `train/checkpoints_legacy/` | 옛 Flutter/C++ 엔진 시절 체크포인트(segnet 포함) — 참고용, `.gitignore` 처리 |
| `train/real_texture_bank/` | `real_texture_augment.py`가 참조하는 노이즈 텍스처 png(304KB) — 학습 전용 코드 의존성이지만 용량이 작아 유지 |
| `train/QUANTIZATION_MOBILE.md` | 모바일 양자화 작업 진행 로그 — 새 세션에서 이 작업 이어갈 때 먼저 읽을 것 |

학습 코드 전반(`train/experiments/`, `train/docs/`, 평가 스크립트 `test/`)은 2026-08-12에
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote)로 이관하고 이 저장소에서
제거했다 — 학습/재현 관련 작업은 그 저장소에서 진행할 것.

## Platform Notes

- Python 전용 저장소(RunPod Docker + 학습/export 스크립트) — Flutter/Dart/Android/iOS
  네이티브 툴체인은 아직 안 씀(모바일 export는 ONNX/TFLite 경유, `export_tflite.py` 참고).
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
  의존성이라 그대로 유지. 이 저장소는 앞으로 RunPod Docker 파이프라인 유지 + 모바일
  온디바이스 양자화 위주로 다룬다 — 진행 상황은 `train/QUANTIZATION_MOBILE.md`.
