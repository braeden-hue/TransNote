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
**사용자 정의 표기법**으로 변환해주는 웹 앱. `server.py`(FastAPI) 하나가 정적 웹앱(`webpage/`)을
서빙하면서 동시에 인식 API도 제공한다 — 별도 백엔드/프론트엔드 레포 분리 없음.

**이 저장소는 서버/배포(FastAPI, Vercel, RunPod Docker) 중심으로 정리돼 있다.** 모델 학습
코드·히스토리·정확도 근거는 별도 저장소 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote)
참고(2026-08-12 분리, `train/docs/`·`train/experiments/`·`test/`를 이 저장소에서 제거하고 이관).
`train/` 최상위 `.py`/`tokenizer258.json`은 RunPod Docker 이미지가 실제로 빌드 시 복사해가는
런타임 의존 파일이라 이 저장소에도 그대로 남아있다(아래 "train/ 내부 구조" 참고) — 지우면
배포가 깨진다.

**2026-08-09 이전에는 Flutter 앱 + 자체 C++ TFLite 엔진 구조였으나 전면 폐기됨** — 웹(FastAPI +
바닐라 JS) 단일 구조로 재구성했다. Flutter 시절 잔재는 더 이상 저장소에 없다.

## Directory Layout

```
server/               # 로컬/LAN 실행용 FastAPI 서버(server.py) + token_to_notes.py + requirements.txt
webpage/              # 정적 웹앱(HTML/CSS/JS), PWA(manifest.json)
api/                  # Vercel 서버리스 함수(얇은 프록시, 실제 추론은 runpod_serverless/가 전담)
runpod_serverless/    # RunPod Serverless GPU 추론 워커(Docker) — .github/workflows가 자동 빌드
train/                # OMR 추론 런타임 코드(PyTorch) — Docker가 복사해가는 최소 세트만 유지
docs/                 # music-notation-rule-designer.md 1개만 유지(커스텀 표기법 규칙, 코드가 실제 참조)
realImage/            # 실사 촬영 이미지(로컬 전용, .gitignore로 git 미포함)
secrets/              # API 키 등(.gitignore로 git 미포함, 절대 커밋 금지)
```

## Build & Run Commands

### 웹 서버
```bash
pip install -r server/requirements.txt
python server/server.py            # 0.0.0.0:8080, webpage/ 서빙 + OMR API
```

### 추론 동작 확인 (`train/`)
```bash
python train/inference.py --seq2seq <ckpt.pt> --tokenizer train/tokenizer258.json --analyze <dir>
```
학습(재학습·데이터 생성·라운드별 정확도)은 이 저장소 범위 밖 —
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote)에서 진행한다.

## Architecture

### 추론 경로 (실제 서비스, `server.py`)
```
webpage/(카메라 촬영·업로드) → POST /api/recognize
  → train/inference.py: run_image()
      1. dataset.py: detect_staffs() — OpenCV 고전 알고리즘으로 오선 검출(학습된 모델 아님)
      2. dataset.py: extract_staff_canvas() / extract_system_canvas() — 오선 크롭·정규화
      3. model.py: OmrSeq2Seq — CNN 인코더 + Transformer 디코더, autoregressive 토큰 생성
  → token_to_notes.py: tokens_to_score() — 토큰 시퀀스 → 커스텀 표기법 JSON
  → webpage/js/notation.js: renderNotation()/renderGrandStaff() — SVG 렌더링
```

**SegNet은 현재 추론 경로에 없다.** `inference.py`에 segnet 관련 코드가 전혀 없음 — 오선 검출은
순수 OpenCV(`detect_staffs()`)로만 한다. SegNet은 삭제된 옛 C++ 모바일 엔진 전용이었고
`train/checkpoints_legacy/segnet_best.pt`에 참고용으로만 남아있다(재도입 논의 시에만 필요).

### 프로덕션 체크포인트
```
train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt   # 유일한 채택 체크포인트(로컬/gitignore, RunPod 빌드는 GitHub Release에서 받음)
train/tokenizer258.json                                    # DeepScore 토큰 vocabulary
```
아키텍처(in_ch/backbone 깊이/pool_h)는 `model.py`의 `infer_arch_from_state_dict()`가 체크포인트
텐서 shape에서 자동 역산 — 별도 config 파일 불필요. r15 채택 근거·r16/r17/r18 기각 이유 등
학습 히스토리는 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote)의
`train/docs/TRAINING_REPORT.md` 참고(이 저장소엔 없음).

### `train/` 내부 구조
| 위치 | 역할 |
|---|---|
| `train/*.py` (top-level, ~14개) | `runpod_serverless/Dockerfile`이 `COPY train/*.py`로 통째로 복사해가는 런타임 세트 — `dataset.py`/`model.py`/`inference.py`가 실제 추론 경로에서 쓰이고, 나머지(`train.py`, `generate_scores.py` 등)는 학습 코드지만 sibling import 안전을 위해 같이 유지 |
| `train/checkpoints/` | r15(채택) — `.gitignore` 처리, 로컬에만 존재 |
| `train/checkpoints_legacy/` | 옛 Flutter/C++ 엔진 시절 체크포인트(segnet 포함) — 참고용, `.gitignore` 처리 |
| `train/real_texture_bank/` | `real_texture_augment.py`가 참조하는 노이즈 텍스처 png(304KB) — 학습 전용 코드 의존성이지만 용량이 작아 유지 |

학습 코드 전반(`train/experiments/`, `train/docs/`, 평가 스크립트 `test/`)은 2026-08-12에
[Model_TransNote](https://github.com/braeden-hue/Model_TransNote)로 이관하고 이 저장소에서
제거했다 — 학습/재현 관련 작업은 그 저장소에서 진행할 것.

## Platform Notes

- Python 서버(FastAPI/uvicorn) — Flutter/Dart/Android/iOS 툴체인 불필요.
- Web MIDI API(`navigator.requestMIDIAccess`)는 보안 컨텍스트(https:// 또는 http://localhost)
  필요 — 일반 LAN `http://<IP>:8080`에서는 동작 안 함, 실물 전자 피아노 연동 테스트 시 주의.
- `webpage/`는 CSS `zoom` 기반 auto-fit(`autoFitTutBoxes()`/`autoFitExpScore()`)으로 세로 스크롤
  없이 화면에 맞추는 패턴을 씀 — 비표준이지만 Chromium/Safari 지원, 태블릿/폰 가로 모드 타겟.

## Known Gaps / Follow-ups

- Firebase(닉네임 저장, 무료 티어) 클라이언트 SDK는 `webpage/js/firebase.js`에 이미 있음, 서버
  측 추가 폴더는 불필요하다고 판단됨(재검토 필요 시 `project.md` 참고).
- 2026-08-12: `test/`, `train/experiments/`, `train/docs/`, `.claude/agent-memory/`,
  `docs/archive/`·`docs/PLAN_booth_companion_page.md`를 저장소에서 제거(학습 히스토리는
  Model_TransNote로 이관, 나머지는 Flutter 시절 등 stale 문서). `docs/music-notation-rule-designer.md`는
  `webpage/js/notation.js`·`samples.js`·`train/generate_scores.py` 등이 실제로 참조하는 활성
  문서라 유지. `train/` 최상위 `.py`/`tokenizer258.json`/`checkpoints/`/`real_texture_bank/`는
  RunPod Docker 런타임 의존성이라 그대로 유지.
