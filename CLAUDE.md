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

**2026-08-09 이전에는 Flutter 앱 + 자체 C++ TFLite 엔진(`ml/omr/engine/`) 구조였으나 전면
폐기됨** — Flutter가 더 이상 타겟 플랫폼이 아니라서 `android/`, `ios/`, `lib/`, `ml/`,
`round1/`, `omr_bridge/`를 전부 삭제하고 웹(FastAPI + 바닐라 JS) 단일 구조로 재구성했다. 과거
Flutter 시절 문서(`appMake.md`, `FLUTTER_UI_PROGRESS.md`, 옛 학습 로드맵 `PODPLAN.md`/
`TrainingStep.md`/`step.md`)는 `docs/archive/`에 이력 보존용으로만 남아있다 — 현재 구조를
파악할 땐 참고하지 말 것, 최신 정보는 이 파일과 [`project.md`](project.md)를 따른다.

## Directory Layout

```
server.py            # FastAPI 서버 — webpage/ 서빙 + /api/recognize, /api/status, /api/score, /api/qr
webpage/              # 정적 웹앱(HTML/CSS/JS), PWA(manifest.json)
train/                # OMR 학습 파이프라인(PyTorch) — 아래 "ML Training" 참고
test/                 # 학습된 모델 평가/진단 스크립트(eval_*.py 등)
realImage/            # 실사 촬영 이미지(로컬 전용, .gitignore로 git 미포함)
designKit/            # 원본 악보(.mscz) 등 소스 자산
docs/                 # 서브에이전트별 상세 문서(docs/*.md) + docs/archive/(과거 로그)
secrets/              # API 키 등(.gitignore로 git 미포함, 절대 커밋 금지)
```

## Build & Run Commands

### 웹 서버
```bash
python server.py                   # 0.0.0.0:8080, webpage/ 서빙 + OMR API
```

### ML Training (`train/`)
```bash
python train/train.py --phase 2 --data_dir <dir> --tokenizer train/tokenizer258.json \
    --resume <ckpt.pt>             # 항상 이전 체크포인트에서 resume — random init은 학습 안 됨(실측 확인됨)
python train/inference.py --seq2seq <ckpt.pt> --tokenizer train/tokenizer258.json --analyze <dir>
python train/generate_scores.py --output train/Round3 ...   # music21+MuseScore로 합성 학습 데이터 생성
bash train/gen_render_local.sh     # 로컬 렌더링+검증 후 일괄 복사 (RunPod 데이터 생성 표준 경로)
```

데이터 생성·에폭·라운드별 정확도·문제 해결 과정(exposure bias, 노이즈 강건성, 마르코프 체인
가중 피치 선택 등)은 [`train/docs/TRAINING_REPORT.md`](train/docs/TRAINING_REPORT.md)에 정리돼
있다 — 학습 관련 작업 전에 먼저 읽을 것.

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
train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt   # 유일한 채택 체크포인트
train/tokenizer258.json                                    # DeepScore 토큰 vocabulary
```
아키텍처(in_ch/backbone 깊이/pool_h)는 `model.py`의 `infer_arch_from_state_dict()`가 체크포인트
텐서 shape에서 자동 역산 — 별도 config 파일 불필요. r15 채택 근거·r16/r17 기각 이유는
`train/docs/TRAINING_REPORT.md`와 `train/docs/HANDOFF_STATUS.md` 참고. RunPod 등 원격 배포 시
필요한 파일은 [`train/deploy_bundle/`](train/deploy_bundle/README.md) 참고(seq2seq+tokenizer
2개 파일만 필요, segnet 불필요).

### `train/` 내부 구조
| 위치 | 역할 |
|---|---|
| `train/*.py` (top-level, ~14개) | 현재 활성 파이프라인 — `dataset.py`/`model.py`/`train.py`/`inference.py`가 핵심, 나머지는 데이터 생성(`generate_scores.py`, `mscz_to_tokens.py`)·렌더링(`render_custom_notation.py`)·증강(`real_texture_augment.py`)·디버그 도구(`dump_canvas.py`, `render_one_exactpicture.py`, `render_sample10_comparison.py`) |
| `train/checkpoints/` | r15(채택) + r16/r17(기각, 참고용) — 전부 `.gitignore` 처리 |
| `train/checkpoints_legacy/` | 옛 Flutter/C++ 엔진 시절 체크포인트(segnet 포함) — 현재 파이프라인 미사용, 참고용 보관 |
| `train/experiments/` | 과거 라운드별 curriculum/pod/prepare/diag 셸스크립트 ~100개 — 이력 보존용 아카이브, 신규 작업은 여기 참고만 하고 새로 작성 |
| `train/docs/` | 학습 운영 문서(`TRAINING_REPORT.md`, `HANDOFF_STATUS.md`, `POD_TRAINING_CHECKLIST.md`, `CLOUD_SETUP.md`, `PLAN_r16_hide_timesig.md`) |
| `train/deploy_bundle/` | RunPod 등 원격 배포용 체크포인트 스테이징(생성물, git 미추적) |

**알려진 orphan(현재 아무 코드에서도 참조 안 됨, 삭제 검토 가능)**:
- `train/export_tflite.py` — 삭제된 Flutter/C++ 모바일 엔진용 TFLite export 스크립트. `server.py`는
  PyTorch 직접 추론이라 TFLite 변환 자체가 현재 배포 경로에 불필요. 향후 네이티브 앱을 다시
  추진할 때만 필요.
- `train/tokenizer1013.json` — vocab 분할(1013→258, `note-{pitch}-{dur}` → `note-{pitch}` +
  `dur-{dur}`) 이전의 구버전 vocab. 코드 어디서도 참조 안 됨.
- `train/tokenizer258_pre_tie.json` — tie(붙임줄) 토큰 추가 이전 스냅샷. `train/experiments/`의
  이미 아카이브된 1회성 마이그레이션 스크립트에서만 참조됨.

## Platform Notes

- Python 서버(FastAPI/uvicorn) — Flutter/Dart/Android/iOS 툴체인 불필요.
- Web MIDI API(`navigator.requestMIDIAccess`)는 보안 컨텍스트(https:// 또는 http://localhost)
  필요 — 일반 LAN `http://<IP>:8080`에서는 동작 안 함, 실물 전자 피아노 연동 테스트 시 주의.
- `webpage/`는 CSS `zoom` 기반 auto-fit(`autoFitTutBoxes()`/`autoFitExpScore()`)으로 세로 스크롤
  없이 화면에 맞추는 패턴을 씀 — 비표준이지만 Chromium/Safari 지원, 태블릿/폰 가로 모드 타겟.

## Known Gaps / Follow-ups

- `test/`, `train/experiments/`의 개별 스크립트 전수 감사는 아직 안 함(상위 레벨 orphan만 확인) —
  필요 시 요청.
- Firebase(닉네임 저장, 무료 티어) 클라이언트 SDK는 `webpage/js/firebase.js`에 이미 있음, 서버
  측 추가 폴더는 불필요하다고 판단됨(재검토 필요 시 `project.md` 참고).
- `git commit`/`push`로 이 저장소 재구성(TransNote 개명 포함)을 확정하는 작업은 사용자 확인
  대기 중 — 임의로 커밋하지 말 것.
