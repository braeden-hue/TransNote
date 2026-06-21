# project.md — 맞춤형 악보 인식 & 변환 앱

---
## 📋 임시 메모 (2026-06-13)

### 가중치 완료 타임라인 (8/17 데모 역산)

| 기간 | 목표 | 완료 조건 |
|---|---|---|
| 6/13~6/23 | Round 2 재학습 | val_acc 75%+ (tiles_per_staff 상향 + perspective 증강) |
| 6/24~7/8 | Round 3 (2오선 + IMSLP fine-tuning) | val_acc 80%+ |
| 7/9~7/24 | Round 4 (실사 사진) + TFLite INT8 변환 | 변환 완료 후 FastAPI 서버 배포 |
| 7/25~8/4 | 앱 엔진 가중치 교체 + 웹 API 연결 + QA | 실기기 갤러리→OMR→커스텀악보 동작 확인 |
| 8/5~8/15 | 영상·포스터 완성 | 제출 완료 |

> **리스크**: Round 2가 7/1까지 75% 미달이면 Round 3 생략 후 바로 TFLite 변환 → 서버/앱 연결 확보 우선

---

### 우선순위 높은 TODO (학습 외)

#### Flutter NotationWidget 미구현 — 데모 전 필수
- [ ] **쉼표 표시**: `ScoreNote`에 타입 필드 추가 (`isRest: bool`), 회색 빈 셀 렌더링
- [ ] **박자표 표시**: `NotationWidget`에 `timeSignature` 파라미터 추가, 보표 좌측에 숫자 표시
- [ ] **셈여림 표시**: `ScoreNote` 또는 별도 이벤트로 `dynamic` 처리, Python처럼 한국어 변환 후 마디 상단
- [ ] **Grand staff (2단 보표)**: treble + bass `NotationWidget` 수직 배치 (`Column` 래퍼)
- [ ] **도돌이표 시각화**: `barline-start/end-repeat` 구분, `:‖` / `‖:` 기호 표시

#### 웹/서버
- [ ] FastAPI OMR 추론 서버 구성 (Round 2 완료 후 병행 시작)
- [ ] 웹 프론트엔드 OMR API 연결 (샘플 악보 즉시 체험 버튼 포함)

#### 예산 (연말 팝업 운영 기준, 100만원 내)
- Apple Developer (130,000원) + Google Play (35,000원) + 도메인 (15,000원)
- 서버: Railway 무료 플랜 먼저, 초과 시 Hetzner CX21 (≈6,000원/월)
- 디자인 에셋: UI8 Music UI Kit 일회성 (55,000~100,000원) + Envato Elements 1개월 집중 (30,000원) + LottieFiles 팩 (30,000원)

#### 영상 제작 일정
- 8/3~8/5: 앱+웹 화면 녹화 / 8/5~8/8: 나레이션+다이어그램 / 8/8~8/12: DaVinci Resolve 편집 / 8/15: 완성본

---

### 커스텀 악보 렌더러 미구현 현황

| 항목 | Python 렌더러 | Flutter Widget | 우선순위 |
|---|---|---|---|
| 쉼표 | ✅ | ❌ | 데모 필수 |
| 박자표 | ✅ | ❌ | 데모 필수 |
| Grand staff (2단) | ✅ | ❌ | 데모 필수 |
| 셈여림 | ✅ | ❌ | 데모 필수 |
| 도돌이표 시각화 | ⚠️ | ❌ | 데모 필수 |
| 크레센도·디미누엔도 | ❌ | ❌ | 있으면 좋음 |
| 아티큘레이션 | ❌ | ❌ | 있으면 좋음 |
| 셋잇단음표 | ❌ | ❌ | 있으면 좋음 |
| 이음줄/꾸밈음/페르마타/ottava/긴트릴 | ❌ | ❌ | 데모 범위 외 |

---

> 최종 수정: 2026-06-08  
> **현재 단계**: Phase 2 완료 — Round 1/2 학습 완료 (팀원 `team_ml` 결과물 병합)  
> **다음 실행**: Round 2 재학습 (누적 데이터 + 증강 강화) → Round 3 IMSLP 실사 fine-tuning  
>  
> **학습 현황**: Round 1 val_acc **72%** (TER 0.28), 평가 84~92% / Round 2 val_acc **64%** (Round 1보다 하락)  
> 가중치 위치: `ml/models/round1/`, `ml/models/round2/`  
> 개선 방안: `docs/score-training-agent.md` 참조

---

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

---

## 목표

악보의 장벽을 허물어 누구나 음악에 "연결"될 수 있도록 한다.  
악보 이미지를 촬영·업로드하면 OMR 모델이 음표를 인식하고 **사용자 정의 표기법**으로 변환한다.  
변환된 악보로 직접 연주 연습하고, 전세계 사용자와 공유할 수 있다.

**타겟 유저**: 악보를 처음 접하거나 읽기 어려운 사람 (현재: 피아노 중심, 향후 확장 가능)  
**입력 범위**: 이미지 한 장 기준 최대 3~4 마디, 초·중급 수준 악보  
**상업 출시**: Android / iOS 앱 + 웹 (학습 파이프라인 전체 상업 라이선스 확인 완료)

### 플랫폼별 역할

| 플랫폼 | 핵심 기능 |
|--------|---------|
| **웹 (온라인 팝업)** | 악보 이미지 업로드 → 커스텀 악보 변환 / 키보드 입력 가상 피아노 연주 |
| **앱 (Flutter)** | 변환 악보 공유 커뮤니티 / 마이크·MIDI 실시간 연주 감지 → 커스텀 악보 대조 피드백 |

---

## 세부 문서 (에이전트별)

| 문서 | 담당 에이전트 | 내용 |
|---|---|---|
| [`docs/project-orchestrator.md`](docs/project-orchestrator.md) | `project-orchestrator` | Phase 계획, 전체 현황, 에이전트 간 의존 관계, 병목 추적 |
| [`docs/score-training-agent.md`](docs/score-training-agent.md) | `score-training-agent` | 데이터 생성, 모델 아키텍처, Round 1~5 학습 파이프라인 |
| [`docs/score-recognition-engine.md`](docs/score-recognition-engine.md) | `score-recognition-engine` | C++ 추론 엔진 구조, INT8 양자화, 속도 최적화 |
| [`docs/sheet-music-qa.md`](docs/sheet-music-qa.md) | `sheet-music-qa` | Round별 정확도 평가, PASS 기준, 평가 스크립트 |
| [`docs/music-notation-rule-designer.md`](docs/music-notation-rule-designer.md) | `music-notation-rule-designer` | DeepScore 토큰 vocabulary, 커스텀 악보 표기법 시각 규칙 |
| [`docs/flutter-integration-architect.md`](docs/flutter-integration-architect.md) | `flutter-integration-architect` | Flutter/Dart FFI 통합, Riverpod 상태 관리, 카메라 연동 |
| [`docs/ui-design-specialist.md`](docs/ui-design-specialist.md) | `ui-design-specialist` | 튜토리얼·악보·연습 화면 UI, CustomPainter 악보 렌더러, 웹 데모 |

---

## 현재 구현 상태

| 레이어 | 상태 |
|---|---|
| Flutter UI (갤러리 선택 → MusicXML 표시) | ✅ 기본 구현 |
| Android Kotlin JNI 브리지 | ✅ 구현 완료 |
| C++ 추론 엔진 (`ml/omr/engine/`) | ✅ 구현 완료 (학습 완료 후 모델 교체 필요) |
| Python 렌더러 (`ml/omr/utils/render_notation.py`) | ✅ 구현 완료 |
| 학습 코드 (`ml/omr/training/`) | ✅ 구현 완료 (model.py·train.py·dataset.py) |
| TFLite 변환 스크립트 (`ml/omr/training/export_tflite.py`) | ✅ 구현 완료 (PyTorch → TFLite INT8) |
| 데이터 생성 (`ml/data/generate_random_scores.py`) | ✅ Round 2 기호 추가 완료 (vocab 1012) |
| 웹 데모 (`online_webpage/`) | ✅ 구현 완료 + UI 개선 (그라데이션·옥타브 연동·마우스 휠) |
| **Flutter 앱 UI 데모 (Phase 5-A)** | ✅ **완료** — 3탭(튜토리얼·악보·연습), CustomPainter 악보, 피아노 위젯 |
| Round 1 실제 학습 실행 | ✅ **완료** — Round 1 val_acc 72%, 평가 정확도 84~92% |
| Round 2 실제 학습 실행 | ⚠️ **완료 (재학습 권장)** — Round 2가 Round 1보다 낮음(64%), 누적 학습 방식으로 재학습 필요 |
| iOS OMR 브리지 | ❌ 스텁만 존재 (방안 A: ObjC++ 브리지 예정) |
| 실시간 카메라 | ❌ 미구현 |
| 웹 가상 피아노 (키보드 입력) | ❌ 미구현 |
| 악보 공유 백엔드 | ❌ 미구현 |
| 실시간 연주 감지 (MIDI / 마이크) | ❌ 미구현 |

---

## Phase 계획 요약

```
Phase 0  ✅  커스텀 표기법 규칙 확정 + C++ 엔진 분석
Phase 1  ✅  데이터 생성 파이프라인 (generate_dataset.py)
Phase 2  ✅  모델 학습 완료 (RTX 3080)
               Round 1: val_acc 72%, TER 0.28 → ml/models/round1/
               Round 2: val_acc 64% (Round 1 역전, 재학습 권장) → ml/models/round2/
               Round 3: 누적 학습 + 증강 강화 후 재학습 → IMSLP 실사 fine-tuning 포함
               Round 4: 실사 사진 fine-tuning
Phase 3  ✅  모델 변환·양자화 스크립트 완료 (ml/omr/training/export_tflite.py)
               실제 변환은 Round 1 학습 완료 후 실행
Phase 4  ✅  C++ 추론 엔진 구현 완료 (ml/omr/engine/)
Phase 5-A ✅  Flutter 앱 UI 데모 완료
               [구현 완료]
                 lib/main.dart          — 3탭 NavigationBar 셸 (튜토리얼·악보·연습)
                 lib/screens/tutorial_screen.dart — PageView 3단계 규칙 설명 + 미니 악보
                 lib/screens/score_screen.dart    — 샘플 악보 선택 → 커스텀 악보 + 피아노
                 lib/screens/practice_screen.dart — 순서대로 누르기 연습 (정답/오답 피드백)
                 lib/widgets/notation_widget.dart — CustomPainter 악보 렌더러
                 lib/widgets/piano_widget.dart    — 88건반 피아노 (하이라이트·오답 빨간 깜빡임)
                 lib/services/audio_service.dart  — 플랫폼 분기 오디오 (Web/네이티브 스텁)
                 lib/data/samples.dart            — 샘플 악보 데이터 (반짝반짝 등)
               [실행]
                 flutter run -d chrome      # 브라우저
                 flutter run -d windows    # 데스크탑
               [iOS 브리지] 방안 A (ObjC++ + MethodChannel) — Phase 5 이후 별도 진행
Phase 5      Flutter 통합 (Dart FFI 전환, 카메라, Riverpod)
Phase 6  🔄  커스텀 표기법 렌더링 (Python 렌더러 완료 / Flutter CustomPainter 미구현)
Phase 7      멀티플랫폼 (iOS CMake/Xcode + 웹 FastAPI)
Phase 8      웹 데모 — 악보 업로드 → 커스텀 변환 + 가상 피아노 연주
               - FastAPI 서버: OMR 추론 API + Python 렌더러 엔드포인트
               - 웹 프론트엔드: 악보 이미지 업로드 → 커스텀 악보 표시
               - 가상 피아노: 키보드 매핑 + Web Audio API 발음
               - 샘플 악보 번들 제공 (업로드 없이 즉시 체험)
Phase 9      공유 커뮤니티 + 연주 감지
               - 악보 공유 백엔드 (Firebase / Supabase)
               - MIDI 입력 (Web MIDI API) → 커스텀 악보 실시간 대조 ← 1순위
               - 마이크 단음 감지 (Web Audio API) → 대조 ← 2순위 (화음 불가)
```

---

## 기술 스택

| 항목 | 현재 | 목표 |
|---|---|---|
| OMR 모델 가중치 | 기존 HOMR 가중치 | 직접 학습 (PyTorch, RTX 3080) |
| 학습 데이터 | — | music21 생성 + 실사 사진 |
| 라벨 형식 | MusicXML | DeepScore 토큰 시퀀스 |
| 모델 포맷 (추론) | TFLite + ONNX | TFLite INT8 통일 |
| 추론 엔진 | MusicScore C++ JNI | 신규 C++ (ml/omr/engine/), Dart FFI |
| Flutter 브리지 | MethodChannel | Dart FFI |
| 이미지 입력 | 갤러리 선택 | 실시간 카메라 + 갤러리 |
| 출력 | MusicXML 텍스트 | 커스텀 표기법 이미지 |
| iOS | 스텁 | Dart FFI 기반 완전 구현 |
| 웹 | 없음 | FastAPI (OMR API) + React/Next.js 또는 Flutter Web |
| 악보 공유 백엔드 | 없음 | Firebase / Supabase (파일 스토리지 + DB) |
| 연주 입력 | 없음 | Web MIDI API (1순위) / Web Audio API 단음 감지 (2순위) |
| 상태 관리 | setState() | Riverpod |

---

## 현실적 검토 사항

### 실현 어려운 부분

| 항목 | 이유 | 대안 |
|------|------|------|
| 마이크 화음 감지 | 피아노 배음 구조로 기본 주파수 감지 어려움, 화음 동시 감지 사실상 불가 | 단음만 지원, MIDI 우선 권장 |
| 웹 OMR 브라우저 직접 실행 | C++ 엔진 WASM 빌드는 큰 작업, 모델 크기(수 MB) 로딩 지연 | FastAPI 서버 사이드 추론 API 방식 |
| 웹 데모 시기 | Phase 2~3 학습 완료 전까지 OMR 정확도 미보장 | 학습 전까지 MusicXML 직접 업로드 입력으로 데모 |

### 추가하면 좋은 것

| 항목 | 이유 |
|------|------|
| **MIDI 키보드 지원** (Web MIDI API) | 마이크보다 신뢰성 높음, USB-MIDI로 실제 피아노 연결 가능 |
| **샘플 악보 번들** | 업로드 없이 바로 체험 → 온보딩 마찰 제거 |
| **연주 점수 표시** | 몇 음 중 몇 음 맞았는지 — 학습 동기 부여 |
| **악보 난이도 자동 태깅** | 음표 밀도 기반 초/중/고급 분류 → 공유 시 유용 |
