# project-orchestrator.md

> 담당 에이전트: `project-orchestrator`  
> 참조 문서: `project.md` (전체 Phase 계획), 각 에이전트 docs/

## 역할

프로젝트 전체 현황을 파악하고 에이전트 간 병목·충돌을 조율한다.  
어느 Phase가 완료되었는지, 다음 실행 대상은 무엇인지, 각 에이전트가 무엇을 기다리는지 추적한다.

---

## 목표

악보 이미지를 촬영·업로드하면 OMR 모델이 음표를 인식하고 **사용자 정의 표기법**으로 변환한다.  
**타겟**: 악보를 처음 접하거나 읽기 어려운 사람 (피아노 중심)  
**출시**: Android / iOS 앱 + 웹

| 플랫폼 | 핵심 기능 |
|---|---|
| 웹 | 악보 이미지 업로드 → 커스텀 악보 변환 / 가상 피아노 연주 |
| 앱 | 변환 악보 공유 커뮤니티 / MIDI 실시간 연주 감지 → 커스텀 악보 대조 |

---

## Phase 계획

```
Phase 0  ✅  커스텀 표기법 규칙 확정 + C++ 엔진 분석
Phase 1  ✅  데이터 생성 파이프라인 (generate_dataset.py, vocab 1012)
Phase 2  🔜  모델 학습 (RTX 3080) ← 다음 실행
               Round 1: 기본 음표 3,000장 → SegNet(50ep) → Enc+Dec(100ep) → E2E(30ep)
               Round 2: 전체 기호 확장 (vocab 1012, load_checkpoint_with_vocab_expansion)
               Round 3: 2오선 grand staff
               Round 4: 실사 사진 fine-tuning
               Round 5: 실제 카메라 촬영 도메인 적응
Phase 3  ✅  모델 변환·양자화 스크립트 완료 (export_tflite.py) — 실제 변환은 Round 1 후
Phase 4  ✅  C++ 추론 엔진 구현 완료 (ml/omr/engine/)
Phase 5-A ✅  Flutter 앱 UI 데모 완료 (3탭, CustomPainter, 피아노 위젯)
Phase 5      Flutter 통합 (Dart FFI 전환, 카메라, Riverpod)
Phase 6  🔄  커스텀 표기법 렌더링 (Python 렌더러 ✅ / Flutter CustomPainter 미구현)
Phase 7      멀티플랫폼 (iOS CMake/Xcode + 웹 FastAPI)
Phase 8      웹 데모 — 악보 업로드 → 커스텀 변환 + 가상 피아노
Phase 9      공유 커뮤니티 + 연주 감지 (MIDI 우선)
```

---

## 현재 구현 상태 요약

| 레이어 | 상태 |
|---|---|
| Flutter UI (3탭 앱 데모) | ✅ |
| Android Kotlin JNI 브리지 | ✅ |
| C++ 추론 엔진 (`ml/omr/engine/`) | ✅ (모델 교체 필요) |
| Python 렌더러 (`ml/omr/utils/render_notation.py`) | ✅ |
| 학습 코드 (`ml/omr/training/`) | ✅ |
| TFLite 변환 스크립트 (`export_tflite.py`) | ✅ |
| 데이터 생성 (`generate_random_scores.py`) | ✅ Round 1~4 grand staff 포함 |
| Round 5 테스트 JSON 레이블 (`ml/scripts/mscz_to_label.py`) | ✅ |
| 웹 데모 (`online_webpage/`) | ✅ |
| **Round 1 실제 학습 실행** | ❌ 다음 단계 |
| iOS OMR 브리지 | ❌ 스텁만 |
| 실시간 카메라 | ❌ 미구현 |
| 악보 공유 백엔드 | ❌ 미구현 |

---

## 에이전트 간 의존 관계

```
score-training-agent
  → (Round 1 완료) → score-recognition-engine (모델 가중치 교체)
  → (Round 1 완료) → score-recognition-engine (TFLite INT8 변환)
  → (Round 4 완료) → flutter-integration-architect (Dart FFI 통합)

music-notation-rule-designer
  → (토큰 확정) → score-training-agent (vocabulary 기준)
  → (렌더링 규칙) → ui-design-specialist (Flutter CustomPainter)

sheet-music-qa
  → (각 Round 완료 후) → project-orchestrator (PASS 여부 보고)
  → (PASS 확인 후) → score-training-agent (다음 Round 진행 허가)

ui-design-specialist
  → (Flutter CustomPainter 완료) → flutter-integration-architect (위젯 통합)
```

---

## 병목 추적

| 병목 | 현재 상태 | 해소 조건 |
|---|---|---|
| Round 1 학습 미실행 | Phase 2 대기 중 | RTX 3080 환경에서 `ml/scripts/train_round.py --round 1` 실행 |
| TFLite 변환 미실행 | 학습 완료 대기 | Round 1 val TER ≤ 0.10 달성 후 `export_tflite.py` |
| Dart FFI 미구현 | MethodChannel 스텁 | 모델 확정 후 `omr_service.dart` 재작성 |
| iOS OMR 없음 | 스텁만 | Phase 7-1에서 진행 |

---

## 기술 스택

| 항목 | 현재 | 목표 |
|---|---|---|
| OMR 모델 가중치 | 기존 HOMR 가중치 | 직접 학습 (PyTorch, RTX 3080) |
| 라벨 형식 | MusicXML | DeepScore 토큰 시퀀스 |
| 모델 포맷 | TFLite + ONNX | TFLite INT8 통일 |
| Flutter 브리지 | MethodChannel | Dart FFI |
| 상태 관리 | setState() | Riverpod |
| 웹 추론 | 없음 | FastAPI 서버 사이드 |

---

## 상업 라이선스 확인 완료

| 라이브러리 | 라이선스 | 상업 이용 |
|---|---|---|
| OpenCV | Apache 2.0 | ✅ |
| TensorFlow Lite | Apache 2.0 | ✅ |
| Tone.js / flutter_midi_pro | MIT | ✅ |
| GeneralUser GS 사운드폰트 | Freeware | ✅ |
