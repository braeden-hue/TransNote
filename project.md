# project.md — 맞춤형 악보 인식 & 변환 앱

> 최종 수정: 2026-05-17  
> **현재 단계**: Phase 5-A 완료 (Flutter 3탭 앱 UI 구현, 웹 데모 UI 개선) — Round 1 학습 데이터(3,000장) 생성 후 학습 예정  
> **다음 실행**: Phase 2 — Round 1 학습 (데이터 3,000장 생성 → SegNet/Encoder/Decoder 학습 실행)

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

## 세부 문서

| 문서 | 내용 |
|---|---|
| [`docs/training.md`](docs/training.md) | 학습 파이프라인 — 데이터 생성, 모델 아키텍처, Round 1~4 계획, 정확도 측정, 대응 방안 |
| [`docs/engine.md`](docs/engine.md) | C++ 추론 엔진 — omr/engine/ 구조, Dart FFI API, INT8 양자화, 속도 최적화 |
| [`docs/custom.md`](docs/custom.md) | DeepScore 토큰 설계 + 커스텀 악보 표기법 규칙 + Python 렌더러 |
| [`docs/fluttering.md`](docs/fluttering.md) | Flutter 앱 (Android / iOS / Web) — Dart FFI 전환, 확장 기능 (MIDI·플레이헤드·IMSLP) |

---

## 현재 구현 상태

| 레이어 | 상태 |
|---|---|
| Flutter UI (갤러리 선택 → MusicXML 표시) | ✅ 기본 구현 |
| Android Kotlin JNI 브리지 | ✅ 구현 완료 |
| C++ 추론 엔진 (`omr/engine/`) | ✅ 구현 완료 (학습 완료 후 모델 교체 필요) |
| Python 렌더러 (`omr/utils/render_notation.py`) | ✅ 구현 완료 |
| 학습 코드 (`omr/training/`) | ✅ 구현 완료 (model.py·train.py·dataset.py) |
| TFLite 변환 스크립트 (`omr/training/export_tflite.py`) | ✅ 구현 완료 (PyTorch → TFLite INT8) |
| 데이터 생성 (`omr/data_gen/generate_dataset.py`) | ✅ Round 2 기호 추가 완료 (vocab 1012) |
| 웹 데모 (`web-demo/`) | ✅ 구현 완료 + UI 개선 (그라데이션·옥타브 연동·마우스 휠) |
| **Flutter 앱 UI 데모 (Phase 5-A)** | ✅ **완료** — 3탭(튜토리얼·악보·연습), CustomPainter 악보, 피아노 위젯 |
| Round 1 실제 학습 실행 | ❌ **다음 단계** — 3,000장 생성 후 RTX 3080 학습 |
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
Phase 2  🔜  모델 학습 (RTX 3080) ← **다음 실행**
               Round 1: 기본 음표 3,000장 생성 → SegNet(50ep) → Enc+Dec(100ep) → E2E(30ep)
               Round 2: 전체 기호 확장 (vocab 1004→1012, load_checkpoint_with_vocab_expansion)
               Round 3: 2오선 grand staff
               Round 4: 실사 사진 fine-tuning
Phase 3  ✅  모델 변환·양자화 스크립트 완료 (omr/training/export_tflite.py)
               실제 변환은 Round 1 학습 완료 후 실행
Phase 4  ✅  C++ 추론 엔진 구현 완료 (omr/engine/)
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
| 추론 엔진 | MusicScore C++ JNI | 신규 C++ (omr/engine/), Dart FFI |
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
