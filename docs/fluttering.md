# fluttering.md — Flutter 앱 (Android / iOS / Web)

> 관련 코드: `lib/`, `android/`, `ios/`, `pubspec.yaml`

---

## 1. 현재 구현 상태

| 레이어 | 파일 | 상태 |
|---|---|---|
| Flutter 앱 셸 | `lib/main.dart` | ✅ 3탭 NavigationBar (튜토리얼·악보·연습) |
| 튜토리얼 화면 | `lib/screens/tutorial_screen.dart` | ✅ PageView 3단계 규칙 설명 + 미니 악보 |
| 악보 화면 | `lib/screens/score_screen.dart` | ✅ 샘플 선택 그리드 → 커스텀 악보 + 피아노 |
| 연습 화면 | `lib/screens/practice_screen.dart` | ✅ 순서대로 누르기 (정답/오답 피드백) |
| 악보 렌더러 | `lib/widgets/notation_widget.dart` | ✅ CustomPainter (옥타브 존·셀·색상 테두리) |
| 피아노 위젯 | `lib/widgets/piano_widget.dart` | ✅ 88건반, 하이라이트·오답 빨간 깜빡임 |
| 오디오 서비스 | `lib/services/audio_service.dart` | ✅ 플랫폼 분기 (Web stub / 네이티브 stub) |
| 샘플 데이터 | `lib/data/samples.dart` | ✅ 샘플 악보 (반짝반짝·학교종이 등) |
| OMR 브리지 (Dart) | `lib/omr_service.dart` | ⚠️ MethodChannel 스텁 — 학습 완료 후 FFI 전환 예정 |
| Android Kotlin | `MainActivity.kt`, `OmrPipeline.kt` | ✅ JNI 호출 구현 완료 |
| C++ JNI | `flutter_jni.cpp` | ✅ 이미지 bytes → OpenCV → homr::OmrPipeline |
| iOS | `AppDelegate.swift` | ❌ 스텁만 존재 — OMR 미구현 |
| 실시간 카메라 | — | ❌ 미구현 (갤러리만 지원) |

---

## 2. 앱 수정 항목 (Phase 5)

### `lib/main.dart`

| 항목 | 현재 | 변경 |
|---|---|---|
| 이미지 입력 | 갤러리 선택만 | 실시간 카메라 스트림 추가 (`camera` 패키지) |
| 결과 표시 | MusicXML 텍스트 | 커스텀 표기법 악보 이미지 |
| UI 구조 | 단일 StatefulWidget | 카메라 뷰 / 처리 뷰 / 결과 화면 분리 |
| 상태 관리 | setState() | **Riverpod** 도입 |

### `lib/omr_service.dart` — 전면 재작성

| 항목 | 현재 | 변경 |
|---|---|---|
| 통신 방식 | MethodChannel | **Dart FFI** (`dart:ffi` + `ffi` 패키지) |
| 메서드 | `initialize()`, `processImageBytes()` | `init(modelDir)`, `process(bytes)` → `List<OmrToken>` |
| 반환값 | MusicXML 문자열 | DeepScore 토큰 구조체 리스트 |

### `pubspec.yaml` 추가 패키지

```yaml
camera: ^0.11.x          # 실시간 카메라
flutter_riverpod: ^2.x   # 상태 관리
ffi: ^2.x                # Dart FFI 지원
```

### Dart FFI 전환 이유

- MethodChannel: 비동기 메시지 직렬화 오버헤드
- Kotlin/Swift 브리지 레이어 제거 → 코드 단순화
- 순수 C++ + Dart 구성 → iOS/Web 이식 용이

### 제거 대상 파일

- `android/app/src/main/kotlin/.../MainActivity.kt` — MethodChannel 핸들러 부분
- `android/app/src/main/kotlin/.../OmrPipeline.kt` — JNI 선언 전체
- `android/app/src/main/cpp/flutter_jni.cpp` — JNI 진입점

### Dart FFI 연동 구조

```
Flutter Dart
  └── dart:ffi + ffi 패키지
      └── DynamicLibrary.open("libomr_engine.so")
          └── C++ OmrEngine (omr/engine/)
```

---

## 3. iOS 구현 계획 (Phase 7-1)

- Dart FFI 방식이므로 Swift 브리지 불필요
- `ios/Runner/`에 `libomr_engine.dylib` 또는 정적 라이브러리 포함
- iOS CMake/Xcode 빌드 설정 추가
- `Info.plist`: 카메라 권한(`NSCameraUsageDescription`) 추가
- Core ML delegate 활성화 (→ `docs/engine.md` Step 1 참조)

---

## 4. 웹 전략 (Phase 7-2)

Flutter Web은 네이티브 Dart FFI 미지원 → 서버 사이드 방식 사용.

| 방안 | 설명 | 복잡도 |
|---|---|---|
| **A. 서버 사이드** ← 권장 | Python FastAPI 백엔드 추론, HTTP로 결과 반환 | 낮음 |
| B. WASM | C++ 엔진을 Emscripten 컴파일, JS interop | 매우 높음 |
| C. TF.js | 모델을 TFJS 포맷으로 변환, 브라우저 직접 추론 | 중간 |

A안으로 시작. 학습 서버(RTX 3080)를 추론 서버로 겸용 가능.

---

## 5. 커스텀 표기법 렌더링 Flutter 구현 (Phase 6-2, 6-3)

- `omr/utils/render_notation.py` 로직을 Dart `CustomPainter`로 포팅
- 처리 흐름: 토큰 파싱 → 마디별 그룹화 → 셀 너비 계산 → 존 위치 결정 → 글자·테두리 색상 할당 → Canvas
- 성능 문제 시: C++ + OpenCV로 이미지 생성 후 Flutter에 전달하는 방식으로 전환
- 실시간 카메라 오버레이 연결

---

## 6. 앱·웹 확장 기능 (Round 4 완료 후 구현)

> 전제: OMR 인식률 90%+ 확보 이후.

### 6-1. 악보 검색 (IMSLP 연동)

| 소스 | 방식 | 라이선스 |
|---|---|---|
| **IMSLP** | MediaWiki API 검색 → 유저 직접 다운로드 | 퍼블릭 도메인 한정, 서버 자동 다운로드 금지 |
| Google Custom Search API | 이미지 검색 | $5/1,000 쿼리, 반환 이미지 저작권은 유저 책임 |
| 내부 샘플 라이브러리 | 서버 저장 퍼블릭 도메인 악보 | 바흐·모차르트·쇼팽 등 사망 70년+ |

검색 흐름:
```
검색창 입력 → IMSLP API 결과 표시
→ A안: IMSLP 페이지 연결 → 유저 직접 다운로드 → 업로드 (저작권 안전)
→ B안: 이미지 검색 URL 직접 전달 → 서버 fetch → 변환 (면책 고지 필수)
```

저작권 기준: 작곡가 사망 후 70년 초과. 편곡자·출판사 권리는 IMSLP 라이선스 태그 확인 필수.

---

### 6-2. MIDI 재생

DeepScore 토큰 → MIDI 이벤트 → 오디오 출력:

```
note-C4-1/4    → MIDI note 60, duration = 60/BPM × (1/4) 초
rest-1/4       → silence
tuplet-3-start → 범위 내 음가를 2/3으로 보정
ottava-8va     → 범위 내 음표 pitch +12
```

**플랫폼별 구현**

| 플랫폼 | 라이브러리 | 라이선스 |
|---|---|---|
| 웹 | **Tone.js** | MIT ✅ |
| 앱 (Flutter) | **flutter_midi_pro** | MIT ✅ |
| 앱 대안 | **just_audio** | MIT ✅ |

**사운드폰트 선택 (라이선스 주의)**

| 사운드폰트 | 라이선스 | 상업 사용 |
|---|---|---|
| MuseScore General SF2 | GPL | ❌ 불가 |
| FluidR3_GM | LGPL | 조건부 |
| **GeneralUser GS** | Freeware | ✅ |
| **SGM-V2.01** | CC BY 3.0 | ✅ (저작자 표시 필수) |

UX: 재생/일시정지/정지 버튼, BPM 슬라이더 (50~150%), 재생 중 음표 하이라이트

---

### 6-3. 동시음 연결선 (2오선 한정)

같은 beat offset에서 두 보표에 동시에 눌러야 할 음을 수직선으로 연결.

렌더링 규칙:
- 두 보표 사이 공간에 얇은 수직 점선
- 연결선 색상 = 해당 열 음표의 색상 테두리와 동일
- 한쪽이 쉼표이면 연결선 없음
- 동시 음이 3개 이상이면 최상단-최하단만 연결

---

### 6-4. 실시간 재생 위치 표시 (플레이헤드)

```
4/4박 한 마디 — 2번째 셀 재생 중:
┌────┬▓▓▓▓┬────┬────┐
│ C  │▓D▓ │ E  │    │
└────┴▓▓▓▓┴────┴────┘
```

동기화 계산:
```
(현재 시간 - 시작 timestamp) × BPM / 60,000 = 현재 beat 위치
beat 위치 → 셀 인덱스 → 반투명 오버레이
```

| 플랫폼 | 구현 |
|---|---|
| 웹 | `requestAnimationFrame` 루프 |
| 앱 | Flutter `AnimationController` + `Ticker` |

---

### 6-5. 확장 기능 연결 구조

```
DeepScore 토큰 (OMR 출력)
    │
    ├─→ [커스텀 악보 렌더러] → 악보 이미지
    │         └─→ [동시음 연결선] → 보표 간 수직선 (6-3)
    │
    ├─→ [MIDI 변환기] → 오디오 재생 (6-2)
    │         └─→ [타임라인] → beat offset 계산
    │                  └─→ [플레이헤드] → 셀 하이라이트 (6-4)
    │
    └─→ [악보 검색] → IMSLP / 샘플 라이브러리 (6-1)
```

세 기능(MIDI 재생, 연결선, 플레이헤드)은 **beat offset**을 공통 입력으로 사용 → 파싱 로직 한 번만 구현.
