# flutter-integration-architect.md

> 담당 에이전트: `flutter-integration-architect`  
> 관련 코드: `lib/`, `android/`, `ios/`, `pubspec.yaml`

## 역할

Flutter UI 컴포넌트, Dart FFI 브리지, Riverpod 상태 관리, 카메라 기능을 연결하는 통합 담당.  
MethodChannel(현재) → Dart FFI(목표) 전환, 카메라 통합, Riverpod 상태 레이어 구축이 핵심 임무다.

---

## 현재 구현 상태

| 레이어 | 파일 | 상태 |
|---|---|---|
| 앱 셸 | `lib/main.dart` | ✅ 3탭 NavigationBar |
| 튜토리얼 화면 | `lib/screens/tutorial_screen.dart` | ✅ PageView 3단계 |
| 악보 화면 | `lib/screens/score_screen.dart` | ✅ 샘플 그리드 → 커스텀 악보 + 피아노 |
| 연습 화면 | `lib/screens/practice_screen.dart` | ✅ 순서대로 누르기 피드백 |
| 악보 렌더러 | `lib/widgets/notation_widget.dart` | ✅ CustomPainter |
| 피아노 위젯 | `lib/widgets/piano_widget.dart` | ✅ 88건반 하이라이트·오답 빨간 깜빡임 |
| 오디오 서비스 | `lib/services/audio_service.dart` | ✅ Web stub / 네이티브 stub |
| 샘플 데이터 | `lib/data/samples.dart` | ✅ 반짝반짝·학교종이 등 |
| OMR 브리지 | `lib/omr_service.dart` | ⚠️ MethodChannel 스텁 — FFI 전환 예정 |
| Android JNI | `MainActivity.kt`, `OmrPipeline.kt` | ✅ 구현 완료 |
| C++ JNI | `flutter_jni.cpp` | ✅ 이미지 bytes → OpenCV → pipeline |
| iOS | `AppDelegate.swift` | ❌ 스텁만 (OMR 미구현) |
| 실시간 카메라 | — | ❌ 미구현 |

---

## Phase 5 — Dart FFI 전환 (학습 완료 후 실행)

### 현재 브리지 구조

```
Flutter Dart
  → MethodChannel "com.example.musicscore/omr"
  → Kotlin (MainActivity / OmrPipeline)
  → JNI → C++ flutter_jni.cpp
```

### 목표 구조

```
Flutter Dart
  → dart:ffi + ffi 패키지
  → DynamicLibrary.open("libomr_engine.so")
  → C++ OmrEngine (ml/omr/engine/)
```

### `lib/omr_service.dart` 재작성 사항

| 항목 | 현재 | 변경 |
|---|---|---|
| 통신 방식 | MethodChannel | Dart FFI |
| 메서드 | `initialize()`, `processImageBytes()` | `init(modelDir)`, `process(bytes)` |
| 반환값 | MusicXML 문자열 | `List<OmrToken>` |

### `pubspec.yaml` 추가 패키지

```yaml
camera: ^0.11.x
flutter_riverpod: ^2.x
ffi: ^2.x
```

### FFI 전환 후 제거 대상

- `android/app/src/main/kotlin/.../MainActivity.kt` (MethodChannel 핸들러 부분)
- `android/app/src/main/kotlin/.../OmrPipeline.kt` (JNI 선언 전체)
- `android/app/src/main/cpp/flutter_jni.cpp`

---

## Riverpod 상태 구조

```
CameraProvider (StateNotifier)
  ├── 카메라 스트림 상태
  └── 캡처 트리거

OmrProvider (FutureProvider)
  ├── FFI 호출 → OmrEngine.process()
  └── List<OmrToken> 반환

NotationProvider (Provider)
  └── OmrToken 리스트 → 커스텀 악보 렌더링 데이터
```

---

## iOS (Phase 7-1)

- Dart FFI 방식이므로 Swift/ObjC 브리지 불필요
- `ios/Runner/`에 `libomr_engine.a` (정적 라이브러리) 포함
- iOS CMake/Xcode 빌드 설정 추가
- `Info.plist`: `NSCameraUsageDescription` 추가

---

## 웹 (Phase 7-2)

Flutter Web은 네이티브 Dart FFI 미지원 → FastAPI 서버 사이드 방식 사용.

```
Flutter Web UI
  → HTTP POST /omr (image bytes)
  → FastAPI 서버 (학습 서버, RTX 3080 겸용)
  → JSON 응답 (DeepScore 토큰)
```

---

## 확장 기능 연결 구조 (인식률 90%+ 확보 후)

```
DeepScore 토큰
  ├─→ 커스텀 악보 렌더러 → 악보 이미지
  │         └─→ 동시음 수직 연결선
  ├─→ MIDI 변환기 → 오디오 재생
  │         └─→ 플레이헤드 셀 하이라이트
  └─→ IMSLP 악보 검색
```

| 플랫폼 | MIDI 라이브러리 | 라이선스 |
|---|---|---|
| 웹 | Tone.js | MIT ✅ |
| 앱 | flutter_midi_pro | MIT ✅ |
