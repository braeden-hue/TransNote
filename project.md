# project.md — 맞춤형 악보 인식 & 변환 앱 세부 계획서

> 최종 수정: 2026-04-11  
> 목표: 실시간 카메라로 악보를 촬영하면 자체 학습 OMR 모델이 음표를 인식하고, 인식 결과를 사용자 정의 표기법으로 변환한 새 악보 이미지를 출력한다.
>
> **현재 단계**: Phase 4 진행 중 — C++ 추론 엔진 구현 완료, Phase 2(학습) 대기 중

---

## 1. 현재 프로젝트 상태 분석

### 1-1. 현재 구현된 것

| 레이어 | 파일 | 상태 |
|--------|------|------|
| Flutter UI | `lib/main.dart` | 갤러리 이미지 선택 → MusicXML 표시 (기본 구현) |
| Dart Bridge | `lib/omr_service.dart` | MethodChannel 2개 메서드 (initialize, processImageBytes) |
| Android Kotlin | `MainActivity.kt`, `OmrPipeline.kt` | JNI 호출 구현 완료 |
| C++ JNI | `flutter_jni.cpp` | 이미지 bytes → OpenCV → homr::OmrPipeline |
| C++ OMR | `MusicScore/` 레포 | HOMR 파이프라인 (segnet + encoder + decoder) |
| Desktop 테스트 | `desktop/test_main.cpp` | C++ 단독 실행 가능 |
| iOS | `AppDelegate.swift` | **스텁만 존재 — OMR 미구현** |
| 웹 | — | **없음** |
| 실시간 카메라 | — | **없음 (갤러리만 지원)** |

### 1-2. 현재 추론 엔진 구조 (참조용, MusicScore 레포)

```
입력 이미지
  → OpenCV 전처리 (crop, resize, denoise)
  → TFLite segnet_308_int8.tflite          (픽셀 분류 segmentation)
  → TFLite encoder_331_int8.tflite         (특징 인코딩)
  → ONNX Runtime decoder_331_int8.onnx     (토큰 시퀀스 디코딩)
  → tr_omr_parser → rhythm/accidental rules
  → MusicXML 출력
```

이 구조(segnet + encoder + decoder)는 유지하되, **모델 가중치를 처음부터 직접 학습**하는 방향으로 전환한다. 기존 MusicScore 레포의 C++ 추론 엔진 코드는 새 엔진 작성의 참조 기반으로 활용한다.

---

## 2. 자체 학습 파이프라인

### 2-1. 전체 흐름

```
[데이터 생성]
  music21 무작위 악보 생성
  + 직접 촬영한 악보 사진
        ↓
[라벨링]
  DeepScore 규격 라벨 시퀀스 생성
        ↓
[학습 (RTX 3080)]
  segnet → encoder → decoder
  loss 계산 → 가중치 업데이트
        ↓
[양자화]
  segnet + encoder: INT8 PTQ → TFLite .tflite
  decoder: INT8 양자화 → .tflite 또는 ONNX
        ↓
[C++ 추론 엔진]
  OpenCV + TFLite C++ API
  (MusicScore 엔진 구조 참조, Dart FFI용 .so 빌드)
        ↓
[Flutter 통합]
  Dart FFI → 공유 라이브러리(.so / .dylib)
  Android / iOS / Web
```

---

### 2-2. 데이터 생성 (music21)

#### 무작위 악보 생성 전략
- `music21.stream.Score` 로 무작위 마디/음표 생성
- 제어 파라미터: 박자표(4/4, 3/4, 6/8 등), 조표, 음역, 음가 분포, 쉼표 비율, 화음 비율
- 렌더링 방법 (선택):
  - LilyPond 렌더링 → PNG (선호, 제어도 높음)
  - MuseScore CLI 렌더링 → PNG
- 출력물: 악보 이미지(.png) + 대응하는 DeepScore 라벨 파일

#### 실사 데이터 확대
- 직접 촬영한 악보 사진 추가
- 오프라인 증강(augmentation):
  - 원근 왜곡 (perspective transform)
  - 밝기/대비/노이즈 변화
  - 부분 흐림 (motion blur, defocus)
  - 그림자, 구김 시뮬레이션
- 촬영 사진은 수동 라벨링 또는 기존 정답 MusicXML에서 역변환으로 라벨 생성

#### 데이터셋 규모 목표 (미확정, 추후 설정)
- 생성 악보: 수만~수십만 장 (music21 자동 생성이므로 확장 용이)
- 실사 사진: 수백~수천 장

---

### 2-3. DeepScore 규격 라벨 시퀀스 (확정)

#### 토큰 형식 (단일 통합 vocabulary)

| 토큰 | 형식 | 예시 |
|------|------|------|
| 특수 토큰 | `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` | — |
| 음자리표 | `clef-{G\|F\|C}` | `clef-G`, `clef-F` |
| 조표 | `key-{name}` | `key-C`, `key-G`, `key-Bb` |
| 박자표 | `time-{num}/{den}` | `time-4/4`, `time-6/8` |
| 음표 (단음) | `note-{pitch}{oct}-{dur}` | `note-C4-1/4`, `note-F#5-1/8`, `note-Bb3-1/2` |
| 화음 추가음 | `chord-{pitch}{oct}` | `chord-E4`, `chord-G4` |
| 쉼표 | `rest-{dur}` | `rest-1/4`, `rest-1/8` |
| 세로줄 | `barline`, `barline-double`, `barline-final`, `barline-start-repeat`, `barline-end-repeat` | — |

#### 음가(duration) 토큰 — 전음표 기준 분수

| 토큰 | 음가 | quarterLength |
|------|------|---------------|
| `1/1` | 온음표 | 4.0 |
| `3/4` | 점2분음표 | 3.0 |
| `1/2` | 2분음표 | 2.0 |
| `3/8` | 점4분음표 | 1.5 |
| `1/4` | 4분음표 | 1.0 |
| `3/16` | 점8분음표 | 0.75 |
| `1/8` | 8분음표 | 0.5 |
| `3/32` | 점16분음표 | 0.375 |
| `1/16` | 16분음표 | 0.25 |
| `1/32` | 32분음표 | 0.125 |

#### 화음 인코딩 방식

화음은 **가장 낮은 음이 `note-` 토큰**, 나머지 음들이 `chord-` 토큰으로 이어진다:
```
note-C4-1/4   chord-E4   chord-G4
→ C4+E4+G4 를 4분음표로 연주 (C장조 으뜸화음)
```

#### 완성된 시퀀스 예시
```
<SOS>  clef-G  key-C  time-4/4
note-C4-1/4  note-E4-1/4  note-G4-1/2
barline
note-F4-1/4  rest-1/4  note-C4-1/4  chord-E4  chord-G4
barline-final
<EOS>
```

#### vocabulary 파일
- `data/tokenizer.json` — `{"<PAD>": 0, "clef-G": 1, ...}` 형식
- 총 토큰 수: 약 1,000개 (17 pitch class × 5 octave × 10 dur + chord + rest + header)
- `scripts/generate_dataset.py` 실행 시 자동 생성됨

#### music21 → DeepScore 변환 (`generate_dataset.py` 구현 완료)
```python
# generate_dataset.py 내 _element_to_tokens() 함수 참조
# Note   → ["note-C4-1/4"]
# Chord  → ["note-C4-1/4", "chord-E4", "chord-G4"]
# Rest   → ["rest-1/4"]
```

---

### 2-4. 입력 이미지 해상도 정책

스마트폰 촬영 환경에 적합하도록 다음 해상도 기준을 사용한다:

| 항목 | 값 | 근거 |
|------|-----|------|
| 학습 데이터 생성 DPI | **150 DPI** | A4 기준 ≈ 1240×1754 px, 스마트폰 중간 해상도와 유사 |
| 추론 시 입력 최대 너비 | **1280 px** | 스마트폰 촬영 후 리사이즈 기준 (전처리 단계에서 통일) |
| 종횡비 | 유지 | autocrop → 비율 유지 resize |

학습 데이터는 `PNG_DPI = 150` 으로 생성 (`scripts/generate_dataset.py` 내 설정).

### 2-5. 모델 아키텍처 및 학습

#### 모델 구성 (HOMR 구조 계승)

| 모듈 | 역할 | 비고 |
|------|------|------|
| **Segnet** | 입력 이미지 → 픽셀별 음악 기호 클래스 분류 | U-Net 계열 또는 FCN |
| **Encoder** | Segnet 출력(feature map) → latent sequence | CNN stride / Transformer patch encoder |
| **Decoder** | Encoder 출력 → DeepScore 토큰 시퀀스 | Autoregressive Transformer |

#### 학습 환경
- GPU: **RTX 3080** (VRAM 10 GB)
- 프레임워크: PyTorch (학습), 변환 후 TFLite (추론)
- 학습 스크립트 위치: `training/` 디렉토리 (Phase 2에서 생성 예정)
- 상업적 이용 가능 라이선스 모델만 참조 (Apache 2.0, MIT, BSD 등)

#### Loss 구성

| 단계 | Loss | 비고 |
|------|------|------|
| Segnet | Cross-Entropy (픽셀 분류) | 클래스 불균형 → weighted CE 또는 Focal Loss 고려 |
| Encoder | Segnet과 공동 학습 또는 분리 학습 | |
| Decoder | Cross-Entropy (토큰 시퀀스) | Teacher forcing 사용 |
| 전체 | Segnet loss + λ × Decoder loss | λ 하이퍼파라미터 실험으로 결정 |

#### 학습 단계
1. Segnet 단독 사전 학습 (픽셀 분류 수렴 확인)
2. Encoder-Decoder 연결 학습 (Segnet frozen → 점진적 unfreeze)
3. End-to-end fine-tuning
4. 정확도 검증 → 데이터 추가/증강 반복

---

### 2-6. INT8 양자화

#### 대상
- **Segnet**: TFLite Post-Training Quantization (PTQ) INT8
- **Encoder**: TFLite PTQ INT8
- **Decoder**: INT8 양자화 검토 (정확도 손실 주의, 실험 필요)

#### 방법
```
PyTorch 모델 (.pth)
  → ONNX export (torch.onnx.export)
  → TFLite 변환 (onnx-tf → TFLite converter)
  → Representative dataset으로 INT8 calibration
  → segnet_INT8.tflite, encoder_INT8.tflite 생성
```

#### 검증 기준
- INT8 모델 정확도 vs FP32 기준 모델 정확도 비교
- `desktop/compare_musicxml.py` 또는 DeepScore 라벨 기준 정확도 측정
- 허용 정확도 손실 기준: 미확정 (추후 설정)

---

## 3. C++ 추론 엔진 (신규 작성)

### 3-1. 설계 방향

기존 `MusicScore/app/src/main/cpp` 의 엔진 코드를 **참조**하되, 다음 차이점을 반영해 새로 작성한다:

| 항목 | 기존 MusicScore 엔진 | 새 엔진 |
|------|---------------------|---------|
| 브리지 | JNI (Kotlin → C++) | **Dart FFI** (Dart → C++ 직접) |
| 입력 포맷 | Android ByteArray | 범용 `uint8_t*` 버퍼 |
| 출력 포맷 | MusicXML 문자열 | **DeepScore 토큰 시퀀스** (구조체 배열 또는 JSON) |
| 모델 로드 | Android AssetManager | 파일 경로 기반 (플랫폼 독립) |
| ONNX Runtime | 사용 | 제거 (decoder도 TFLite로 통일) |

### 3-2. 엔진 내부 구조

```
OmrEngine (C++)
  ├── Preprocessor          (OpenCV)
  │     autocrop, resize, normalize
  ├── SegnetRunner           (TFLite C++ API)
  │     INT8 segmentation
  ├── EncoderRunner          (TFLite C++ API)
  │     INT8 feature encoding
  ├── DecoderRunner          (TFLite C++ API)
  │     token sequence generation (autoregressive)
  ├── TokenParser            (자체 구현)
  │     DeepScore 토큰 → 음표 구조체 변환
  └── Public C API           (Dart FFI용)
        omr_init(model_path)
        omr_process(image_bytes, len) → token_list
        omr_free()
```

### 3-3. Dart FFI 연동

기존 MethodChannel(Dart → Kotlin → JNI → C++) 방식을 **Dart FFI**(Dart → C++ 직접)로 교체한다.

```
Flutter Dart
  └── dart:ffi + ffi 패키지
      └── DynamicLibrary.open("libomr_engine.so")
          └── C++ OmrEngine (공유 라이브러리)
```

#### 변경이 필요한 이유
- MethodChannel은 비동기 메시지 직렬화 오버헤드 발생
- Kotlin/Swift 브리지 레이어 제거로 코드 단순화
- 플랫폼 코드 없이 순수 C++ + Dart로 구성 가능 → iOS/Web 이식 용이

#### 제거 대상 파일
- `android/app/src/main/kotlin/.../MainActivity.kt` — MethodChannel 핸들러 부분
- `android/app/src/main/kotlin/.../OmrPipeline.kt` — JNI 선언 전체
- `android/app/src/main/cpp/flutter_jni.cpp` — JNI 진입점

#### 수정 대상 파일
- `lib/omr_service.dart`: MethodChannel → Dart FFI로 전면 교체
- `android/app/src/main/cpp/CMakeLists.txt`: JNI 제거, FFI용 공유 라이브러리 타깃으로 변경

---

## 4. Flutter 앱 수정/추가 항목

### 4-1. `lib/main.dart` — 수정 필요

| 항목 | 현재 상태 | 필요한 변경 |
|------|-----------|------------|
| 이미지 입력 | 갤러리 선택만 가능 | **실시간 카메라 스트림 추가** (`camera` 패키지) |
| 결과 표시 | MusicXML 텍스트 출력 | **커스텀 표기법 악보 이미지 표시**로 교체 |
| UI 구조 | 단일 StatefulWidget | 카메라 뷰 / 처리 뷰 / 결과 화면으로 분리 |
| 상태 관리 | setState() | **Riverpod** 도입 권장 (카메라·추론·렌더링 상태 분리) |

### 4-2. `lib/omr_service.dart` — 전면 재작성

| 항목 | 현재 상태 | 필요한 변경 |
|------|-----------|------------|
| 통신 방식 | MethodChannel | **Dart FFI** (`dart:ffi` + `ffi` 패키지) |
| 메서드 | `initialize()`, `processImageBytes()` | `init(modelDir)`, `process(bytes)` → `List<OmrToken>` |
| 반환값 | MusicXML 문자열 | DeepScore 토큰 구조체 리스트 |
| 오류 처리 | 없음 | FFI 예외, 모델 미초기화, null 포인터 가드 추가 |

### 4-3. `pubspec.yaml` — 추가 필요

```yaml
camera: ^0.11.x          # 실시간 카메라
flutter_riverpod: ^2.x   # 상태 관리
ffi: ^2.x                # Dart FFI 지원
```

### 4-4. iOS — 신규 구현 필요

- Dart FFI 방식이므로 Kotlin/Swift 브리지 불필요
- `ios/Runner/` 에 `libomr_engine.dylib` 또는 정적 라이브러리 포함
- iOS CMake/Xcode 빌드 설정 추가
- `Info.plist`: 카메라 권한(`NSCameraUsageDescription`) 추가 필요

### 4-5. 웹 — 신규 구현 필요

Flutter Web은 네이티브 Dart FFI 미지원 → 별도 전략 필요:

| 방안 | 설명 | 복잡도 |
|------|------|--------|
| **A. 서버 사이드** | Python FastAPI 백엔드에서 PyTorch 모델 추론, HTTP로 결과 반환 | 낮음 |
| B. WASM | C++ 엔진을 Emscripten으로 컴파일, Flutter Web에서 JS interop | 매우 높음 |
| C. TensorFlow.js | 모델을 TFJS 포맷으로 변환, 브라우저 직접 추론 | 중간 |

**권장**: A안으로 시작. 학습 서버(RTX 3080)를 추론 서버로 겸용 가능.

---

## 5. 커스텀 악보 표기법 (렌더링 대상)

### 5-1. 핵심 규칙

기존 오선지(5줄) 대신 **오선지 크기의 빈 박스** 위에 다음 규칙으로 표기:

| 건반 종류 | 표기 방식 |
|-----------|-----------|
| 흰 건반 (C, D, E, F, G, A, B) | **영문자** |
| 검은 건반 (C#/Db, D#/Eb, F#/Gb, G#/Ab, A#/Bb) | **숫자** |
| 박자(음가) | 마디 내에서 **박자 크기에 비례하는 너비의 블록** |

### 5-2. 미확정 세부 규칙 (구현 전 반드시 확정)

- 검은 건반 숫자 매핑 (C#=?, D#=?, ...)
- 옥타브 구분 방법 (첨자, 위아래 위치, 색상 등)
- 쉼표 표기
- 점음표, 이음줄, 붙임줄
- 화음(동시 여러 음) 배치 방식
- 박자표, 조표 표기 여부
- 박스 크기 기준값, 폰트 선택

### 5-3. 렌더링 구현 방식

**권장**: Flutter `CustomPainter`

- DeepScore 토큰 리스트를 입력으로 받아 박스+문자 레이아웃 계산
- 성능 문제 발생 시 C++ + OpenCV로 이미지 생성 후 Flutter에 전달하는 방식으로 전환 가능

---

## 6. 전체 구현 순서 (Phase 계획)

```
Phase 0: 기반 작업 (현재)
  0-1. DeepScore 라벨 필드 및 커스텀 표기법 세부 규칙 확정 (문서화)
  0-2. MusicScore 레포 C++ 엔진 코드 상세 분석 → 재사용 가능 모듈 목록 작성

Phase 1: 데이터 생성 파이프라인 (Python) ← 현재 진행 중
  1-1. ✅ music21 무작위 악보 생성 스크립트 작성 (scripts/generate_dataset.py)
  1-2. ✅ MuseScore 렌더링 자동화 (generate_dataset.py 내 render_png)
  1-3. ✅ music21 → DeepScore 라벨 변환 (generate_dataset.py 내 _element_to_tokens)
  1-4. ✅ tokenizer.json vocabulary 생성 (data/tokenizer.json)
  1-5. OK 100쌍 생성 완료 (seed=2026, 1240x1754 px, 150 DPI)
       MuseScore 4 경로: C:\Program Files\MuseScore 4\bin\MuseScore4.exe
  1-6. 실사 사진 증강 스크립트 작성 (augment.py)

Phase 2: 모델 학습 (PyTorch, RTX 3080)
  2-1. Segnet 아키텍처 구현 및 단독 학습
  2-2. Encoder 아키텍처 구현 및 Segnet 연결 학습
  2-3. Decoder (Transformer) 구현 및 end-to-end 학습
  2-4. 검증: compare 스크립트로 DeepScore 기준 정확도 측정

Phase 3: 모델 변환 및 양자화
  3-1. PyTorch → ONNX 변환
  3-2. ONNX → TFLite 변환
  3-3. Segnet + Encoder INT8 PTQ 적용
  3-4. Decoder INT8 실험 (정확도 손실 허용 범위 내인지 확인)
  3-5. Desktop C++ 환경에서 변환 모델 정확도 재검증

Phase 4: C++ 추론 엔진 (Dart FFI용)
  4-1. OK omr_engine.h — public C API (Dart FFI 진입점)
  4-2. OK preprocessor.cpp — autocrop + resize(1920) + CLAHE
  4-3. OK segnet_runner.cpp — 320x320 패치, 50% 오버랩, TFLite NCHW
  4-4. OK staff_detector.cpp — horizontal projection + peak NMS + 5-line grouping
  4-5. OK staff_canvas.cpp — crop + scale to 256px + tile 1280px (64px overlap)
  4-6. OK encoder_runner.cpp — TFLite, normalize mean=0.7931 std=0.1738
  4-7. OK decoder_runner.cpp — greedy KV-cache decoding, MAX_SEQ=608
  4-8. OK token_parser.cpp — tokenizer.json 로드, ID<->문자열 변환
  4-9. OK omr_engine.cpp — 전체 파이프라인 + Dart FFI C API
  4-10. OK CMakeLists.txt — OpenCV + TFLite, Android/desktop 공용
  4-11. OK test/test_engine.cpp — standalone CLI 테스트
  위치: omr/engine/
  빌드: 모델 학습(Phase 2) 완료 후 TFLITE_ROOT 지정하여 cmake 빌드

  디렉토리 구조:
    omr/
      engine/                 C++ 추론 엔진
        include/              헤더 파일 (7개)
        src/                  구현 파일 (8개)
        test/                 CLI 테스트
        CMakeLists.txt
      data_gen/               데이터 생성 스크립트
        generate_dataset.py
        requirements.txt

Phase 5: Flutter 통합
  5-1. lib/omr_service.dart Dart FFI로 재작성
  5-2. 기존 MethodChannel 관련 코드 제거 (Kotlin JNI, flutter_jni.cpp)
  5-3. Flutter camera 패키지 도입, 실시간 캡처 파이프라인 구현
  5-4. Riverpod 상태 관리 도입

Phase 6: 커스텀 표기법 렌더링
  6-1. DeepScore 토큰 → 커스텀 표기 레이아웃 변환 로직 구현
  6-2. Flutter CustomPainter로 렌더링 구현
  6-3. 실시간 카메라 오버레이 연결

Phase 7: 멀티플랫폼
  7-1. iOS: CMake/Xcode 빌드 설정, 카메라 권한, 검증
  7-2. 웹: Python FastAPI 백엔드 서버 구축, Flutter Web HTTP 연동
```

---

## 7. 기술 스택 요약

| 항목 | 현재 | 목표 |
|------|------|------|
| OMR 모델 가중치 | 기존 HOMR 가중치 | **직접 학습 (PyTorch, RTX 3080)** |
| 학습 데이터 | — | **music21 생성 + 실사 사진** |
| 라벨 형식 | MusicXML | **DeepScore 토큰 시퀀스** |
| 모델 포맷 (추론) | TFLite + ONNX | **TFLite INT8 통일** |
| 추론 엔진 | MusicScore C++ JNI | **신규 C++ (OpenCV + TFLite C++ API), Dart FFI** |
| Flutter 브리지 | MethodChannel | **Dart FFI** |
| 이미지 입력 | 갤러리 선택 | **실시간 카메라 + 갤러리** |
| 출력 | MusicXML 텍스트 | **커스텀 표기법 이미지** |
| iOS | 스텁 | **Dart FFI 기반 완전 구현** |
| 웹 | 없음 | **서버 사이드 (FastAPI) + Flutter Web** |
| 상태 관리 | setState() | **Riverpod** |

---

## 8. 추론 엔진 설계 요약 (omr/engine/)

### 8-1. 파이프라인 단계 및 핵심 수치

```
입력 이미지 (스마트폰 촬영 JPEG/PNG)
  [1] Preprocessor : autocrop -> resize 1920px -> CLAHE(clip=1.0, tile=8)
  [2] SegnetRunner : [1,3,320,320] 50%오버랩 패치 -> 5개 확률맵 (TFLite)
  [3] StaffDetector: 수평투영 -> 피크NMS -> 5-line 그룹화 (unit 11~60px)
  [4] StaffCanvas  : crop(margin=2u) -> scale 256px -> tile 1280px(64px overlap)
  [5] EncoderRunner: [1,1,256,1280] norm=(px/255-0.7931)/0.1738 -> [1,320,512]
  [6] DecoderRunner: greedy KV-cache (8layer,8head,64dim) MAX=608 토큰
  [7] TokenParser  : ID <-> DeepScore 문자열, tokenizer.json 기반
```

### 8-2. Dart FFI 공개 API (`omr_engine.h`)

```c
OmrEngineHandle omr_create(seg, enc, dec, tok paths);
OmrResultC*     omr_process(handle, image_bytes, len);
void            omr_free_result(result);
void            omr_destroy(handle);
```

### 8-3. 사용 라이브러리 라이선스

| 라이브러리 | 라이선스 | 상업 이용 |
|-----------|---------|---------|
| OpenCV | Apache 2.0 | OK |
| TensorFlow Lite | Apache 2.0 | OK |
| 알고리즘 (CLAHE, 수평투영, KV-cache Transformer) | 공개 논문 | OK |

---

## 9. 학습 데이터 요구량 및 권장 순서

### 9-1. 최소/권장 데이터량

| 모듈 | 최소 | 권장 | 현재 상태 |
|------|------|------|---------|
| Segnet (픽셀 분류) | 2,000 악보 | 20,000+ | 100 생성 완료 |
| Encoder-Decoder (토큰 시퀀스) | 5,000 staff | 50,000+ | 100 악보 x ~6 staff |
| 실사 사진 (domain gap 보정) | 200 장 | 1,000+ | 0 |

증강(밝기/노이즈/원근/블러) x8 적용 시: 1,000 악보 -> 8,000 유효 이미지

### 9-2. 권장 학습 순서

```
1. generate_dataset.py -n 5000  (5,000 악보 생성)
2. augment.py 실행 -> x8 = 40,000 이미지
3. Segnet 단독 사전학습 (~50 epoch, Focal Loss)
4. Encoder+Decoder end-to-end (~100 epoch, Teacher Forcing)
5. 실사 사진 fine-tuning (~20 epoch)
6. PyTorch -> ONNX -> TFLite INT8 PTQ 변환
7. omr/engine/ 에 .tflite 모델 배포 후 CMake 빌드
```

### 9-3. 권장 모델 아키텍처 (상업 이용 가능)

| 모듈 | 아키텍처 | 라이선스 |
|------|---------|---------|
| Segnet | MobileNetV3 + U-Net decoder | Apache 2.0 |
| Encoder | ConvNeXt-Tiny | MIT |
| Decoder | Transformer (Vaswani 2017) | 공개 알고리즘 |

---

## 11. 보류/추후 논의 항목

- **커스텀 표기법 세부 규칙** (섹션 5-2) — Phase 6 착수 전에 반드시 확정
- **DeepScore 라벨 필드 확정** — Phase 1-3 착수 전에 반드시 확정
- **Decoder 양자화 허용 손실 기준** — 실험 후 결정
- **웹 인프라 선택** — 자체 서버(RTX 3080 겸용) vs 클라우드 (AWS, GCP)
- **카메라 처리 방식** — 연속 프레임 스트리밍 vs 버튼 트리거 스냅샷
- **학습 Python 코드 레포 위치** — 현재 Flutter 레포에 포함 vs 별도 레포
