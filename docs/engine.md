# engine.md — C++ 추론 엔진

> 관련 코드: `omr/engine/`, `omr/training/export_tflite.py`  
> 빌드 전제: Phase 2 (모델 학습) 완료 후 TFLITE_ROOT 지정하여 cmake 빌드

---

## 1. 현재 엔진 구조 (참조용 — MusicScore 레포)

```
입력 이미지
  → OpenCV 전처리 (crop, resize, denoise)
  → TFLite segnet_308_int8.tflite     (픽셀 분류)
  → TFLite encoder_331_int8.tflite    (특징 인코딩)
  → ONNX Runtime decoder_331_int8.onnx (토큰 디코딩)
  → tr_omr_parser → rhythm/accidental rules
  → MusicXML 출력
```

이 구조를 유지하되 **모델 가중치를 처음부터 직접 학습**하고, 브리지를 JNI → Dart FFI로 교체한다.

---

## 2. 신규 엔진 설계 방향

| 항목 | 기존 MusicScore 엔진 | 신규 엔진 |
|---|---|---|
| 브리지 | JNI (Kotlin → C++) | **Dart FFI** (Dart → C++ 직접) |
| 입력 포맷 | Android ByteArray | 범용 `uint8_t*` 버퍼 |
| 출력 포맷 | MusicXML 문자열 | **DeepScore 토큰 시퀀스** |
| 모델 로드 | Android AssetManager | 파일 경로 기반 (플랫폼 독립) |
| ONNX Runtime | 사용 | **제거** (Decoder도 TFLite로 통일) |

### 엔진 내부 구조

```
OmrEngine (C++)  [omr/engine/]
  ├── Preprocessor          (OpenCV)
  │     PerspectiveCorrector + autocrop + resize 1920px + CLAHE + bilateral
  ├── SegnetRunner           (TFLite)
  │     320×320 패치, 50% 오버랩, 6-class 확률맵
  ├── StaffDetector
  │     수평 투영 → 피크 NMS → 5-line 그룹화
  ├── PageDewarper
  │     staff_mask 기반 수직 변위장 → cv::remap
  ├── StaffCanvas
  │     crop + scale 256px + tile 1280px (64px 오버랩)
  ├── EncoderRunner          (TFLite)
  │     norm=(px/255−0.7931)/0.1738 → [1, 320, 512]
  ├── DecoderRunner          (TFLite)
  │     greedy KV-cache (8layer, 8head, 64dim), MAX=608 토큰
  ├── TokenParser
  │     tokenizer.json 기반 ID ↔ DeepScore 문자열
  └── Public C API           (Dart FFI 진입점)
        omr_create(seg, enc, dec, tok paths)
        omr_process(handle, image_bytes, len) → OmrResultC*
        omr_free_result(result)
        omr_destroy(handle)
```

### 구현 완료 파일 목록 (`omr/engine/`)

| 파일 | 상태 |
|---|---|
| `include/omr_engine.h` | ✅ |
| `src/preprocessor.cpp` | ✅ |
| `src/perspective_corrector.cpp` | ✅ |
| `src/noise_filter.cpp` | ✅ |
| `src/segnet_runner.cpp` | ✅ |
| `src/staff_detector.cpp` | ✅ |
| `src/staff_canvas.cpp` | ✅ |
| `src/encoder_runner.cpp` | ✅ |
| `src/decoder_runner.cpp` | ✅ |
| `src/token_parser.cpp` | ✅ |
| `src/omr_engine.cpp` + `page_dewarper.cpp` | ✅ |
| `CMakeLists.txt` | ✅ |
| `test/test_engine.cpp` | ✅ |
| `test/eval_engine.cpp` | ✅ |

---

## 3. 파이프라인 단계별 핵심 수치

```
[1] Preprocessor
      PerspectiveCorrector: Otsu → 최대 윤곽 → 4-코너 쿼드 → warpPerspective
                            (4코너 미검출 시 HoughLinesP → 중앙값 각도 deskew)
      autocrop → resize 1920px → CLAHE(clip=1.0, tile=8)
      NoiseFilter: flat-field 조도 정규화 → bilateral(d=9, σc=20, σs=7)
[2] SegnetRunner : [1,3,320,320] 50% 오버랩 패치 → 6-class 확률맵 (TFLite)
[3] StaffDetector: 수평 투영 → 피크 NMS → 5-line 그룹화 (unit 11~60px)
[4] PageDewarper : staff_mask → staff-line 추적 → 수직 변위장 → cv::remap
[5] StaffCanvas  : crop(margin=2u) → scale 256px → tile 1280px(64px 오버랩)
[6] EncoderRunner: [1,1,256,1280] norm → [1,320,512]
[7] DecoderRunner: greedy KV-cache, MAX=608 토큰
[8] TokenParser  : ID ↔ DeepScore 문자열, tokenizer.json
```

---

## 4. INT8 양자화 (Phase 3)

### 변환 방법

```
PyTorch 모델 (.pt)
  → ONNX export (torch.onnx.export)
  → TFLite 변환 (onnx2tf → TFLite converter)
  → Representative dataset으로 INT8 calibration
  → segnet_INT8.tflite, encoder_INT8.tflite 생성
```

스크립트: `omr/training/export_tflite.py`
- `--version` 태그로 버전별 스냅샷 저장 (`segnet_INT8_v1.tflite` 등)
- `--quantize_decoder` 플래그로 Decoder INT8 실험 가능

### 양자화 대상별 전략

| 모듈 | 방식 | 비고 |
|---|---|---|
| SegNet | INT8 PTQ | 충분히 안전 |
| Encoder | INT8 PTQ | 충분히 안전 |
| Decoder | FP16 기본 → INT8 실험 | 정확도 손실 주의 |

### 모델 크기 목표

| 모듈 | FP32 추정 | INT8 목표 |
|---|---|---|
| SegNet | ~10 MB | ~3 MB |
| Encoder | ~5 MB | ~2 MB |
| Decoder | ~20 MB | ~5 MB (FP16) |
| **합계** | **~35 MB** | **~10 MB** |

---

## 5. 추론 속도 최적화 (인식률 90%+ 확보 이후 순서대로 적용)

### 적용 순서

```
Step 1. Hardware Delegate 활성화        ← 가중치 변경 없음, 즉시 효과
Step 2. Decoder KV-Cache 구현          ← 가장 큰 속도 향상
Step 3. Decoder INT8 양자화 실험       ← KV-Cache 이후
Step 4. SegNet 패치 오버랩 감소        ← eval로 정확도 확인 후
Step 5. 모델 경량화 (필요 시)          ← 마지막 수단
```

### Step 1 — Hardware Delegate

| Delegate | OS | 속도 향상 |
|---|---|---|
| NNAPI | Android API 27+ | 2~5× |
| GPU (OpenCL) | Android | 2~4× |
| Core ML | iOS | 2~6× |
| XNNPACK | 전체 (기본 내장) | 1.5~2× |

수정 파일: `omr/engine/src/segnet_runner.cpp`, `encoder_runner.cpp`, `decoder_runner.cpp`

### Step 2 — Decoder KV-Cache

```
현재 (O(T²)):   스텝 50 → [SOS, t1, ..., t49] 전체 재연산
KV-Cache (O(1)): 스텝 50 → [t49] 만 연산, 이전 K/V 재사용
```

- `model.py`: `DecoderStepWithCache` 모듈 추가
- `export_tflite.py`: 해당 모듈로 decoder export 교체
- `decoder_runner.cpp`: 새 TFLite 인터페이스에 맞게 수정
- **train.py는 수정 불필요** (학습은 teacher-forcing 유지)

### Step 4 — SegNet 패치 오버랩 감소

```
현재 (STRIDE=160, 50% 오버랩): 1920px → 약 11 패치
조정 (STRIDE=240, 25% 오버랩): 1920px → 약  7 패치 (35% 감소)
```

`omr/engine/src/segnet_runner.cpp` stride 값 조정. 적용 전 `omr_eval`로 정확도 비교 필수.

### Step 5 — 모델 경량화 (재학습 필요)

| 방법 | 효과 |
|---|---|
| Knowledge Distillation | 크기 1/2~1/4 |
| Decoder 레이어 8 → 4~6 | 속도 1.3~2× |
| SegNet base_ch 32 → 16 | 크기 1/4 |

---

## 6. 스마트폰 최적화 (인식률 90%+ 확보 이후)

### 성능 프로파일링 도구

| 플랫폼 | 도구 | 측정 항목 |
|---|---|---|
| Android | Android Studio Profiler | CPU·GPU·메모리·발열 |
| Android | TFLite Benchmark Tool | 모듈별 추론 시간 |
| iOS | Xcode Instruments | CPU/GPU usage, 메모리 |
| iOS | Core ML Performance Report | 레이어별 실행 시간 |

### 배포 전 체크리스트

- [ ] 전체 모델 합계 크기 < 50 MB
- [ ] 최신 안드로이드 기기 추론 3초 이내
- [ ] 저사양 기기(Snapdragon 665급) 10초 이내
- [ ] NNAPI/GPU delegate 활성화 상태에서 TER 동일 수준 확인
- [ ] 연속 3회 추론 시 발열 성능 저하 < 20%

### 사용 라이브러리 라이선스

| 라이브러리 | 라이선스 | 상업 이용 |
|---|---|---|
| OpenCV | Apache 2.0 | ✅ |
| TensorFlow Lite | Apache 2.0 | ✅ |
| CLAHE, KV-Cache Transformer 등 알고리즘 | 공개 논문 | ✅ |
