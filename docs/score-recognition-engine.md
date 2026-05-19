# score-recognition-engine.md

> 담당 에이전트: `score-recognition-engine`  
> 관련 코드: `ml/omr/engine/`, `ml/omr/training/export_tflite.py`  
> 빌드 전제: Phase 2 (모델 학습) 완료 후 모델 가중치 교체하여 cmake 빌드

## 역할

악보 이미지에서 음악 기호를 인식하는 C++ 추론 엔진을 설계·유지한다.  
OMR 파이프라인의 각 단계(전처리→세그멘테이션→보표 탐지→인코더→디코더)를 관리하고  
INT8 양자화 및 속도 최적화를 담당한다.

---

## 엔진 내부 구조 (`ml/omr/engine/`)

```
OmrEngine (C++)
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

### 구현 완료 파일 목록

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

## 파이프라인 단계별 핵심 수치

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

## INT8 양자화 (Phase 3 — Round 1 학습 완료 후 실행)

```
PyTorch 모델 (.pt)
  → ONNX export (torch.onnx.export)
  → TFLite 변환 (onnx2tf → TFLite converter)
  → Representative dataset으로 INT8 calibration
  → segnet_INT8.tflite, encoder_INT8.tflite 생성
```

스크립트: `ml/omr/training/export_tflite.py`  
`--version` 태그로 버전별 스냅샷 저장 (`segnet_INT8_v1.tflite` 등)

### 양자화 전략

| 모듈 | 방식 | 비고 |
|---|---|---|
| SegNet | INT8 PTQ | 안전 |
| Encoder | INT8 PTQ | 안전 |
| Decoder | FP16 기본 → INT8 실험 | 정확도 손실 주의 |

### 모델 크기 목표

| 모듈 | FP32 추정 | INT8 목표 |
|---|---|---|
| SegNet | ~10 MB | ~3 MB |
| Encoder | ~5 MB | ~2 MB |
| Decoder | ~20 MB | ~5 MB (FP16) |
| **합계** | **~35 MB** | **~10 MB** |

---

## 속도 최적화 (인식률 90%+ 확보 후 순서대로 적용)

```
Step 1. Hardware Delegate 활성화     ← 즉시 효과, 가중치 변경 없음
Step 2. Decoder KV-Cache 구현        ← 가장 큰 속도 향상
Step 3. Decoder INT8 양자화 실험     ← KV-Cache 이후
Step 4. SegNet 패치 오버랩 감소      ← eval로 정확도 확인 후
Step 5. 모델 경량화 (필요 시)        ← 마지막 수단
```

### Step 1 — Hardware Delegate

| Delegate | OS | 속도 향상 |
|---|---|---|
| NNAPI | Android API 27+ | 2~5× |
| GPU (OpenCL) | Android | 2~4× |
| Core ML | iOS | 2~6× |
| XNNPACK | 전체 (기본 내장) | 1.5~2× |

### Step 2 — Decoder KV-Cache

```
현재 (O(T²)):   스텝 50 → [SOS, t1, ..., t49] 전체 재연산
KV-Cache (O(1)): 스텝 50 → [t49] 만 연산, 이전 K/V 재사용
```

수정 대상: `model.py` (DecoderStepWithCache 추가), `export_tflite.py`, `decoder_runner.cpp`

### Step 4 — SegNet 패치 오버랩 감소

```
현재 (STRIDE=160, 50% 오버랩): 1920px → 약 11 패치
조정 (STRIDE=240, 25% 오버랩): 1920px → 약  7 패치 (35% 감소)
```

---

## 배포 전 체크리스트

- [ ] 전체 모델 합계 크기 < 50 MB
- [ ] 최신 안드로이드 기기 추론 3초 이내
- [ ] 저사양 기기(Snapdragon 665급) 10초 이내
- [ ] NNAPI/GPU delegate 활성화 상태에서 TER 동일 수준 확인
- [ ] 연속 3회 추론 시 발열 성능 저하 < 20%
