# OMR Team Manual

This manual covers the complete workflow: generating training data, training models, building the C++ inference engine, and integrating into Flutter.

---

## Directory Overview

```
omr/
  data_gen/
    generate_dataset.py   # Music21 score generator + MuseScore PNG renderer
    requirements.txt      # Python dependencies
  engine/
    include/              # Public headers (omr_engine.h, types.hpp, ...)
    src/                  # C++ source files
    test/                 # Desktop smoke test (test_engine.cpp)
    CMakeLists.txt        # Builds shared library + desktop test binary
  utils/
    compare_musicxml.py   # Compare two MusicXML files
    inspect_tflite.py     # Print TFLite model I/O tensor shapes
    inspect_decoder.py    # Trace decoder token-by-token output
  MANUAL.md               # This file

data/
  train/                  # Generated training sets (num1..numN)
  test/                   # Sample images for engine smoke testing
  tokenizer.json          # 978-token vocabulary (do not edit)
  tokenizer_reference.txt # Human-readable token listing
```

---

## Prerequisites

### System
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Data generation, training |
| MuseScore 4 | 4.x | Rendering MusicXML to PNG |
| CUDA | 11.8+ | GPU training (RTX 3080) |
| PyTorch | 2.2+ | Model training |
| CMake | 3.22+ | C++ engine build |
| OpenCV | 4.9+ | Preprocessing (desktop build) |
| TFLite | 2.16+ | Inference runtime |
| Android NDK | r26+ | Cross-compile for Android |

### Windows: MuseScore 4 path
```
C:\Program Files\MuseScore 4\bin\MuseScore4.exe
```

### Linux / WSL
```bash
sudo apt install musescore4   # or download AppImage
```

---

## Step 0: Clone and Install Python Dependencies

```bash
git clone <repo-url>
cd musicscore_flutter

pip install -r omr/data_gen/requirements.txt
```

`requirements.txt` contents:
```
music21>=9.1
numpy>=1.26
Pillow>=10.0
```

---

## Step 1: Generate Training Data

### Basic usage (100 scores)
```bash
python omr/data_gen/generate_dataset.py \
  --count 100 \
  --out_dir data/train \
  --musescore "C:\Program Files\MuseScore 4\bin\MuseScore4.exe" \
  --seed 42
```

### Full training set (recommended: 5,000+, minimum: 2,000)
```bash
python omr/data_gen/generate_dataset.py \
  --count 5000 \
  --out_dir data/train \
  --musescore "C:\Program Files\MuseScore 4\bin\MuseScore4.exe" \
  --seed 42
```

Each score generates three files:
```
data/train/
  num1.musicxml   # MusicXML source
  num1.png        # 1240x1754 px rendered sheet (A4 at 150 DPI)
  num1.json       # DeepScore token label sequence
```

### Verify output
```bash
python -c "
import json, os
files = [f for f in os.listdir('data/train') if f.endswith('.json')]
print(f'{len(files)} label files')
with open(f'data/train/{files[0]}') as f:
    d = json.load(f)
print('Sample tokens:', d['tokens'][:10])
"
```

---

## Step 2: Train the Models

### Prerequisites

```bash
pip install -r omr/training/requirements.txt
# torch>=2.2.0, torchvision>=0.17.0, numpy>=1.26, opencv-python>=4.9, tensorboard>=2.15
```

CUDA 11.8+ required. Training is designed for RTX 3080 (10 GB VRAM).

---

### Model architecture summary

| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| SegNet | 4-level U-Net (base_ch=32) | [B,1,320,320] grayscale | [B,6,320,320] class logits |
| Encoder | 8-stage strided CNN | [B,1,256,1280] grayscale tile | [B,320,512] latent sequence |
| Decoder | 8-layer pre-LN Transformer | encoder output + prev tokens | 978-token logits |

---

### Phase 1 — SegNet pretraining

Trains the segmentation network from weak pixel labels (image-processing-based).
Recommended: ≥2,000 scores (see Step 1).

```bash
python omr/training/train.py \
  --phase 1 \
  --data_dir data/train \
  --tokenizer data/tokenizer.json \
  --out_dir models/ \
  --epochs 50 \
  --batch 16 \
  --lr 3e-4 \
  --device cuda
```

Output: `models/segnet_best.pt`, `models/segnet_last.pt`

Monitor with TensorBoard:
```bash
tensorboard --logdir models/
```

---

### Phase 2 — Encoder + Decoder training

Uses the SegNet checkpoint from Phase 1. The encoder is frozen for the first
`epochs // 5` epochs, then unfrozen at `lr / 3`.

```bash
python omr/training/train.py \
  --phase 2 \
  --data_dir data/train \
  --tokenizer data/tokenizer.json \
  --out_dir models/ \
  --segnet_ckpt models/segnet_best.pt \
  --epochs 100 \
  --batch 8 \
  --lr 1e-4 \
  --device cuda
```

Output: `models/seq2seq_best.pt` (best validation TER), `models/seq2seq_last.pt`

---

### Phase 3 — End-to-end fine-tuning

Jointly fine-tunes all three modules at a lower learning rate.

```bash
python omr/training/train.py \
  --phase 3 \
  --data_dir data/train \
  --tokenizer data/tokenizer.json \
  --out_dir models/ \
  --resume models/seq2seq_best.pt \
  --epochs 30 \
  --batch 4 \
  --lr 3e-5 \
  --device cuda
```

---

### Resume an interrupted run

Pass `--resume <checkpoint>` to any phase to continue from the last saved epoch:

```bash
python omr/training/train.py --phase 2 ... --resume models/seq2seq_last.pt
```

---

### Evaluate mid-training accuracy

After any phase, run the Python evaluator against a held-out test set:

```bash
python omr/utils/evaluate.py \
  --test_dir data/test \
  --segnet models/segnet_best.pt \
  --seq2seq models/seq2seq_best.pt \
  --tokenizer data/tokenizer.json \
  --threshold 0.90 \
  --report models/eval_report.csv
```

Exit code 0 = PASS (mean accuracy ≥ threshold), 1 = FAIL.

---

### Expected training data requirements

| Quality | Score count | Approx GPU time (RTX 3080) |
|---------|------------|---------------------------|
| Minimum viable | 2,000 | ~6 h (Phase 1+2) |
| Recommended | 10,000 | ~24 h |
| Production | 20,000+ | ~48 h |

---

### 누적 학습: 데이터를 추가할 때마다 반복하는 절차

학습을 여러 번 반복해도 이전 결과가 누적되도록 하려면 아래 순서를 따릅니다.

#### 핵심 원칙

| 항목 | 방법 |
|------|------|
| 데이터 관리 | `data/train`에 새 데이터를 **추가(append)** — 이전 데이터를 지우지 않음 |
| 체크포인트 | 새 학습 전에 현재 `.pt` 파일을 **버전 이름으로 복사**해 보존 |
| `--resume` | 이전 학습의 최적 체크포인트 하나만 지정 (여러 개 나열 불필요) |
| 학습률 | fine-tuning 시 초기 학습률의 1/3~1/10 수준으로 낮춤 (아래 설명 참조) |

#### Fine-tuning 시 학습률을 낮추는 이유

처음 학습 후 체크포인트는 이미 loss 최솟값 근처에 있습니다.
같은 lr로 계속 학습하면 그 최솟값을 **지나쳐** 이전에 학습한 표현을 덮어쓸 수 있습니다.
lr을 낮추면 새 데이터에 대한 기울기 방향으로 **작은 보정 스텝**만 밟아 기존 지식을 보존합니다.

```
초기 Phase 2 학습 : --lr 1e-4
1차 fine-tuning   : --lr 3e-5   (약 3배 감소)
2차 fine-tuning   : --lr 1e-5   (다시 3배 감소, 필요 시)
```

#### 1회차 (최초 학습)

```bash
# 1-1. 데이터 생성 (예: 2,000장)
python omr/data_gen/generate_dataset.py \
  --count 2000 --out_dir data/train \
  --musescore "C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

# 1-2. Phase 1 (SegNet)
python omr/training/train.py --phase 1 \
  --data_dir data/train --tokenizer data/tokenizer.json \
  --out_dir models/ --epochs 50 --batch 16 --lr 3e-4

# 1-3. Phase 2 (Encoder + Decoder)
python omr/training/train.py --phase 2 \
  --data_dir data/train --tokenizer data/tokenizer.json \
  --out_dir models/ --segnet_ckpt models/segnet_best.pt \
  --epochs 100 --batch 8 --lr 1e-4

# 1-4. Phase 3 (end-to-end fine-tune)
python omr/training/train.py --phase 3 \
  --data_dir data/train --tokenizer data/tokenizer.json \
  --out_dir models/ --resume models/seq2seq_best.pt \
  --epochs 30 --batch 4 --lr 3e-5

# 1-5. 체크포인트 버전 저장 (롤백 보존용)
cp models/segnet_best.pt   models/segnet_best_v1.pt
cp models/seq2seq_best.pt  models/seq2seq_best_v1.pt

# 1-6. TFLite export
python omr/training/export_tflite.py \
  --segnet models/segnet_best.pt --seq2seq models/seq2seq_best.pt \
  --data_dir data/train --tokenizer data/tokenizer.json \
  --out_dir assets/ --version v1
```

#### 2회차 (데이터 추가 후 계속 학습)

```bash
# 2-1. 새 데이터 추가 (기존 data/train에 누적)
python omr/data_gen/generate_dataset.py \
  --count 3000 --out_dir data/train \
  --musescore "C:\Program Files\MuseScore 4\bin\MuseScore4.exe" \
  --seed 100    # seed를 바꿔 새로운 악보 생성

# 2-2. v1 체크포인트에서 이어서 fine-tuning (lr 낮춤)
python omr/training/train.py --phase 2 \
  --data_dir data/train --tokenizer data/tokenizer.json \
  --out_dir models/ --segnet_ckpt models/segnet_best_v1.pt \
  --resume models/seq2seq_best_v1.pt \
  --epochs 50 --batch 8 --lr 3e-5       # lr 낮춤 (1e-4 → 3e-5)

python omr/training/train.py --phase 3 \
  --data_dir data/train --tokenizer data/tokenizer.json \
  --out_dir models/ --resume models/seq2seq_best.pt \
  --epochs 20 --batch 4 --lr 1e-5

# 2-3. 체크포인트 버전 저장
cp models/segnet_best.pt   models/segnet_best_v2.pt
cp models/seq2seq_best.pt  models/seq2seq_best_v2.pt

# 2-4. TFLite export (v2)
python omr/training/export_tflite.py \
  --segnet models/segnet_best.pt --seq2seq models/seq2seq_best.pt \
  --data_dir data/train --tokenizer data/tokenizer.json \
  --out_dir assets/ --version v2
```

> **요약**: 이전 `.pt` 파일을 여러 개 명령어에 나열하지 않습니다.  
> `--resume`에는 **직전 최적 체크포인트 하나**만 지정하고,  
> `data/train`에 데이터를 계속 쌓아가면 모델이 누적 데이터를 학습합니다.

---

## Step 2.5: Export to TFLite

Phase 3 학습이 끝난 후 실행합니다. PyTorch `.pt` → ONNX → TFLite INT8 변환.

### 의존성 설치 (최초 1회)

```bash
pip install onnx onnxsim onnx2tf tensorflow
```

> `onnx2tf`는 tensorflow와 함께 설치됩니다. CUDA 환경에서는 `tensorflow-gpu` 대신
> `tensorflow` (CPU)로도 export 가능합니다.

### 기본 실행 (INT8, version v1)

```bash
python omr/training/export_tflite.py \
  --segnet    models/segnet_best.pt \
  --seq2seq   models/seq2seq_best.pt \
  --data_dir  data/train \
  --tokenizer data/tokenizer.json \
  --out_dir   assets/ \
  --version   v1
```

### 출력 파일

```
assets/
  segnet_INT8_v1.tflite    ← 버전 보존 (삭제 금지)
  encoder_INT8_v1.tflite
  decoder_INT8_v1.tflite   ← FP32 (아래 참조)
  tokenizer_v1.json
  segnet_INT8.tflite       ← latest (다음 export 시 덮어써짐)
  encoder_INT8.tflite
  decoder_INT8.tflite
  tokenizer.json
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--version v2` | 버전 태그 변경 |
| `--calib_n 200` | INT8 캘리브레이션 이미지 수 (기본 100) |
| `--no_quantize` | 모든 모델을 FP32로 export |
| `--quantize_decoder` | Decoder도 INT8 양자화 (정확도 손실 위험, 실험용) |

### Decoder가 FP32인 이유

Encoder와 SegNet은 INT8 양자화 후 정확도 손실이 거의 없습니다.  
Decoder(Transformer)는 Softmax·LayerNorm 등의 연산이 양자화 오차에 민감해
INT8로 바꾸면 token error rate가 크게 올라갈 수 있습니다.  
먼저 FP32로 배포한 뒤, `--quantize_decoder`로 실험해 손실 허용 범위를 확인하세요.

### 이전 버전으로 롤백

```bash
# v1으로 되돌리기
cp assets/segnet_INT8_v1.tflite  assets/segnet_INT8.tflite
cp assets/encoder_INT8_v1.tflite assets/encoder_INT8.tflite
cp assets/decoder_INT8_v1.tflite assets/decoder_INT8.tflite
cp assets/tokenizer_v1.json      assets/tokenizer.json
```

---

## Step 3: Build the C++ Inference Engine

### Desktop (Linux / WSL) — for testing

```bash
cd omr/engine

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_ROOT=/path/to/tflite \
  -DOpenCV_DIR=/usr/include/opencv4

cmake --build build -j$(nproc)
```

This produces:
- `build/libomr_engine.so` — shared library (Dart FFI target)
- `build/omr_test` — desktop smoke-test binary

### Android (cross-compile from WSL)

```bash
cd omr/engine

cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_ROOT=/path/to/tflite-android \
  -DOPENCV_ANDROID_SDK=/path/to/OpenCV-android-sdk

cmake --build build-android -j$(nproc)
```

Output: `build-android/libomr_engine.so` → copy to `android/app/src/main/jniLibs/arm64-v8a/`

### Required library paths

**TFLite (desktop)**
- Header root: `$TFLITE_ROOT/include/` (must contain `tensorflow/lite/interpreter.h`)
- Library: `$TFLITE_ROOT/lib/libtensorflow-lite.a`

Download prebuilt: [github.com/tensorflow/tensorflow releases](https://github.com/tensorflow/tensorflow)

**TFLite (Android)**
- Library: `$TFLITE_ROOT/lib/arm64-v8a/libtensorflowlite.so`

**OpenCV (desktop)**
```bash
sudo apt install libopencv-dev   # Ubuntu/Debian
brew install opencv              # macOS
```

---

## Step 4: Run the Desktop Smoke Test

```bash
./omr/engine/build/omr_test \
  data/test/num2.png \
  assets/models/segnet.tflite \
  assets/models/encoder.tflite \
  assets/models/decoder.tflite \
  data/tokenizer.json
```

Expected output:
```
Loading engine...
Engine ready.
Reading image: data/test/num2.png
Running inference...
Decoded 47 tokens:
  [   0]    4  clef-G
  [   1]    7  key-C
  [   2]   20  time-4/4
  [   3]   43  note-C4-1/4
  ...
```

### Utility scripts

```bash
# Inspect TFLite model tensor shapes
python omr/utils/inspect_tflite.py assets/models/encoder.tflite

# Trace decoder step-by-step
python omr/utils/inspect_decoder.py \
  assets/models/decoder.tflite \
  data/tokenizer.json

# Compare two MusicXML files
python omr/utils/compare_musicxml.py data/train/num1.musicxml data/train/num2.musicxml
```

---

## Step 5: Flutter Integration via Dart FFI

Place the built `.so` files in the Flutter asset paths:

```
android/app/src/main/jniLibs/
  arm64-v8a/libomr_engine.so
  x86_64/libomr_engine.so    # emulator
ios/
  Frameworks/omr_engine.framework/  # (iOS build TBD)
```

Place TFLite model files:
```
assets/models/
  segnet.tflite
  encoder.tflite
  decoder.tflite
```

Register in `pubspec.yaml`:
```yaml
flutter:
  assets:
    - assets/models/segnet.tflite
    - assets/models/encoder.tflite
    - assets/models/decoder.tflite
    - data/tokenizer.json
```

The Dart FFI bindings will call `omr_create`, `omr_process`, `omr_free_result`, `omr_destroy` declared in `omr/engine/include/omr_engine.h`.

---

## Token Format Reference

See `data/tokenizer_reference.txt` for the full 978-token vocabulary.

Key token ranges:
| Range | Category |
|-------|---------|
| 0 | `<PAD>` |
| 1 | `<SOS>` |
| 2 | `<EOS>` |
| 3 | `<UNK>` |
| 4-6 | Clef (G, F, C) |
| 7-19 | Key signatures |
| 20-27 | Time signatures |
| 28-37 | Rests |
| 38-42 | Barlines |
| 43-530 | Notes (`note-{pitch}-{duration}`) |
| 531-977 | Chord tones (`chord-{pitch}`) |

Duration values: `1/1 3/4 1/2 3/8 1/4 3/16 1/8 3/32 1/16 1/32`
(fraction of a whole note)

---

## Troubleshooting

### MuseScore renders blank PNGs
- Make sure MuseScore 4 is installed, not MuseScore 3.
- On Windows, use the full path: `"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"`.
- Check that MuseScore exits with code 0: `echo $?` after running it manually.

### CMake cannot find TFLite
```
TFLite library not found. Set -DTFLITE_ROOT=<path> when invoking CMake.
```
Ensure `$TFLITE_ROOT/lib/libtensorflow-lite.a` (desktop) or `libtensorflowlite.so` (Android) exists.

### `omr_create` returns null at runtime
- Model paths are wrong — verify with `inspect_tflite.py`.
- Tokenizer JSON is empty or corrupt — check `data/tokenizer.json` is non-empty.

### Android: `UnsatisfiedLinkError`
- Verify ABI matches: `arm64-v8a` for most modern devices.
- Run `adb shell cat /proc/cpuinfo | grep -i arch` to confirm.
