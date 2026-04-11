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

> Training scripts will be added in a future phase. The sections below describe the intended workflow.

### Model architecture summary
| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| SegNet | MobileNetV3 + U-Net decoder | 320x320 grayscale | 6-class segmentation map |
| Encoder | ConvNeXt-Tiny | 256x1280 grayscale tile | seq_len x 512 latent |
| Decoder | 8-layer Transformer | encoder output + prev tokens | 978-token logits |

### Expected training data requirements
| Quality | Score count | Approximate GPU time (RTX 3080) |
|---------|------------|-------------------------------|
| Minimum viable | 2,000 | ~4 hours |
| Recommended | 10,000 | ~20 hours |
| Production | 20,000+ | ~48 hours |

### Training pipeline (planned)

```bash
# 1. Train segnet
python omr/training/train_segnet.py \
  --data_dir data/train \
  --epochs 50 \
  --batch 16 \
  --out models/segnet.pt

# 2. Train encoder + decoder jointly
python omr/training/train_seq2seq.py \
  --data_dir data/train \
  --segnet models/segnet.pt \
  --epochs 100 \
  --batch 8 \
  --out_dir models/

# 3. Convert to TFLite INT8
python omr/training/export_tflite.py \
  --segnet models/segnet.pt \
  --encoder models/encoder.pt \
  --decoder models/decoder.pt \
  --tokenizer data/tokenizer.json \
  --out_dir assets/models/
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
