# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Guidelines


Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

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

## 4. Goal-Driven Execution

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

클로드 md 파일 가이드라인 적용

## Project Overview

Flutter-based Optical Music Recognition (OMR) app that captures sheet music images and converts them to structured data. The OMR engine (`ml/omr/engine/`) is a self-authored C++ pipeline that lives entirely inside this repo — there is no dependency on any sibling repository.

An earlier prototype linked a sibling `MusicScore/` repo (JNI bridge, based on the AGPLv3-licensed `homr` project). That path has been removed: AGPL is incompatible with the planned commercial release, so `ml/omr/engine/` was built from scratch with its own architecture and its own single unified "DeepScore" token vocabulary instead of reusing homr's code, model weights, or its rhythm/pitch/lift/note tokenizer split. See `docs/project-orchestrator.md` ("상업 라이선스 확인 완료") and `project.md` (기술 스택 표) for the decision record.

## Build & Run Commands

### Flutter
```bash
flutter pub get
flutter run                        # run on connected device
flutter build apk                  # Android release APK
flutter build ios                  # iOS (requires macOS)
flutter build web                  # Web (not yet configured; OMR is unavailable on web — see Platform Notes)
```

### Desktop C++ Test Binaries (`ml/omr/engine/`, Linux / WSL / Windows)
```bash
cd ml/omr/engine
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTFLITE_ROOT=/path/to/tflite -DOpenCV_DIR=/usr/include/opencv4
cmake --build build -j$(nproc)

./build/omr_test <image.jpg> <segnet.tflite> <encoder.tflite> <decoder.tflite> <tokenizer.json>
./build/omr_eval <test_dir> <segnet.tflite> <encoder.tflite> <decoder.tflite> <tokenizer.json> [--threshold 0.90] [--report report.csv]
```

### Python Validation
```bash
python ml/omr/utils/compare_musicxml.py <ground_truth.musicxml> <output.xml>
python ml/omr/utils/inspect_tflite.py    # inspect TFLite model tensor shapes
```

### ML Training (Round 3 / grand staff — `round3train/`)
```bash
python round3train/train.py --phase 2 --data_dir <dir> --tokenizer round3train/tokenizer.json \
    --resume <old_ckpt.pt> --resume_tokenizer <old_tokenizer.json>   # only needed when the vocab changed since <old_ckpt.pt>
python round3train/train.py --phase 3 --data_dir <dir> --tokenizer round3train/tokenizer.json --resume <ckpt.pt>
python round3train/inference.py --seq2seq <ckpt.pt> --tokenizer round3train/tokenizer.json --analyze <dir>
```

## Architecture

### Layer Stack
```
Flutter UI (Dart)
  ↓ dart:ffi — DynamicLibrary.open("libomr_engine.so")
C++ OmrEngine (ml/omr/engine/, self-authored, TFLite-only)
```
No JNI, no MethodChannel, no sibling repo involved. `MainActivity.kt` is a bare `FlutterActivity` — Dart loads the native library directly.

### Key Files
| File | Role |
|------|------|
| `lib/main.dart` | Entire Flutter UI — init, image pick, result display |
| `lib/omr_service.dart` | `dart:ffi` bridge to `libomr_engine` — `init(modelDir)`, `process(bytes) -> List<OmrToken>` |
| `android/app/src/main/kotlin/.../MainActivity.kt` | Bare `FlutterActivity`, no OMR-specific code |
| `android/app/build.gradle.kts` | `externalNativeBuild.cmake.path` → `ml/omr/engine/CMakeLists.txt` (builds `libomr_engine.so` straight into the APK) |
| `ml/omr/engine/` | Self-authored C++ OMR engine (pipeline stages below) |
| `round3train/` | Round 3 (grand staff) PyTorch training pipeline — see "ML Training Pipelines" note below |

### Native Library (Android NDK, C++20) — `ml/omr/engine/`
- **OpenCV** — image decode & preprocessing
- **TensorFlow Lite** — segnet + encoder + decoder; all three stages are unified on TFLite. (No ONNX Runtime — that was only needed by the removed homr-based prototype.)
- No third-party OMR/ML source is vendored or linked in.

### Model Assets
```
<modelDir>/
  segnet.tflite
  encoder.tflite
  decoder.tflite
  tokenizer.json      -- single unified DeepScore vocabulary (round3train/tokenizer.json)
```
Not yet bundled as Flutter assets — model export (`ml/omr/training/export_tflite.py`) hasn't been run against a finished Round 3 checkpoint yet. `OmrService.init(modelDir)` takes a plain filesystem directory path; copying the models out of the Flutter asset bundle onto disk is still open (see Known Gaps).

### OMR Pipeline Stages (C++, `ml/omr/engine/src/`)
1. `preprocessor` → `perspective_corrector` → `noise_filter` — image cleanup
2. `segnet_runner` (TFLite) → `staff_detector` → `staff_canvas` → `page_dewarper` — staff geometry
3. `encoder_runner` (TFLite) → `decoder_runner` (TFLite, autoregressive) — seq2seq inference
4. `token_parser` — token IDs ↔ strings via `tokenizer.json`

## ML Training Pipelines (two parallel implementations — read before touching)

There are **two** separate PyTorch training pipelines in this repo implementing roughly the same architecture:
- `ml/omr/training/` + `ml/omr/data_gen/round{1,2,3}/` — what `ml/scripts/train_round.py` actually invokes for every round, defaulting to the shared `ml/data/tokenizer.json`.
- `round3train/` — a separately-built Round 3 (grand staff) fork used for the actual recent Round 3 experiments; has its own `round3train/tokenizer.json`.

These have drifted: `round3train/tokenizer.json` was recently changed to fix low note-pitch recognition accuracy (`note-{pitch}-{dur}` split into `note-{pitch}` + `dur-{dur}`, vocab 1013 → 258 — see `round3train/relabel_notes.py` for migrating existing labels without re-rendering images). `ml/data/tokenizer.json` and `ml/omr/training/` were **not** updated to match, so `train_round.py --round 3` currently still uses the old, un-split vocab. Decide whether to consolidate onto one pipeline before running more Round 3 training.

## iOS Status

iOS has only a stub `AppDelegate.swift`. No native OMR bridge yet — planned as Dart FFI + a statically-linked `libomr_engine.a` (no ObjC/Swift bridge code needed, since `dart:ffi` works the same way on iOS). See `docs/flutter-integration-architect.md`.

## Known Gaps / Follow-ups

- `android/app/build.gradle.kts` now points `externalNativeBuild` at `ml/omr/engine/CMakeLists.txt`, but that CMake file's Android branch expects `-DOPENCV_ANDROID_SDK=` / `-DTFLITE_ROOT=` cache variables (classic `find_package`/`find_library`), while Gradle currently supplies OpenCV/TensorFlow Lite via **prefab** AARs (`buildFeatures.prefab = true`). This mismatch has not been reconciled or build-tested — no Android SDK/NDK toolchain was available to verify a real `flutter build apk` after this change; resolve and test on a machine with the Android toolchain before relying on it.
- `OmrService.init(modelDir)` expects real filesystem paths; nothing yet copies the model files out of the Flutter asset bundle onto disk (native code can't read directly into the asset bundle on Android/iOS). Likely needs `path_provider` + a one-time copy step once models are actually exported.
- Two training pipelines and their tokenizers have diverged — see "ML Training Pipelines" above.

## Platform Notes

- **Min Android SDK:** 24, NDK ABIs: `arm64-v8a`, `x86_64`
- **C++ Standard:** C++20
- **Java/Kotlin:** Java 17
- **Gradle JVM heap:** 8 GB (`gradle.properties`)
- `ml/omr/engine/` builds standalone on Linux/WSL/Windows too (`omr_test`/`omr_eval` desktop binaries) — no Android-only code paths in the C++ itself.
