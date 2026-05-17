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

Flutter-based Optical Music Recognition (OMR) app that captures sheet music images and converts them to structured data. The core OMR logic lives in a sibling C++ project (`MusicScore/`) and is exposed via JNI to the Flutter app.

## Build & Run Commands

### Flutter
```bash
flutter pub get
flutter run                        # run on connected device
flutter build apk                  # Android release APK
flutter build ios                  # iOS (requires macOS)
flutter build web                  # Web (not yet configured)
```

### Desktop C++ Test Binary (WSL / Linux)
```bash
cd desktop
bash build_flutter_desktop.sh build
bash build_flutter_desktop.sh run desktop/num2.png output.xml
bash build_flutter_desktop.sh clean
```

### Python Validation
```bash
python desktop/compare_musicxml.py desktop/ground_truth/num2.musicxml desktop/output.xml
python desktop/inspect_tflite.py    # inspect TFLite model tensor shapes
```

## Architecture

### Layer Stack
```
Flutter UI (Dart)
  ↓ MethodChannel "com.example.musicscore/omr"
Kotlin (MainActivity.kt, OmrPipeline.kt)
  ↓ JNI
C++ flutter_jni.cpp  →  homr::OmrPipeline  (from MusicScore/app/src/main/cpp)
```

### Key Files
| File | Role |
|------|------|
| `lib/main.dart` | Entire Flutter UI — init, image pick, result display |
| `lib/omr_service.dart` | MethodChannel bridge (2 methods: `initialize`, `processImageBytes`) |
| `android/app/src/main/kotlin/.../MainActivity.kt` | MethodChannel handler, calls JNI |
| `android/app/src/main/kotlin/.../OmrPipeline.kt` | `external fun` JNI declarations |
| `android/app/src/main/cpp/flutter_jni.cpp` | JNI entrypoint — decodes image bytes, calls pipeline |
| `android/app/src/main/cpp/CMakeLists.txt` | Links to `../../../../../../MusicScore/app/src/main/cpp` |
| `desktop/test_main.cpp` | Standalone C++ test harness (no Flutter required) |
| `desktop/asset_manager_impl.cpp` | Mock Android AssetManager for desktop testing |

### Native Libraries (Android NDK, C++20)
- **OpenCV 4.9.0** — image decode & preprocessing
- **TensorFlow Lite 2.16.1** — segmentation model (`segnet_308_int8.tflite`) + encoder (`encoder_331_int8.tflite`)
- **ONNX Runtime 1.18.0** — transformer decoder (`decoder_331_int8.onnx`)
- **tinyxml2 10.0.0** — MusicXML output generation

### Model Assets (loaded via Android AssetManager)
```
assets/
  segnet_308_int8.tflite
  encoder_331_int8.tflite
  decoder_331_int8.onnx
  tokenizer_{rhythm,pitch,lift,note}.json
```

### OMR Pipeline Stages (C++, lives in MusicScore repo)
1. Preprocessing: autocrop → resize → color_adjust → noise_filtering
2. Detection: staff_detection → staff_dewarping → note_detection → bar_line_detection
3. Inference: TFLite segmentation → TFLite encoder → ONNX decoder
4. Post-processing: tr_omr_parser → rhythm_rules → accidental_rules → music_xml_generator

## iOS Status

iOS has only a stub `AppDelegate.swift`. There is no native OMR bridge implemented for iOS yet.

## Platform Notes

- **Min Android SDK:** 24, NDK ABIs: `arm64-v8a`, `x86_64`
- **C++ Standard:** C++20
- **Java/Kotlin:** Java 17
- **Gradle JVM heap:** 8 GB (`gradle.properties`)
- The C++ source is NOT inside this repo — it is referenced by relative path from `CMakeLists.txt`: `../../../../../../MusicScore/app/src/main/cpp`. Both repos must be siblings on disk.
