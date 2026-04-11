# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
