import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';

/// 디코딩된 OMR 토큰 하나 (C++ omr::OmrToken과 대응, ml/omr/engine/include/types.hpp).
class OmrToken {
  final int id;
  final String text;
  const OmrToken(this.id, this.text);

  @override
  String toString() => 'OmrToken(id: $id, text: $text)';
}

final class _OmrTokenC extends Struct {
  @Int32()
  external int id;

  @Array(64)
  external Array<Uint8> text;
}

final class _OmrResultC extends Struct {
  @Int32()
  external int success;

  @Array(256)
  external Array<Uint8> error;

  external Pointer<_OmrTokenC> tokens;

  @Int32()
  external int tokenCount;
}

typedef _OmrCreateNative = Pointer<Void> Function(
    Pointer<Utf8> segnetPath,
    Pointer<Utf8> encoderPath,
    Pointer<Utf8> decoderPath,
    Pointer<Utf8> tokenizerPath);
typedef _OmrCreateDart = Pointer<Void> Function(
    Pointer<Utf8> segnetPath,
    Pointer<Utf8> encoderPath,
    Pointer<Utf8> decoderPath,
    Pointer<Utf8> tokenizerPath);

typedef _OmrProcessNative = Pointer<_OmrResultC> Function(
    Pointer<Void> engine, Pointer<Uint8> imageData, Int32 imageLen);
typedef _OmrProcessDart = Pointer<_OmrResultC> Function(
    Pointer<Void> engine, Pointer<Uint8> imageData, int imageLen);

typedef _OmrFreeResultNative = Void Function(Pointer<_OmrResultC> result);
typedef _OmrFreeResultDart = void Function(Pointer<_OmrResultC> result);

typedef _OmrDestroyNative = Void Function(Pointer<Void> engine);
typedef _OmrDestroyDart = void Function(Pointer<Void> engine);

/// ml/omr/engine (libomr_engine)에 dart:ffi로 직접 바인딩하는 서비스.
///
/// 이전 MethodChannel("com.example.musicscore/omr") + Kotlin/JNI(OmrPipeline.kt,
/// flutter_jni.cpp) 브리지를 대체한다 — Android 쪽에는 더 이상 OMR 전용 네이티브
/// 코드가 필요 없고, Dart가 ml/omr/engine이 빌드한 공유 라이브러리를 직접 로드한다.
///
/// 가정: [modelDir] 아래에 다음 4개 파일이 존재한다 (ml/omr/engine/include/types.hpp
/// 의 ModelPaths와 대응). 아직 모델 export(export_tflite.py 등)가 끝나지 않아
/// 실제 파일명 규칙이 확정되지 않았으므로, 확정되면 아래 상수만 바꾸면 된다.
///   segnet.tflite / encoder.tflite / decoder.tflite / tokenizer.json
class OmrService {
  OmrService._();
  static final OmrService instance = OmrService._();

  DynamicLibrary? _lib;
  late final _OmrCreateDart _omrCreate;
  late final _OmrProcessDart _omrProcess;
  late final _OmrFreeResultDart _omrFreeResult;
  late final _OmrDestroyDart _omrDestroy;

  Pointer<Void> _engine = nullptr;

  DynamicLibrary _openLibrary() {
    if (Platform.isAndroid || Platform.isLinux) {
      return DynamicLibrary.open('libomr_engine.so');
    }
    if (Platform.isIOS || Platform.isMacOS) {
      return DynamicLibrary.process(); // 정적 링크 (libomr_engine.a)
    }
    if (Platform.isWindows) {
      return DynamicLibrary.open('omr_engine.dll');
    }
    throw UnsupportedError('OMR engine은 이 플랫폼을 지원하지 않음: ${Platform.operatingSystem}');
  }

  /// [modelDir]에서 4개 모델 파일을 읽어 네이티브 엔진을 초기화한다.
  /// 웹에서는 항상 false (Dart FFI 미지원 — FastAPI 서버 사이드 경로 사용).
  bool init(String modelDir) {
    if (kIsWeb) return false;

    final lib = _lib ??= _openLibrary();
    _omrCreate = lib.lookupFunction<_OmrCreateNative, _OmrCreateDart>('omr_create');
    _omrProcess = lib.lookupFunction<_OmrProcessNative, _OmrProcessDart>('omr_process');
    _omrFreeResult =
        lib.lookupFunction<_OmrFreeResultNative, _OmrFreeResultDart>('omr_free_result');
    _omrDestroy = lib.lookupFunction<_OmrDestroyNative, _OmrDestroyDart>('omr_destroy');

    final segnetPath = '$modelDir/segnet.tflite'.toNativeUtf8();
    final encoderPath = '$modelDir/encoder.tflite'.toNativeUtf8();
    final decoderPath = '$modelDir/decoder.tflite'.toNativeUtf8();
    final tokenizerPath = '$modelDir/tokenizer.json'.toNativeUtf8();
    try {
      _engine = _omrCreate(segnetPath, encoderPath, decoderPath, tokenizerPath);
    } finally {
      malloc.free(segnetPath);
      malloc.free(encoderPath);
      malloc.free(decoderPath);
      malloc.free(tokenizerPath);
    }
    return _engine != nullptr;
  }

  /// 이미지 바이트(JPEG/PNG) → OMR 토큰 리스트. 미초기화·실패 시 null.
  List<OmrToken>? process(Uint8List bytes) {
    if (kIsWeb || _engine == nullptr) return null;

    final buf = malloc<Uint8>(bytes.length);
    buf.asTypedList(bytes.length).setAll(0, bytes);

    Pointer<_OmrResultC> resultPtr = nullptr;
    try {
      resultPtr = _omrProcess(_engine, buf, bytes.length);
      if (resultPtr == nullptr || resultPtr.ref.success == 0) return null;

      final result = resultPtr.ref;
      final tokens = <OmrToken>[
        for (var i = 0; i < result.tokenCount; i++) _readToken(result.tokens, i),
      ];
      return tokens;
    } finally {
      malloc.free(buf);
      if (resultPtr != nullptr) _omrFreeResult(resultPtr);
    }
  }

  static const int _tokenTextLen = 64; // OmrTokenC.text[64] (omr_engine.h)

  OmrToken _readToken(Pointer<_OmrTokenC> tokens, int index) {
    final tok = (tokens + index).ref;
    return OmrToken(tok.id, _readCString(tok.text, _tokenTextLen));
  }

  String _readCString(Array<Uint8> chars, int maxLen) {
    final codes = <int>[];
    for (var i = 0; i < maxLen; i++) {
      final c = chars[i];
      if (c == 0) break;
      codes.add(c);
    }
    return String.fromCharCodes(codes);
  }

  void dispose() {
    if (_engine != nullptr) {
      _omrDestroy(_engine);
      _engine = nullptr;
    }
  }
}
