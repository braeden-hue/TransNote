import 'dart:typed_data';
import 'package:flutter/services.dart';

class OmrService {
  static const _channel = MethodChannel('com.example.musicscore/omr');
  static bool _initialized = false;

  static Future<bool> initialize() async {
    if (_initialized) return true;
    _initialized = await _channel.invokeMethod<bool>('initialize') ?? false;
    return _initialized;
  }

  static Future<String?> processImageBytes(Uint8List bytes) async {
    if (!_initialized) await initialize();
    return _channel.invokeMethod<String>('processImageBytes', bytes);
  }
}
