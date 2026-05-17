import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class OmrService {
  static const _channel = MethodChannel('com.example.musicscore/omr');
  static bool _initialized = false;

  static Future<bool> initialize() async {
    if (kIsWeb) return true; // OMR pipeline not available on web
    if (_initialized) return true;
    try {
      _initialized = await _channel.invokeMethod<bool>('initialize') ?? false;
    } catch (_) {
      _initialized = false;
    }
    return _initialized;
  }

  static Future<String?> processImageBytes(Uint8List bytes) async {
    if (kIsWeb) return null;
    if (!_initialized) await initialize();
    try {
      return await _channel.invokeMethod<String>('processImageBytes', bytes);
    } catch (_) {
      return null;
    }
  }
}
