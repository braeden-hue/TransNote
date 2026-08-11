import 'dart:typed_data';

/// 합성된 WAV 바이트를 실제로 재생하는 최소 인터페이스. 웹/네이티브 구현이 서로
/// 다른 이유는 audio_backend_selector.dart 참고.
abstract class AudioBackend {
  Future<void> playWav(Uint8List bytes);
  Future<void> stop();
}
