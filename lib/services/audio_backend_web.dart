import 'dart:convert';
import 'dart:typed_data';
import 'dart:html' as html;
import 'audio_backend.dart';

/// 웹용 — audioplayers의 web 플러그인 채널 등록이 `flutter run -d web-server`
/// 조합에서 깨지는 경우가 있어(MissingPluginException: .../audioplayers.global/events)
/// 플러그인 없이 dart:html의 <audio> 엘리먼트로 직접 재생한다. data: URI라 별도
/// 파일 서빙/CORS 이슈도 없다.
class WebAudioBackend implements AudioBackend {
  static const _poolSize = 8;
  final List<html.AudioElement> _pool = List.generate(_poolSize, (_) => html.AudioElement());
  int _next = 0;

  @override
  Future<void> playWav(Uint8List bytes) async {
    final element = _pool[_next];
    _next = (_next + 1) % _pool.length;
    element.pause();
    element.src = 'data:audio/wav;base64,${base64Encode(bytes)}';
    try {
      await element.play();
    } catch (_) {
      // 브라우저가 사용자 제스처 없는 재생을 막은 경우 등 — 무음으로 넘어간다.
    }
  }

  @override
  Future<void> stop() async {
    for (final el in _pool) {
      el.pause();
    }
  }
}

AudioBackend createAudioBackend() => WebAudioBackend();
