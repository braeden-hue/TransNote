import 'dart:typed_data';
import 'package:audioplayers/audioplayers.dart';
import 'audio_backend.dart';

/// Android/iOS/데스크톱용 — audioplayers 플러그인 채널 사용.
class IoAudioBackend implements AudioBackend {
  static const _poolSize = 8;
  final List<AudioPlayer> _pool =
      List.generate(_poolSize, (_) => AudioPlayer()..setReleaseMode(ReleaseMode.stop));
  int _next = 0;

  @override
  Future<void> playWav(Uint8List bytes) async {
    final player = _pool[_next];
    _next = (_next + 1) % _pool.length;
    try {
      await player.stop();
      await player.play(BytesSource(bytes));
    } catch (_) {
      // 재생 실패는 무음으로 넘어간다 — 연습 흐름을 막지 않는다.
    }
  }

  @override
  Future<void> stop() async {
    for (final p in _pool) {
      await p.stop();
    }
  }
}

AudioBackend createAudioBackend() => IoAudioBackend();
