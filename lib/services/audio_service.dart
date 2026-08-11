import 'dart:math';
import 'dart:typed_data';
import 'audio_backend.dart';
import 'audio_backend_selector.dart' show createAudioBackend;

/// 사운드폰트/오디오 파일 없이, 음높이마다 짧은 피아노풍 톤을 그 자리에서 합성해
/// WAV 바이트로 만들어 재생한다. 실제 재생은 플랫폼별 AudioBackend에 위임 —
/// audioplayers의 web 플러그인 채널이 `flutter run -d web-server`에서 깨지는
/// 경우가 있어(MissingPluginException) 웹은 dart:html 오디오로 우회한다
/// (audio_backend_web.dart 참고).
class AudioService {
  static const _sampleRate = 44100;
  static const _toneSeconds = 0.9;

  final AudioBackend _backend = createAudioBackend();
  final Map<String, Uint8List> _cache = {};

  Future<void> playNote(String pitch, {double durationSeconds = 0.4}) async {
    if (pitch.isEmpty) return;
    final bytes = _cache.putIfAbsent(pitch, () => _synthesize(_frequencyOf(pitch)));
    await _backend.playWav(bytes);
  }

  Future<void> stop() => _backend.stop();

  double _frequencyOf(String pitch) {
    const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    final name = pitch.substring(0, pitch.length - 1);
    final octave = int.parse(pitch.substring(pitch.length - 1));
    final semitone = names.indexOf(name);
    if (semitone < 0) return 440;
    return 440 * pow(2, (semitone + (octave - 4) * 12 - 9) / 12) as double;
  }

  /// 기본음 + 배음 2개를 섞고 지수 감쇠 엔벨로프를 걸어 건반을 튕기는 듯한 소리를 낸다.
  Uint8List _synthesize(double freq) {
    final n = (_sampleRate * _toneSeconds).round();
    final pcm = Int16List(n);
    for (var i = 0; i < n; i++) {
      final t = i / _sampleRate;
      final envelope = exp(-t * 4.5);
      final wave = sin(2 * pi * freq * t) * 0.55 +
          sin(2 * pi * freq * 2 * t) * 0.25 +
          sin(2 * pi * freq * 3 * t) * 0.12;
      pcm[i] = (wave * envelope).clamp(-1.0, 1.0) * 32767 ~/ 1;
    }
    return _wavBytesOf(pcm);
  }

  Uint8List _wavBytesOf(Int16List pcm) {
    const blockAlign = 2; // mono, 16-bit
    const byteRate = _sampleRate * blockAlign;
    final dataSize = pcm.length * 2;
    final bytes = ByteData(44 + dataSize);

    void writeString(int offset, String s) {
      for (var i = 0; i < s.length; i++) {
        bytes.setUint8(offset + i, s.codeUnitAt(i));
      }
    }

    writeString(0, 'RIFF');
    bytes.setUint32(4, 36 + dataSize, Endian.little);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    bytes.setUint32(16, 16, Endian.little);
    bytes.setUint16(20, 1, Endian.little); // PCM
    bytes.setUint16(22, 1, Endian.little); // mono
    bytes.setUint32(24, _sampleRate, Endian.little);
    bytes.setUint32(28, byteRate, Endian.little);
    bytes.setUint16(32, blockAlign, Endian.little);
    bytes.setUint16(34, 16, Endian.little); // bits per sample
    writeString(36, 'data');
    bytes.setUint32(40, dataSize, Endian.little);
    for (var i = 0; i < pcm.length; i++) {
      bytes.setInt16(44 + i * 2, pcm[i], Endian.little);
    }
    return bytes.buffer.asUint8List();
  }
}

final audioService = AudioService();
