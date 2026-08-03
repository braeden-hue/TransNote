import 'dart:math' as math;
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';

import '../data/samples.dart';

/// 모바일/데스크톱(Android/iOS/Windows/macOS/Linux)용 오디오 -- audio_web.dart(Web Audio
/// API 오실레이터)와 동일한 소리(삼각파 + 지수 감쇠 0.5초)를 순수 Dart로 WAV PCM 합성해
/// audioplayers로 재생한다. 이전엔 이 플랫폼들이 전부 audio_stub.dart(아무 소리도 안 남)를
/// 썼던 게 "기기에서 소리가 안 들린다"는 문제의 근본 원인이었음.
///
/// 양손(대보표) 동시 재생 시 두 음이 겹쳐 울려야 하므로, AudioPlayer 하나를 재사용하지
/// 않고 여러 개를 풀로 돌려쓴다(재사용 시 이전 소리가 끊기고 새 소리로 대체돼버림 --
/// 한쪽 손 음역대가 안 들리던 문제의 원인).
class AudioService {
  static final AudioService instance = AudioService._();
  AudioService._();

  static const _poolSize = 8;
  final List<AudioPlayer> _pool = List.generate(_poolSize, (_) => AudioPlayer()..setReleaseMode(ReleaseMode.stop));
  int _nextPlayer = 0;

  final Map<String, Uint8List> _wavCache = {};

  void unlock() {}

  void playNote(String pitch) {
    final freq = noteToFrequency(pitch);
    final wav = _wavCache.putIfAbsent(pitch, () => _synthesizeWav(freq));
    final player = _pool[_nextPlayer];
    _nextPlayer = (_nextPlayer + 1) % _poolSize;
    player.play(BytesSource(wav));
  }

  // audio_web.dart의 오실레이터 설정과 동일하게 맞춤: type='triangle',
  // gain 0.35*0.4 -> 0.001로 0.5초에 걸쳐 지수 감쇠.
  static Uint8List _synthesizeWav(
    double freq, {
    int sampleRate = 44100,
    double durationSec = 0.5,
    double volume = 0.35,
  }) {
    final numSamples = (sampleRate * durationSec).round();
    final startGain = volume * 0.4;
    const endGain = 0.001;
    final pcm = Int16List(numSamples);
    for (int i = 0; i < numSamples; i++) {
      final t = i / sampleRate;
      // 삼각파: 2/pi * asin(sin(2*pi*f*t)) -> [-1, 1]
      final triangle = 2 / math.pi * math.asin(math.sin(2 * math.pi * freq * t));
      final envelope = startGain * math.pow(endGain / startGain, t / durationSec);
      final sample = (triangle * envelope * 32767).round();
      pcm[i] = sample.clamp(-32768, 32767);
    }
    return _pcmToWav(pcm, sampleRate);
  }

  static Uint8List _pcmToWav(Int16List pcm, int sampleRate) {
    const channels = 1;
    const bitsPerSample = 16;
    final byteRate = sampleRate * channels * bitsPerSample ~/ 8;
    final blockAlign = channels * bitsPerSample ~/ 8;
    final dataSize = pcm.lengthInBytes;
    final buffer = ByteData(44 + dataSize);

    void writeString(int offset, String s) {
      for (int i = 0; i < s.length; i++) {
        buffer.setUint8(offset + i, s.codeUnitAt(i));
      }
    }

    writeString(0, 'RIFF');
    buffer.setUint32(4, 36 + dataSize, Endian.little);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    buffer.setUint32(16, 16, Endian.little); // fmt chunk size
    buffer.setUint16(20, 1, Endian.little); // PCM
    buffer.setUint16(22, channels, Endian.little);
    buffer.setUint32(24, sampleRate, Endian.little);
    buffer.setUint32(28, byteRate, Endian.little);
    buffer.setUint16(32, blockAlign, Endian.little);
    buffer.setUint16(34, bitsPerSample, Endian.little);
    writeString(36, 'data');
    buffer.setUint32(40, dataSize, Endian.little);

    for (int i = 0; i < pcm.length; i++) {
      buffer.setInt16(44 + i * 2, pcm[i], Endian.little);
    }

    return buffer.buffer.asUint8List();
  }
}
