import 'dart:js_interop';
import '../data/samples.dart';

@JS('_playNoteFreq')
external void _playNoteFreq(double freq, double volume);

@JS('_audioUnlock')
external void _audioUnlock();

class AudioService {
  static final AudioService instance = AudioService._();
  AudioService._();

  void unlock() => _audioUnlock();

  void playNote(String pitch) {
    final freq = noteToFrequency(pitch);
    _playNoteFreq(freq, 0.35);
  }
}
