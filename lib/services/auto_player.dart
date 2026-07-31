import 'dart:async';

import 'package:flutter/foundation.dart';

import '../data/samples.dart';

/// 대보표(treble+bass)/단일 오선 공용 자동 재생 엔진. 인덱스를 한 스텝씩 전진시키는 대신
/// 경과 시간(박자 단위)을 주기적으로 갱신해 현재 위치를 이진 탐색으로 찾는다 -- treble/bass가
/// 서로 다른 리듬이어도 같은 타임라인을 공유하므로 항상 같이 맞아 들어가고(실제 피아노처럼
/// 양손이 동시에 울림), Timer 체이닝 방식보다 누적 오차(drift)도 없다.
class AutoPlayer {
  final List<ScoreNote> treble;
  final List<ScoreNote>? bass;
  final int tempo;
  final void Function(int trebleIdx, int bassIdx) onTick;
  final VoidCallback onDone;

  Timer? _timer;
  double _elapsedBeats = 0;
  late final List<double> _trebleStarts;
  late final List<double>? _bassStarts;
  late final double _totalBeats;

  AutoPlayer({
    required this.treble,
    this.bass,
    required this.tempo,
    required this.onTick,
    required this.onDone,
  }) {
    _trebleStarts = _cumulativeStarts(treble);
    _bassStarts = bass != null ? _cumulativeStarts(bass!) : null;
    final trebleEnd = treble.isEmpty ? 0.0 : _trebleStarts.last + treble.last.duration;
    final bassEnd = (bass == null || bass!.isEmpty) ? 0.0 : _bassStarts!.last + bass!.last.duration;
    _totalBeats = trebleEnd > bassEnd ? trebleEnd : bassEnd;
  }

  static List<double> _cumulativeStarts(List<ScoreNote> notes) {
    final starts = <double>[];
    double t = 0;
    for (final n in notes) {
      starts.add(t);
      t += n.duration;
    }
    return starts;
  }

  bool get isPlaying => _timer != null;

  void start() {
    _elapsedBeats = 0;
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(milliseconds: 60), _tick);
  }

  void stop() {
    _timer?.cancel();
    _timer = null;
  }

  void _tick(Timer t) {
    if (_totalBeats <= 0) {
      stop();
      onDone();
      return;
    }
    final beatsPerMs = tempo / 60000.0;
    _elapsedBeats += 60 * beatsPerMs;
    if (_elapsedBeats >= _totalBeats) {
      stop();
      onDone();
      return;
    }
    final ti = _indexAt(_trebleStarts, _elapsedBeats);
    final bi = _bassStarts != null ? _indexAt(_bassStarts, _elapsedBeats) : -1;
    onTick(ti, bi);
  }

  static int _indexAt(List<double> starts, double t) {
    if (starts.isEmpty) return -1;
    int lo = 0, hi = starts.length - 1, ans = 0;
    while (lo <= hi) {
      final mid = (lo + hi) ~/ 2;
      if (starts[mid] <= t) {
        ans = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return ans;
  }

  void dispose() => stop();
}
