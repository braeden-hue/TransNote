import 'package:flutter/material.dart';
import '../data/samples.dart';

// 규칙1(세로 위치=음높이) 범례 색 -- notation_widget.dart의 _zoneBg 톤과 계열을 맞춤.
// 여기서 공개(export)해서 tutorial_screen.dart의 범례(_ZoneLabel)가 같은 값을 import해
// 쓰게 하면, "두 곳 색이 반드시 일치해야 한다"는 동기화 부담이 사라진다.
const zoneHighColor = Color(0xFF7C93C4);
const zoneMidColor = Color(0xFFA79C89);
const zoneLowColor = Color(0xFFD99B5C);
const _zoneColors = [zoneHighColor, zoneMidColor, zoneLowColor];

const _whiteNames = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
const _blackOffsets = {'C#': 0.64, 'D#': 1.64, 'F#': 3.65, 'G#': 4.64, 'A#': 5.64};
const _whiteCount = 52; // A0,B0 + 7옥타브(C1~B7) + C8

class _Key {
  final String note;
  final double slot;
  const _Key(this.note, this.slot);
}

List<_Key> _buildWhites() {
  final keys = <_Key>[const _Key('A0', 0), const _Key('B0', 1)];
  for (int o = 1; o <= 7; o++) {
    final base = 2 + (o - 1) * 7;
    for (int wi = 0; wi < _whiteNames.length; wi++) {
      keys.add(_Key('${_whiteNames[wi]}$o', (base + wi).toDouble()));
    }
  }
  keys.add(const _Key('C8', 51));
  return keys;
}

List<_Key> _buildBlacks() {
  final keys = <_Key>[const _Key('A#0', 0.64)];
  for (int o = 1; o <= 7; o++) {
    final base = 2 + (o - 1) * 7;
    for (final e in _blackOffsets.entries) {
      keys.add(_Key('${e.key}$o', base + e.value));
    }
  }
  return keys;
}

/// 88건반(A0~C8) 전체를 화면 폭에 맞춰(스크롤 없이) 표시하고, 그 위에 옥타브 구역별
/// 반투명 색 밴드를 오버레이해 "어느 구역을 눌러야 하는지"를 한눈에 보여주는 튜토리얼 전용
/// 피아노. 구역 경계는 samples.dart의 pitchToZone(clef: 'treble')과 100% 동일하게 계산한다.
class OctaveZonePianoWidget extends StatefulWidget {
  final void Function(String note)? onKeyDown;
  final void Function(String note)? onKeyUp;
  const OctaveZonePianoWidget({super.key, this.onKeyDown, this.onKeyUp});

  @override
  State<OctaveZonePianoWidget> createState() => _OctaveZonePianoWidgetState();
}

class _OctaveZonePianoWidgetState extends State<OctaveZonePianoWidget> {
  String? _pressed;
  late final List<_Key> _whites = _buildWhites();
  late final List<_Key> _blacks = _buildBlacks();

  // 흰 건반을 옥타브 구역(0=높음/1=중간/2=낮음)별로 연속 구간으로 묶는다 -- pitchToZone이
  // 옥타브에 대해 단조 증가하므로 항상 정확히 3구간(낮음→중간→높음)으로 나뉜다.
  List<(int zone, double start, double end)> get _zoneRanges {
    final ranges = <(int, double, double)>[];
    int? curZone;
    double curStart = 0;
    for (final k in _whites) {
      final zone = pitchToZone(k.note, clef: 'treble');
      if (curZone == null) {
        curZone = zone;
        curStart = k.slot;
      } else if (zone != curZone) {
        ranges.add((curZone, curStart, k.slot));
        curZone = zone;
        curStart = k.slot;
      }
    }
    if (curZone != null) ranges.add((curZone, curStart, _whiteCount.toDouble()));
    return ranges;
  }

  void _onDown(String note) {
    setState(() => _pressed = note);
    widget.onKeyDown?.call(note);
  }

  void _onUp(String note) {
    if (_pressed == note) setState(() => _pressed = null);
    widget.onKeyUp?.call(note);
  }

  @override
  Widget build(BuildContext context) {
    final labels = zoneLabels('treble'); // ['6옥+', '5옥', '4옥']
    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final wkW = w / _whiteCount;
        const wkH = 118.0;
        final bkW = wkW * 0.62;
        const bkH = wkH * 0.62;
        final ranges = _zoneRanges;

        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              height: 16,
              width: w,
              child: Stack(
                children: [
                  for (final r in ranges)
                    Positioned(
                      left: r.$2 * wkW,
                      width: (r.$3 - r.$2) * wkW,
                      top: 0,
                      bottom: 0,
                      child: Center(
                        child: Text(labels[r.$1],
                            style: TextStyle(
                                color: _zoneColors[r.$1],
                                fontSize: 10.5,
                                fontWeight: FontWeight.bold)),
                      ),
                    ),
                ],
              ),
            ),
            SizedBox(
              width: w,
              height: wkH,
              child: Stack(
                children: [
                  for (final k in _whites)
                    Positioned(
                      left: k.slot * wkW,
                      top: 0,
                      child: GestureDetector(
                        onTapDown: (_) => _onDown(k.note),
                        onTapUp: (_) => _onUp(k.note),
                        onTapCancel: () => _onUp(k.note),
                        child: Container(
                          width: wkW - 1,
                          height: wkH,
                          decoration: BoxDecoration(
                            color: k.note == _pressed ? const Color(0xFFFFB0B0) : const Color(0xFFF4EFE6),
                            border: Border.all(color: const Color(0xFFC8C0B0), width: 0.5),
                            borderRadius: const BorderRadius.vertical(bottom: Radius.circular(4)),
                          ),
                        ),
                      ),
                    ),
                  for (final k in _blacks)
                    Positioned(
                      left: k.slot * wkW - bkW / 2,
                      top: 0,
                      child: GestureDetector(
                        onTapDown: (_) => _onDown(k.note),
                        onTapUp: (_) => _onUp(k.note),
                        onTapCancel: () => _onUp(k.note),
                        child: Container(
                          width: bkW,
                          height: bkH,
                          decoration: BoxDecoration(
                            color: k.note == _pressed ? const Color(0xFF8B0000) : const Color(0xFF1C1C1C),
                            borderRadius: const BorderRadius.vertical(bottom: Radius.circular(4)),
                            boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 3, offset: Offset(1, 1))],
                          ),
                        ),
                      ),
                    ),
                  // 구역 반투명 오버레이 -- 탭은 아래 건반으로 그대로 통과시켜야 하므로 IgnorePointer.
                  for (final r in ranges)
                    Positioned(
                      left: r.$2 * wkW,
                      width: (r.$3 - r.$2) * wkW,
                      top: 0,
                      bottom: 0,
                      child: IgnorePointer(
                        child: Container(color: _zoneColors[r.$1].withValues(alpha: .32)),
                      ),
                    ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }
}
