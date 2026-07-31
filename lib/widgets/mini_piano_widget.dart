import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';

/// 화면 너비에 맞춰 스크롤 없이 표시되는 컴팩트 피아노. [lowNote]~[highNote] 범위를
/// 자동으로 계산해 흰/검은 건반을 배치한다. 전체 88건반 [PianoWidget]과 달리
/// 튜토리얼의 "예시 악보를 그대로 눌러보기"용 — 항상 그 페이지의 실제 예시 음 범위와 일치해야 함.
const _chromatic = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const _whiteSet = {'C', 'D', 'E', 'F', 'G', 'A', 'B'};
const _blackOffsetInOctave = {'C#': 0.64, 'D#': 1.64, 'F#': 3.65, 'G#': 4.64, 'A#': 5.64};
// 커스텀 악보 표기 규칙과 동일: 검은 건반은 음이름(#)이 아닌 1~5 숫자로 표시
// (online_webpage/js/samples.js BLACK_LABEL, lib/data/samples.dart _blackLabel와 일치해야 함)
const _blackKeyLabel = {'C#': '1', 'D#': '2', 'F#': '3', 'G#': '4', 'A#': '5'};
const _solfege = {
  'C': '도', 'C#': '도#', 'D': '레', 'D#': '레#', 'E': '미', 'F': '파', 'F#': '파#',
  'G': '솔', 'G#': '솔#', 'A': '라', 'A#': '라#', 'B': '시',
};

/// 두 음의 상대적 높낮이를 비교하기 위한 공개 헬퍼(값 자체엔 의미 없음).
int pitchOrder(String pitch) => _semitoneIndex(pitch);

int _semitoneIndex(String pitch) {
  final oct = int.parse(pitch[pitch.length - 1]);
  final name = pitch.substring(0, pitch.length - 1);
  return oct * 12 + _chromatic.indexOf(name);
}

class _KeySpec {
  final String note;
  final String name;
  final bool isWhite;
  final double slot;
  const _KeySpec(this.note, this.name, this.isWhite, this.slot);
}

List<_KeySpec> _buildRange(String low, String high) {
  final loIdx = _semitoneIndex(low);
  final hiIdx = _semitoneIndex(high);
  final loOct = loIdx ~/ 12;
  final keys = <_KeySpec>[];
  double whiteSlot = 0;
  for (int idx = loIdx; idx <= hiIdx; idx++) {
    final name = _chromatic[idx % 12];
    if (_whiteSet.contains(name)) {
      final oct = idx ~/ 12;
      keys.add(_KeySpec('$name$oct', name, true, whiteSlot));
      whiteSlot += 1;
    }
  }
  for (int idx = loIdx; idx <= hiIdx; idx++) {
    final name = _chromatic[idx % 12];
    if (!_whiteSet.contains(name)) {
      final oct = idx ~/ 12;
      final slot = (oct - loOct) * 7 + _blackOffsetInOctave[name]!;
      keys.add(_KeySpec('$name$oct', name, false, slot));
    }
  }
  return keys;
}

/// 범위를 옥타브(C~C) 경계로 반올림 — 검은 건반 위치 계산이 항상 옥타브 시작(C) 기준이어야 함.
(String, String) snapToOctaves(String low, String high) {
  final loOct = int.parse(low[low.length - 1]);
  var hiOct = int.parse(high[high.length - 1]);
  final hiName = high.substring(0, high.length - 1);
  if (hiName != 'C') hiOct += 1;
  return ('C$loOct', 'C$hiOct');
}

class MiniPianoWidget extends StatefulWidget {
  final String lowNote;
  final String highNote;
  final Set<String> highlight;
  final String? wrongNote;
  final bool detailed;
  final void Function(String note)? onKeyDown;
  final void Function(String note)? onKeyUp;

  const MiniPianoWidget({
    super.key,
    this.lowNote = 'C4',
    this.highNote = 'C5',
    this.highlight = const {},
    this.wrongNote,
    this.detailed = false,
    this.onKeyDown,
    this.onKeyUp,
  });

  @override
  State<MiniPianoWidget> createState() => _MiniPianoWidgetState();
}

class _MiniPianoWidgetState extends State<MiniPianoWidget> {
  String? _pressed;

  void _onDown(String note) {
    setState(() => _pressed = note);
    widget.onKeyDown?.call(note);
  }

  void _onUp(String note) {
    if (_pressed == note) setState(() => _pressed = null);
    widget.onKeyUp?.call(note);
  }

  Color _whiteColor(String note) {
    if (note == _pressed) return const Color(0xFFFFB0B0);
    if (note == widget.wrongNote) return const Color(0xFFEE7777);
    if (widget.highlight.contains(note)) return const Color(0xFFE0B98C);
    return const Color(0xFFF4EFE6);
  }

  Color _blackColor(String note) {
    if (note == _pressed) return const Color(0xFF8B0000);
    if (note == widget.wrongNote) return const Color(0xFFAA2222);
    if (widget.highlight.contains(note)) return const Color(0xFF7A4A20);
    return const Color(0xFF1C1C1C);
  }

  @override
  Widget build(BuildContext context) {
    final range = snapToOctaves(widget.lowNote, widget.highNote);
    final keys = _buildRange(range.$1, range.$2);
    final whiteCount = keys.where((k) => k.isWhite).length;

    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final wkW = w / whiteCount;
        final wkH = widget.detailed ? 140.0 : 116.0;
        final bkW = wkW * 0.62;
        final bkH = wkH * 0.62;

        return SizedBox(
          width: w,
          height: wkH,
          child: Stack(
            children: [
              for (final k in keys.where((k) => k.isWhite))
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
                        color: _whiteColor(k.note),
                        border: Border.all(color: const Color(0xFFC8C0B0), width: 0.5),
                        borderRadius: const BorderRadius.vertical(bottom: Radius.circular(6)),
                      ),
                      child: Align(
                        alignment: Alignment.bottomCenter,
                        child: Padding(
                          padding: const EdgeInsets.only(bottom: 6),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(k.name,
                                  style: TextStyle(
                                      fontSize: widget.detailed ? 13 : 10,
                                      fontWeight: widget.detailed ? FontWeight.bold : FontWeight.normal,
                                      color: gloryInk.withValues(alpha: widget.detailed ? .7 : .35))),
                              if (widget.detailed)
                                Text(_solfege[k.name] ?? '',
                                    style: TextStyle(fontSize: 11, color: gloryInk.withValues(alpha: .45))),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              for (final k in keys.where((k) => !k.isWhite))
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
                        color: _blackColor(k.note),
                        borderRadius: const BorderRadius.vertical(bottom: Radius.circular(4)),
                        boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 3, offset: Offset(1, 1))],
                      ),
                      child: widget.detailed
                          ? Align(
                              alignment: Alignment.bottomCenter,
                              child: Padding(
                                padding: const EdgeInsets.only(bottom: 6),
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Text(_blackKeyLabel[k.name] ?? k.name, style: const TextStyle(fontSize: 10, color: Colors.white70, fontWeight: FontWeight.bold)),
                                    Text(_solfege[k.name] ?? '', style: const TextStyle(fontSize: 9, color: Colors.white54)),
                                  ],
                                ),
                              ),
                            )
                          : null,
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
}
