import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';

/// 한 옥타브(C~C) 고정, 화면 너비에 맞춰 스크롤 없이 표시되는 컴팩트 피아노.
/// 스크롤 가능한 전체 88건반 [PianoWidget]과 달리 튜토리얼의 "한 음씩 맞추기"용.
const _whiteNames = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C'];
const _blackDefs = [
  ('C#', 0.64), ('D#', 1.64), ('F#', 3.65), ('G#', 4.64), ('A#', 5.64),
];

class MiniPianoWidget extends StatefulWidget {
  final String octave;
  final String? highlightNote;
  final String? wrongNote;
  final void Function(String note)? onKeyDown;
  final void Function(String note)? onKeyUp;

  const MiniPianoWidget({
    super.key,
    this.octave = '4',
    this.highlightNote,
    this.wrongNote,
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
    if (note == widget.highlightNote) return const Color(0xFFE0B98C);
    return const Color(0xFFF4EFE6);
  }

  Color _blackColor(String note) {
    if (note == _pressed) return const Color(0xFF8B0000);
    if (note == widget.wrongNote) return const Color(0xFFAA2222);
    if (note == widget.highlightNote) return const Color(0xFF7A4A20);
    return const Color(0xFF1C1C1C);
  }

  @override
  Widget build(BuildContext context) {
    final octNext = (int.parse(widget.octave) + 1).toString();
    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final wkW = w / _whiteNames.length;
        const wkH = 116.0;
        final bkW = wkW * 0.6;
        final bkH = wkH * 0.62;

        String noteAt(int i) => i == _whiteNames.length - 1
            ? '${_whiteNames[i]}$octNext'
            : '${_whiteNames[i]}${widget.octave}';

        return SizedBox(
          width: w,
          height: wkH,
          child: Stack(
            children: [
              for (int i = 0; i < _whiteNames.length; i++)
                Positioned(
                  left: i * wkW,
                  top: 0,
                  child: GestureDetector(
                    onTapDown: (_) => _onDown(noteAt(i)),
                    onTapUp: (_) => _onUp(noteAt(i)),
                    onTapCancel: () => _onUp(noteAt(i)),
                    child: Container(
                      width: wkW - 1,
                      height: wkH,
                      decoration: BoxDecoration(
                        color: _whiteColor(noteAt(i)),
                        border: Border.all(color: const Color(0xFFC8C0B0), width: 0.5),
                        borderRadius: const BorderRadius.vertical(bottom: Radius.circular(6)),
                      ),
                      child: Align(
                        alignment: Alignment.bottomCenter,
                        child: Padding(
                          padding: const EdgeInsets.only(bottom: 6),
                          child: Text(_whiteNames[i],
                              style: TextStyle(fontSize: 10, color: gloryInk.withValues(alpha: .35))),
                        ),
                      ),
                    ),
                  ),
                ),
              for (final (name, pos) in _blackDefs)
                Positioned(
                  left: pos * wkW - bkW / 2,
                  top: 0,
                  child: GestureDetector(
                    onTapDown: (_) => _onDown('$name${widget.octave}'),
                    onTapUp: (_) => _onUp('$name${widget.octave}'),
                    onTapCancel: () => _onUp('$name${widget.octave}'),
                    child: Container(
                      width: bkW,
                      height: bkH,
                      decoration: BoxDecoration(
                        color: _blackColor('$name${widget.octave}'),
                        borderRadius: const BorderRadius.vertical(bottom: Radius.circular(4)),
                        boxShadow: const [BoxShadow(color: Colors.black45, blurRadius: 3, offset: Offset(1, 1))],
                      ),
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
