import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';

const _wkW = 36.0;
const _wkH = 148.0;
const _bkW = 22.0;
const _bkH = 92.0;

const _whiteNames = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
const _blackDefs = [
  ('C#', 0.64), ('D#', 1.64), ('F#', 3.65), ('G#', 4.64), ('A#', 5.64),
];

class PianoWidget extends StatefulWidget {
  final String? highlightNote; // 음표 탭으로 표시할 음 (파란 하이라이트)
  final String? wrongNote;     // 틀리게 누른 음 (빨간 지속 깜빡임)
  final void Function(String note)? onKeyTap;
  final void Function(String note)? onKeyDown;
  final void Function(String note)? onKeyUp;

  const PianoWidget({
    super.key,
    this.highlightNote,
    this.wrongNote,
    this.onKeyTap,
    this.onKeyDown,
    this.onKeyUp,
  });

  @override
  State<PianoWidget> createState() => _PianoWidgetState();
}

class _PianoWidgetState extends State<PianoWidget>
    with SingleTickerProviderStateMixin {
  final _scrollCtrl = ScrollController();
  String? _pressed; // 현재 물리적으로 눌린 건반

  late final AnimationController _blinkCtrl;
  late final Animation<double> _blink;

  @override
  void initState() {
    super.initState();
    _blinkCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 350),
    );
    _blink = CurvedAnimation(parent: _blinkCtrl, curve: Curves.easeInOut);
  }

  @override
  void didUpdateWidget(PianoWidget old) {
    super.didUpdateWidget(old);
    if (widget.wrongNote != null && old.wrongNote == null) {
      _blinkCtrl.repeat(reverse: true);
    } else if (widget.wrongNote == null && old.wrongNote != null) {
      _blinkCtrl.stop();
      _blinkCtrl.value = 0;
    }
    if (widget.highlightNote != old.highlightNote) {
      _scrollToNote(widget.highlightNote);
    }
    if (widget.wrongNote != old.wrongNote && widget.wrongNote != null) {
      _scrollToNote(widget.wrongNote);
    }
  }

  void _scrollToNote(String? note) {
    if (note == null) return;
    final name = note.substring(0, note.length - 1);
    final oct = int.tryParse(note[note.length - 1]);
    if (oct == null || !_scrollCtrl.hasClients) return;

    double x;
    if (oct == 0) {
      x = _whiteNames.contains(name) ? _whiteNames.indexOf(name) * _wkW : 0;
    } else {
      final base = (2 + (oct - 1) * 7) * _wkW;
      final baseName = name.replaceAll('#', '');
      final wi = _whiteNames.indexOf(baseName).clamp(0, 6);
      x = base + wi * _wkW;
    }

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollCtrl.hasClients) return;
      final vp = _scrollCtrl.position.viewportDimension;
      final target =
          (x - vp / 2 + _wkW * 2).clamp(0.0, _scrollCtrl.position.maxScrollExtent);
      _scrollCtrl.animateTo(target,
          duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
    });
  }

  @override
  void dispose() {
    _blinkCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  double get _totalW => (2 + 7 * 7 + 1) * _wkW;

  List<_WhiteKey> get _whites {
    final keys = <_WhiteKey>[];
    keys.add(_WhiteKey('A0', 0 * _wkW));
    keys.add(_WhiteKey('B0', 1 * _wkW));
    for (int o = 1; o <= 7; o++) {
      final base = (2 + (o - 1) * 7) * _wkW;
      for (int wi = 0; wi < _whiteNames.length; wi++) {
        keys.add(_WhiteKey('${_whiteNames[wi]}$o', base + wi * _wkW));
      }
    }
    keys.add(_WhiteKey('C8', (2 + 7 * 7) * _wkW));
    return keys;
  }

  List<_BlackKey> get _blacks {
    final keys = <_BlackKey>[];
    keys.add(_BlackKey('A#0', 0.64 * _wkW));
    for (int o = 1; o <= 7; o++) {
      final base = (2 + (o - 1) * 7) * _wkW;
      for (final (name, pos) in _blackDefs) {
        keys.add(_BlackKey('$name$o', base + pos * _wkW - _bkW / 2));
      }
    }
    return keys;
  }

  bool _isBlack(String note) =>
      note.length > 1 && note[1] == '#';

  // ── 건반 색상 결정 ──────────────────────────────────────────────────────────

  Color _whiteColor(String note, double blinkVal) {
    if (note == _pressed) return const Color(0xFFFFB0B0);
    if (note == widget.wrongNote) {
      // 지속 빨간 깜빡임: 1.0→빨간, 0.0→연한
      final r = Color.lerp(const Color(0xFFFF8888), const Color(0xFFFF2222), blinkVal)!;
      return r;
    }
    if (note == widget.highlightNote) return const Color(0xFFE0B98C);
    return const Color(0xFFF4EFE6);
  }

  Color _blackColor(String note, double blinkVal) {
    if (note == _pressed) return const Color(0xFF8B0000);
    if (note == widget.wrongNote) {
      return Color.lerp(const Color(0xFF660000), const Color(0xFFCC0000), blinkVal)!;
    }
    if (note == widget.highlightNote) return const Color(0xFF7A4A20);
    return const Color(0xFF1C1C1C);
  }

  List<BoxShadow>? _whiteShadow(String note) {
    if (note == widget.wrongNote) {
      return [BoxShadow(color: Colors.red.withAlpha(180), blurRadius: 14, spreadRadius: 3)];
    }
    if (note == widget.highlightNote) {
      return [BoxShadow(color: gloryAccent.withValues(alpha: .55), blurRadius: 10, spreadRadius: 2)];
    }
    return null;
  }

  List<BoxShadow>? _blackShadow(String note) {
    if (note == widget.wrongNote) {
      return [BoxShadow(color: Colors.red.withAlpha(200), blurRadius: 12, spreadRadius: 2)];
    }
    return [const BoxShadow(color: Colors.black54, blurRadius: 4, offset: Offset(1, 2))];
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
    final whites = _whites;
    final blacks = _blacks;

    return Container(
      height: _wkH + 8,
      color: glorySurface,
      child: SingleChildScrollView(
        controller: _scrollCtrl,
        scrollDirection: Axis.horizontal,
        child: AnimatedBuilder(
          animation: _blink,
          builder: (_, __) {
            final blinkVal = _blink.value;
            return SizedBox(
              width: _totalW,
              height: _wkH,
              child: Stack(
                children: [
                  // White keys
                  ...whites.map((k) => Positioned(
                    left: k.x,
                    top: 0,
                    child: GestureDetector(
                      onTapDown: (_) => _onDown(k.note),
                      onTapUp: (_) {
                        _onUp(k.note);
                        widget.onKeyTap?.call(k.note);
                      },
                      onTapCancel: () => _onUp(k.note),
                      child: Container(
                        width: _wkW - 1,
                        height: _wkH,
                        decoration: BoxDecoration(
                          color: _whiteColor(k.note, blinkVal),
                          border: Border.all(color: const Color(0xFFC8C0B0), width: 0.5),
                          borderRadius: const BorderRadius.vertical(bottom: Radius.circular(4)),
                          boxShadow: _whiteShadow(k.note),
                        ),
                        child: Align(
                          alignment: Alignment.bottomCenter,
                          child: Padding(
                            padding: const EdgeInsets.only(bottom: 4),
                            child: Text(
                              k.note.startsWith('C') ? k.note : k.note.substring(0, k.note.length - 1),
                              style: TextStyle(
                                fontSize: 8,
                                color: k.note == widget.wrongNote
                                    ? Colors.white
                                    : k.note == widget.highlightNote
                                        ? gloryInk
                                        : const Color(0xFFB0A090),
                                fontWeight: (k.note == widget.wrongNote || k.note == widget.highlightNote)
                                    ? FontWeight.bold
                                    : FontWeight.normal,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  )),
                  // Black keys
                  ...blacks.map((k) => Positioned(
                    left: k.x,
                    top: 0,
                    child: GestureDetector(
                      onTapDown: (_) => _onDown(k.note),
                      onTapUp: (_) {
                        _onUp(k.note);
                        widget.onKeyTap?.call(k.note);
                      },
                      onTapCancel: () => _onUp(k.note),
                      child: Container(
                        width: _bkW,
                        height: _bkH,
                        decoration: BoxDecoration(
                          color: _blackColor(k.note, blinkVal),
                          borderRadius: const BorderRadius.vertical(bottom: Radius.circular(4)),
                          boxShadow: _blackShadow(k.note),
                        ),
                      ),
                    ),
                  )),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

class _WhiteKey {
  final String note;
  final double x;
  const _WhiteKey(this.note, this.x);
}

class _BlackKey {
  final String note;
  final double x;
  const _BlackKey(this.note, this.x);
}
