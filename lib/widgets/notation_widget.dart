import 'package:flutter/material.dart';
import '../data/samples.dart';

const _unitW   = 80.0;
const _cellH   = 46.0;
const _zoneH   = 56.0;
const _marginL = 88.0;  // H/L 음자리표 + 박자표 공간 확보 (72 → 88)
const _marginY = 10.0;
const _annotH  = 24.0;  // 상단 어노테이션 행 (셈여림·헤어핀·도돌이표)

const _zoneBg      = [Color(0xFF0f0f22), Color(0xFF121230), Color(0xFF0f0f22)];
const _dividerColor = Color(0xFF252550);
const _bgCell      = Color(0xFF16162e);

class NotationWidget extends StatefulWidget {
  final List<ScoreNote> notes;
  final int highlightIdx;
  final int expectedIdx;
  final void Function(int idx, ScoreNote note)? onNoteTap;
  final String clef;
  final List<int>? timeSignature;
  final ScrollController? scrollController;

  const NotationWidget({
    super.key,
    required this.notes,
    this.highlightIdx  = -1,
    this.expectedIdx   = -1,
    this.onNoteTap,
    this.clef          = 'treble',
    this.timeSignature,
    this.scrollController,
  });

  @override
  State<NotationWidget> createState() => _NotationWidgetState();
}

class _NotationWidgetState extends State<NotationWidget>
    with SingleTickerProviderStateMixin {
  late final ScrollController _ownScrollCtrl;
  ScrollController get _scrollCtrl =>
      widget.scrollController ?? _ownScrollCtrl;
  late final AnimationController _pulseCtrl;
  late final Animation<double> _pulse;

  @override
  void initState() {
    super.initState();
    _ownScrollCtrl = ScrollController();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 850),
    )..repeat(reverse: true);
    _pulse = CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut);
  }

  @override
  void didUpdateWidget(NotationWidget old) {
    super.didUpdateWidget(old);
    if (widget.expectedIdx != old.expectedIdx && widget.expectedIdx >= 0) {
      WidgetsBinding.instance
          .addPostFrameCallback((_) => _scrollToIdx(widget.expectedIdx));
    }
    if (widget.highlightIdx != old.highlightIdx && widget.highlightIdx >= 0) {
      WidgetsBinding.instance
          .addPostFrameCallback((_) => _scrollToIdx(widget.highlightIdx));
    }
  }

  void _scrollToIdx(int idx) {
    if (idx < 0 || idx >= widget.notes.length) return;
    if (!_scrollCtrl.hasClients) return;
    double x = _marginL + 4;
    for (int i = 0; i < idx; i++) {
      x += widget.notes[i].duration * _unitW;
    }
    final cx = x + widget.notes[idx].duration * _unitW / 2;
    final vp = _scrollCtrl.position.viewportDimension;
    final target =
        (cx - vp / 2).clamp(0.0, _scrollCtrl.position.maxScrollExtent);
    _scrollCtrl.animateTo(target,
        duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
  }

  double get _totalW {
    double w = _marginL + 4;
    for (final n in widget.notes) w += n.duration * _unitW;
    return w + 40;
  }

  double get _totalH => _annotH + _zoneH * 3 + _marginY * 2;

  int? _hitTest(Offset local) {
    double x = _marginL + 4;
    for (int i = 0; i < widget.notes.length; i++) {
      final note = widget.notes[i];
      if (note.isRest) {
        x += note.duration * _unitW;
        continue;
      }
      final w    = note.duration * _unitW - 4;
      final zone = pitchToZone(note.pitch, clef: widget.clef);
      final y    = _annotH + _marginY + zone * _zoneH + 5;
      if (local.dx >= x && local.dx <= x + w &&
          local.dy >= y && local.dy <= y + _cellH) {
        return i;
      }
      x += note.duration * _unitW;
    }
    return null;
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _ownScrollCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: _totalH,
      decoration: BoxDecoration(
        color: const Color(0xFF0a0a1a),
        borderRadius: BorderRadius.circular(8),
      ),
      child: SingleChildScrollView(
        controller: _scrollCtrl,
        scrollDirection: Axis.horizontal,
        child: GestureDetector(
          onTapUp: (d) {
            final idx = _hitTest(d.localPosition);
            if (idx != null) widget.onNoteTap?.call(idx, widget.notes[idx]);
          },
          child: AnimatedBuilder(
            animation: _pulse,
            builder: (_, __) => CustomPaint(
              size: Size(_totalW, _totalH),
              painter: _NotationPainter(
                notes:         widget.notes,
                highlightIdx:  widget.highlightIdx,
                expectedIdx:   widget.expectedIdx,
                pulseValue:    _pulse.value,
                clef:          widget.clef,
                timeSignature: widget.timeSignature,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────

class _NotationPainter extends CustomPainter {
  final List<ScoreNote> notes;
  final int    highlightIdx;
  final int    expectedIdx;
  final double pulseValue;
  final String clef;
  final List<int>? timeSignature;

  const _NotationPainter({
    required this.notes,
    required this.highlightIdx,
    required this.expectedIdx,
    required this.pulseValue,
    required this.clef,
    this.timeSignature,
  });

  // 존 상단 y (어노테이션 행 + 상단 여백 포함)
  static double _staffY(int zone) => _annotH + _marginY + zone * _zoneH;

  // ── 셈여림 텍스트·크기 ─────────────────────────────────────────────────────
  static String _dynText(String mark) {
    switch (mark) {
      case 'ppp':
      case 'pp':
      case 'p':
      case 'mp':  return '약';
      case 'fp':  return '강↓';
      case 'sf':
      case 'sfz': return '강!';
      default:    return '강';
    }
  }

  static double _dynSize(String mark) {
    switch (mark) {
      case 'ppp': return 8.0;
      case 'pp':  return 10.0;
      case 'p':   return 12.0;
      case 'mp':  return 14.0;
      case 'mf':  return 14.0;
      case 'f':   return 16.0;
      case 'ff':  return 18.0;
      case 'fff': return 20.0;
      default:    return 13.0;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────

  @override
  void paint(Canvas canvas, Size size) {
    _drawZones(canvas, size);
    _drawNotes(canvas);
    _drawMeasureDots(canvas);
  }

  void _drawZones(Canvas canvas, Size size) {
    final right  = size.width - 8;
    final labels = zoneLabels(clef);
    for (int z = 0; z < 3; z++) {
      final zy = _staffY(z);
      canvas.drawRect(
        Rect.fromLTWH(_marginL, zy, right - _marginL, _zoneH),
        Paint()..color = _zoneBg[z],
      );
      if (z > 0) _drawDashed(canvas, _marginL, right, zy);
      _drawZoneLabel(canvas, labels[z], zy);
    }
    _drawClefLabel(canvas);
    _drawTimeSig(canvas);
  }

  // H / L 음자리표 레이블
  void _drawClefLabel(Canvas canvas) {
    final label = clef == 'treble' ? 'H' : 'L';
    final tp = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: Color(0xFFCCCCCC),
          fontSize: 20,
          fontWeight: FontWeight.bold,
          fontFamily: 'sans-serif',
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final midY = _annotH + _marginY + _zoneH * 1.5;
    tp.paint(canvas, Offset(4, midY - tp.height / 2));
  }

  // 박자표: 위/아래 숫자 스택
  void _drawTimeSig(Canvas canvas) {
    if (timeSignature == null || timeSignature!.length < 2) return;
    const style = TextStyle(
      color: Color(0xFFCCCCCC),
      fontSize: 16,
      fontWeight: FontWeight.bold,
      fontFamily: 'sans-serif',
    );
    final numTp = TextPainter(
      text: TextSpan(text: timeSignature![0].toString(), style: style),
      textDirection: TextDirection.ltr,
    )..layout();
    final denTp = TextPainter(
      text: TextSpan(text: timeSignature![1].toString(), style: style),
      textDirection: TextDirection.ltr,
    )..layout();
    final midY = _annotH + _marginY + _zoneH * 1.5;
    const x = 30.0;
    numTp.paint(canvas, Offset(x, midY - numTp.height - 1));
    denTp.paint(canvas, Offset(x, midY + 1));
  }

  void _drawDashed(Canvas canvas, double x1, double x2, double y) {
    final p = Paint()..color = _dividerColor..strokeWidth = 1.5;
    double x = x1;
    while (x < x2) {
      canvas.drawLine(Offset(x, y), Offset((x + 6).clamp(x, x2), y), p);
      x += 10;
    }
  }

  void _drawZoneLabel(Canvas canvas, String text, double zy) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: Color(0xFF3c3c60),
          fontSize: 10,
          fontFamily: 'sans-serif',
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas,
        Offset(_marginL - tp.width - 6, zy + _zoneH / 2 - tp.height / 2));
  }

  // ─────────────────────────────────────────────────────────────────────────

  void _drawNotes(Canvas canvas) {
    double x = _marginL + 4;
    for (int i = 0; i < notes.length; i++) {
      final note      = notes[i];
      final w         = note.duration * _unitW - 4;
      final beatColor = Color(beatColorValues[note.beat] ?? 0xFF888888);
      final isHL      = i == highlightIdx;
      final isExp     = i == expectedIdx;
      final zone      = note.isRest ? 1 : pitchToZone(note.pitch, clef: clef);
      final y         = _staffY(zone) + 5;

      if (note.isRest) {
        _drawRestCell(canvas, x, y, w, beatColor);
      } else if (note.chordNotes.isNotEmpty) {
        _drawChordCell(canvas, note, x, y, w, beatColor, isHL, isExp);
      } else {
        _drawSingleNote(canvas, note, x, y, w, beatColor, isHL, isExp);
      }

      // expected 금색 화살표
      if (isExp && !note.isRest) {
        final arrowAlpha = (180 + pulseValue * 75).round();
        final cx = x + w / 2;
        final arrowY = y - 8;
        final path = Path()
          ..moveTo(cx, arrowY + 7)
          ..lineTo(cx - 6, arrowY)
          ..lineTo(cx + 6, arrowY)
          ..close();
        canvas.drawPath(path,
            Paint()..color = Color.fromARGB(arrowAlpha, 0xFF, 0xD7, 0x00));
      }

      _drawAnnotations(canvas, note, x, w);
      x += note.duration * _unitW;
    }
  }

  // 쉼표: 흰색 반투명 테두리만 있는 빈 박스
  void _drawRestCell(Canvas canvas, double x, double y, double w, Color beatColor) {
    final rRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(x + 2, y + 2, w - 4, _cellH - 4),
      const Radius.circular(4),
    );
    canvas.drawRRect(rRect, Paint()..color = const Color(0x18FFFFFF));
    canvas.drawRRect(rRect,
        Paint()
          ..color = beatColor.withAlpha(70)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5);
  }

  void _drawSingleNote(Canvas canvas, ScoreNote note,
      double x, double y, double w,
      Color beatColor, bool isHL, bool isExp) {
    final textColor = isHL
        ? Colors.white
        : isExp
            ? const Color(0xFFFFD700)
            : beatColor;
    final fs = w < 26 ? 10.0 : w < 42 ? 12.0 : 14.0;
    final tp = TextPainter(
      text: TextSpan(
        text: formatNoteName(note.pitch),
        style: TextStyle(
          color: textColor,
          fontSize: fs,
          fontWeight: FontWeight.bold,
          fontFamily: 'sans-serif',
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas,
        Offset(x + w / 2 - tp.width / 2, y + _cellH / 2 - tp.height / 2));
  }

  void _drawChordCell(Canvas canvas, ScoreNote note,
      double x, double y, double w,
      Color beatColor, bool isHL, bool isExp) {
    Color fillColor;
    Color strokeColor;
    double strokeW;

    if (isHL) {
      fillColor   = beatColor.withAlpha(80);
      strokeColor = Colors.white.withAlpha(220);
      strokeW     = 3.0;
    } else if (isExp) {
      final alpha = (25 + pulseValue * 55).round();
      fillColor   = const Color(0xFFFFD700).withAlpha(alpha);
      final strokeAlpha = (160 + pulseValue * 95).round();
      strokeColor = Color.fromARGB(strokeAlpha, 0xFF, 0xD7, 0x00);
      strokeW     = 2.5;
    } else {
      fillColor   = _bgCell;
      strokeColor = beatColor;
      strokeW     = 2.0;
    }

    final rRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(x, y, w, _cellH),
      const Radius.circular(5),
    );
    canvas.drawRRect(rRect, Paint()..color = fillColor);
    canvas.drawRRect(rRect,
        Paint()
          ..color = strokeColor
          ..style = PaintingStyle.stroke
          ..strokeWidth = strokeW);

    if (isHL) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(
            Rect.fromLTWH(x - 3, y - 3, w + 6, _cellH + 6),
            const Radius.circular(8)),
        Paint()
          ..color = beatColor.withAlpha(60)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5,
      );
    }

    final allNotes  = [note.pitch, ...note.chordNotes];
    final textColor = isHL
        ? Colors.white
        : isExp
            ? const Color(0xFFFFD700)
            : beatColor;
    final fs    = allNotes.length <= 2 ? 10.0 : 8.5;
    final label = allNotes.map(formatNoteName).join('\n');
    final tp = TextPainter(
      text: TextSpan(
        text: label,
        style: TextStyle(
          color: textColor,
          fontSize: fs,
          fontWeight: FontWeight.bold,
          fontFamily: 'sans-serif',
          height: 1.15,
        ),
      ),
      textAlign: TextAlign.center,
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: w);
    tp.paint(canvas,
        Offset(x + w / 2 - tp.width / 2, y + _cellH / 2 - tp.height / 2));
  }

  // 상단 어노테이션 행 (_annotH 공간에 그림)
  void _drawAnnotations(Canvas canvas, ScoreNote note, double x, double w) {
    if (note.dynamicMark != null) {
      _drawAnnotText(
        canvas, _dynText(note.dynamicMark!), x, w,
        _dynSize(note.dynamicMark!), const Color(0xFFFF9966),
      );
    } else if (note.hairpin != null) {
      final text = note.hairpin == 'cresc' ? '점점 세게' : '점점 약하게';
      _drawAnnotText(canvas, text, x, w, 9.0, const Color(0xFF99CCFF));
    } else if (note.repeatMark != null) {
      final text = note.repeatMark == 'end-repeat' ? '반복' : '여기부터';
      _drawAnnotText(canvas, text, x, w, 10.0, const Color(0xFFFFCC44));
    }
  }

  void _drawAnnotText(Canvas canvas, String text,
      double x, double w, double fontSize, Color color) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: fontSize,
          fontWeight: FontWeight.bold,
          fontFamily: 'sans-serif',
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final ty = ((_annotH - tp.height) / 2).clamp(0.0, _annotH);
    tp.paint(canvas, Offset(x + w / 2 - tp.width / 2, ty));
  }

  void _drawMeasureDots(Canvas canvas) {
    final refCY = _annotH + _marginY + _zoneH + _zoneH / 2;
    double x = _marginL + 4;
    for (final note in notes) {
      if (note.beat == 1) {
        canvas.drawCircle(
            Offset(x, refCY), 5, Paint()..color = const Color(0xFFFF4444));
      }
      x += note.duration * _unitW;
    }
  }

  @override
  bool shouldRepaint(_NotationPainter old) =>
      old.notes != notes ||
      old.highlightIdx != highlightIdx ||
      old.expectedIdx != expectedIdx ||
      old.clef != clef ||
      old.timeSignature != timeSignature ||
      (old.pulseValue - pulseValue).abs() > 0.01;
}
