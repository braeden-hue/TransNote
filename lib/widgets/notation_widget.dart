import 'package:flutter/material.dart';
import '../data/samples.dart';

const _unitW = 80.0;
const _cellH = 46.0;
const _zoneH = 56.0;
const _marginL = 72.0;
const _marginY = 10.0;

const _zoneLabels = ['높음 (5옥+)', '중간 (4옥)', '낮음 (3옥↓)'];
const _zoneBg = [Color(0xFF0f0f22), Color(0xFF121230), Color(0xFF0f0f22)];
const _dividerColor = Color(0xFF252550);
const _bgCell = Color(0xFF16162e);

class NotationWidget extends StatefulWidget {
  final List<ScoreNote> notes;
  final int highlightIdx;  // 방금 맞힌 음표 (밝게 강조)
  final int expectedIdx;   // 다음에 눌러야 할 음표 (금색 펄스)
  final void Function(int idx, ScoreNote note)? onNoteTap;

  const NotationWidget({
    super.key,
    required this.notes,
    this.highlightIdx = -1,
    this.expectedIdx = -1,
    this.onNoteTap,
  });

  @override
  State<NotationWidget> createState() => _NotationWidgetState();
}

class _NotationWidgetState extends State<NotationWidget>
    with SingleTickerProviderStateMixin {
  final _scrollCtrl = ScrollController();
  late final AnimationController _pulseCtrl;
  late final Animation<double> _pulse;

  @override
  void initState() {
    super.initState();
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

  double get _totalH => _zoneH * 3 + _marginY * 2;

  int? _hitTest(Offset local) {
    double x = _marginL + 4;
    for (int i = 0; i < widget.notes.length; i++) {
      final note = widget.notes[i];
      final w = note.duration * _unitW - 4;
      final zone = pitchToZone(note.pitch);
      final y = _marginY + zone * _zoneH + 5;
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
    _scrollCtrl.dispose();
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
                notes: widget.notes,
                highlightIdx: widget.highlightIdx,
                expectedIdx: widget.expectedIdx,
                pulseValue: _pulse.value,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _NotationPainter extends CustomPainter {
  final List<ScoreNote> notes;
  final int highlightIdx;
  final int expectedIdx;
  final double pulseValue; // 0.0 ~ 1.0

  const _NotationPainter({
    required this.notes,
    required this.highlightIdx,
    required this.expectedIdx,
    required this.pulseValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    _drawZones(canvas, size);
    _drawNotes(canvas);
    _drawMeasureDots(canvas);
  }

  void _drawZones(Canvas canvas, Size size) {
    final right = size.width - 8;
    for (int z = 0; z < 3; z++) {
      final zy = _marginY + z * _zoneH;
      canvas.drawRect(
        Rect.fromLTWH(_marginL, zy, right - _marginL, _zoneH),
        Paint()..color = _zoneBg[z],
      );
      if (z > 0) _drawDashed(canvas, _marginL, right, zy);
      _drawZoneLabel(canvas, _zoneLabels[z], zy);
    }
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
            color: Color(0xFF3c3c60), fontSize: 10, fontFamily: 'sans-serif'),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas,
        Offset(_marginL - tp.width - 6, zy + _zoneH / 2 - tp.height / 2));
  }

  void _drawNotes(Canvas canvas) {
    double x = _marginL + 4;
    for (int i = 0; i < notes.length; i++) {
      final note = notes[i];
      final w = note.duration * _unitW - 4;
      final zone = pitchToZone(note.pitch);
      final y = _marginY + zone * _zoneH + 5;
      final beatColor = Color(beatColorValues[note.beat] ?? 0xFF888888);
      final isHL = i == highlightIdx;
      final isExp = i == expectedIdx;

      Color fillColor;
      Color strokeColor;
      double strokeW;

      if (isHL) {
        // 방금 맞힌 음 — 박자 색으로 밝게
        fillColor = beatColor.withAlpha(80);
        strokeColor = Colors.white.withAlpha(220);
        strokeW = 3.0;
      } else if (isExp) {
        // 다음에 눌러야 할 음 — 금색 펄스
        final alpha = (25 + pulseValue * 55).round();
        fillColor = const Color(0xFFFFD700).withAlpha(alpha);
        final strokeAlpha = (160 + pulseValue * 95).round();
        strokeColor = Color.fromARGB(strokeAlpha, 0xFF, 0xD7, 0x00);
        strokeW = 2.5;
      } else {
        fillColor = _bgCell;
        strokeColor = beatColor;
        strokeW = 2.0;
      }

      final rRect = RRect.fromRectAndRadius(
        Rect.fromLTWH(x, y, w, _cellH),
        const Radius.circular(5),
      );
      canvas.drawRRect(rRect, Paint()..color = fillColor);
      canvas.drawRRect(
          rRect,
          Paint()
            ..color = strokeColor
            ..style = PaintingStyle.stroke
            ..strokeWidth = strokeW);

      // 하이라이트 링 (방금 맞힌 음)
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

      // expected 화살표 (위쪽 삼각형)
      if (isExp) {
        final arrowAlpha = (180 + pulseValue * 75).round();
        final arrowPaint = Paint()
          ..color = Color.fromARGB(arrowAlpha, 0xFF, 0xD7, 0x00);
        final cx = x + w / 2;
        final arrowY = y - 8;
        final path = Path()
          ..moveTo(cx, arrowY + 7)
          ..lineTo(cx - 6, arrowY)
          ..lineTo(cx + 6, arrowY)
          ..close();
        canvas.drawPath(path, arrowPaint);
      }

      final fs = w < 26 ? 9.0 : w < 42 ? 11.0 : 13.0;
      final label = formatNoteName(note.pitch);
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: TextStyle(
            color: isHL
                ? Colors.white
                : isExp
                    ? const Color(0xFFFFD700)
                    : beatColor,
            fontSize: fs,
            fontWeight: FontWeight.bold,
            fontFamily: 'sans-serif',
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas,
          Offset(x + w / 2 - tp.width / 2, y + _cellH / 2 - tp.height / 2));

      x += note.duration * _unitW;
    }
  }

  void _drawMeasureDots(Canvas canvas) {
    const refCY = _marginY + _zoneH + _zoneH / 2;
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
      (old.pulseValue - pulseValue).abs() > 0.01;
}
