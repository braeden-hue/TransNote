import 'package:flutter/material.dart';
import '../data/samples.dart';

const _unitW = 70.0;
const _marginL = 56.0;
const _marginR = 24.0;
const _staffTop = 46.0;
const _totalH = 168.0;
const _lineColor = Color(0xFF6E6259);
const _noteColor = Color(0xFF222222);
const _highlightColor = Color(0xFFFF4444);
const _expectedColor = Color(0xFFC99400);
const _labelColor = Color(0xFF6E6259);

const _letterOrder = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
int _diatonicIndex(String letter, int oct) => oct * 7 + _letterOrder.indexOf(letter);

// 오선 맨 아래 줄(step 0)의 기준 음 -- 높은음자리표 E4, 낮은음자리표 G2.
int _anchor(String clef) =>
    clef == 'bass' ? _diatonicIndex('G', 2) : _diatonicIndex('E', 4);

int _stepOf(String pitch, String clef) {
  final oct = int.parse(pitch[pitch.length - 1]);
  final namePart = pitch.substring(0, pitch.length - 1);
  final letter = namePart.endsWith('#') ? namePart.substring(0, 1) : namePart;
  return _diatonicIndex(letter, oct) - _anchor(clef);
}

/// 커스텀 악보(색칠 셀 표기) 대신 볼 수 있는, 자체 엔진으로 재렌더링한 오선 악보 뷰.
/// NotationWidget과 동일한 외부 API(notes/clef/timeSignature/highlightIdx/expectedIdx/
/// scrollController/onNoteTap)를 가진 드롭인 대체 위젯 -- 호출부는 두 위젯을 조건부로
/// 바꿔 끼우기만 하면 된다. MuseScore급 정교한 엔그레이빙은 아니고(빔/다성부 없음),
/// 음표머리/기둥/꼬리/덧줄/임시표/쉼표/마디선을 갖춘 "깔끔한 오선 뷰" 수준으로 구현.
class StaffNotationWidget extends StatefulWidget {
  final List<ScoreNote> notes;
  final int highlightIdx;
  final int expectedIdx;
  final void Function(int idx, ScoreNote note)? onNoteTap;
  final String clef;
  final List<int>? timeSignature;
  final ScrollController? scrollController;

  const StaffNotationWidget({
    super.key,
    required this.notes,
    this.highlightIdx = -1,
    this.expectedIdx = -1,
    this.onNoteTap,
    this.clef = 'treble',
    this.timeSignature,
    this.scrollController,
  });

  @override
  State<StaffNotationWidget> createState() => _StaffNotationWidgetState();
}

class _StaffNotationWidgetState extends State<StaffNotationWidget> {
  late final ScrollController _ownScrollCtrl;
  ScrollController get _scrollCtrl => widget.scrollController ?? _ownScrollCtrl;

  @override
  void initState() {
    super.initState();
    _ownScrollCtrl = ScrollController();
  }

  @override
  void didUpdateWidget(StaffNotationWidget old) {
    super.didUpdateWidget(old);
    if (widget.highlightIdx != old.highlightIdx && widget.highlightIdx >= 0) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToIdx(widget.highlightIdx));
    } else if (widget.expectedIdx != old.expectedIdx && widget.expectedIdx >= 0) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToIdx(widget.expectedIdx));
    }
  }

  void _scrollToIdx(int idx) {
    if (idx < 0 || idx >= widget.notes.length || !_scrollCtrl.hasClients) return;
    double x = _marginL;
    for (int i = 0; i < idx; i++) {
      x += widget.notes[i].duration * _unitW;
    }
    final cx = x + widget.notes[idx].duration * _unitW / 2;
    final vp = _scrollCtrl.position.viewportDimension;
    final target = (cx - vp / 2).clamp(0.0, _scrollCtrl.position.maxScrollExtent);
    _scrollCtrl.animateTo(target, duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
  }

  double get _totalW {
    double w = _marginL;
    for (final n in widget.notes) {
      w += n.duration * _unitW;
    }
    return w + _marginR;
  }

  int? _hitTest(double dx) {
    double x = _marginL;
    for (int i = 0; i < widget.notes.length; i++) {
      final w = widget.notes[i].duration * _unitW;
      if (dx >= x && dx <= x + w) return i;
      x += w;
    }
    return null;
  }

  @override
  void dispose() {
    _ownScrollCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: _totalH,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFE4D6C6)),
      ),
      clipBehavior: Clip.antiAlias,
      child: SingleChildScrollView(
        controller: _scrollCtrl,
        scrollDirection: Axis.horizontal,
        child: GestureDetector(
          onTapUp: (d) {
            final idx = _hitTest(d.localPosition.dx);
            if (idx != null) widget.onNoteTap?.call(idx, widget.notes[idx]);
          },
          child: CustomPaint(
            size: Size(_totalW, _totalH),
            painter: _StaffNotationPainter(
              notes: widget.notes,
              clef: widget.clef,
              timeSignature: widget.timeSignature,
              highlightIdx: widget.highlightIdx,
              expectedIdx: widget.expectedIdx,
            ),
          ),
        ),
      ),
    );
  }
}

class _StaffNotationPainter extends CustomPainter {
  final List<ScoreNote> notes;
  final String clef;
  final List<int>? timeSignature;
  final int highlightIdx;
  final int expectedIdx;

  const _StaffNotationPainter({
    required this.notes,
    required this.clef,
    required this.timeSignature,
    required this.highlightIdx,
    required this.expectedIdx,
  });

  static const _gap = 11.0; // 오선 줄 간격
  static double get _bottomLineY => _staffTop + 4 * _gap;
  static double _stepToY(int step) => _bottomLineY - step * (_gap / 2);

  @override
  void paint(Canvas canvas, Size size) {
    _drawStaffLines(canvas, size);
    _drawClef(canvas);
    _drawTimeSig(canvas);
    _drawNotesAndBarlines(canvas);
  }

  void _drawStaffLines(Canvas canvas, Size size) {
    final p = Paint()
      ..color = _lineColor
      ..strokeWidth = 1.1;
    for (int i = 0; i < 5; i++) {
      final y = _staffTop + i * _gap;
      canvas.drawLine(Offset(_marginL - 8, y), Offset(size.width - _marginR + 8, y), p);
    }
  }

  void _drawClef(Canvas canvas) {
    final glyph = clef == 'bass' ? '\u{1D122}' : '\u{1D11E}';
    final tp = TextPainter(
      text: TextSpan(
        text: glyph,
        style: TextStyle(color: _noteColor, fontSize: clef == 'bass' ? 34 : 46),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final midY = _staffTop + _gap * 2;
    tp.paint(canvas, Offset(6, midY - tp.height / 2 + (clef == 'bass' ? -2 : 4)));
  }

  void _drawTimeSig(Canvas canvas) {
    if (timeSignature == null || timeSignature!.length < 2) return;
    const style = TextStyle(color: _labelColor, fontSize: 17, fontWeight: FontWeight.bold);
    final numTp = TextPainter(text: TextSpan(text: '${timeSignature![0]}', style: style), textDirection: TextDirection.ltr)..layout();
    final denTp = TextPainter(text: TextSpan(text: '${timeSignature![1]}', style: style), textDirection: TextDirection.ltr)..layout();
    const x = 34.0;
    numTp.paint(canvas, Offset(x, _staffTop - 1));
    denTp.paint(canvas, Offset(x, _staffTop + _gap * 2 + 1));
  }

  void _drawNotesAndBarlines(Canvas canvas) {
    double x = _marginL;
    int? lastBeat;
    for (int i = 0; i < notes.length; i++) {
      final note = notes[i];
      // 새 마디 시작(beat가 이전 값 이하로 리셋) -- samples.dart approximateNotePositions와 동일한 휴리스틱.
      if (lastBeat != null && note.beat <= lastBeat && i > 0) {
        _drawBarline(canvas, x - 6);
      }
      lastBeat = note.beat;

      final w = note.duration * _unitW;
      final isHL = i == highlightIdx;
      final isExp = i == expectedIdx;
      if (note.isRest) {
        _drawRest(canvas, x + w / 2, note.duration);
      } else {
        _drawNote(canvas, note, x + 8, isHL, isExp);
      }
      x += w;
    }
  }

  void _drawBarline(Canvas canvas, double x) {
    canvas.drawLine(
      Offset(x, _staffTop),
      Offset(x, _staffTop + 4 * _gap),
      Paint()
        ..color = _lineColor
        ..strokeWidth = 1.2,
    );
  }

  void _drawNote(Canvas canvas, ScoreNote note, double cx, bool isHL, bool isExp) {
    final step = _stepOf(note.pitch, clef);
    final cy = _stepToY(step);
    final namePart = note.pitch.substring(0, note.pitch.length - 1);
    final sharp = namePart.endsWith('#');

    if (isHL) {
      canvas.drawCircle(Offset(cx, cy), 9, Paint()..color = _highlightColor.withValues(alpha: .35));
    } else if (isExp) {
      canvas.drawCircle(Offset(cx, cy), 9, Paint()..color = _expectedColor.withValues(alpha: .35));
    }

    // 덧줄(ledger line) -- 오선을 벗어난 음
    if (step < 0) {
      for (int s = -2; s >= step; s -= 2) {
        final ly = _stepToY(s);
        canvas.drawLine(Offset(cx - 9, ly), Offset(cx + 9, ly), Paint()..color = _lineColor..strokeWidth = 1.1);
      }
    } else if (step > 8) {
      for (int s = 10; s <= step; s += 2) {
        final ly = _stepToY(s);
        canvas.drawLine(Offset(cx - 9, ly), Offset(cx + 9, ly), Paint()..color = _lineColor..strokeWidth = 1.1);
      }
    }

    if (sharp) {
      final tp = TextPainter(
        text: const TextSpan(text: '♯', style: TextStyle(color: _noteColor, fontSize: 14, fontWeight: FontWeight.bold)),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(cx - 10 - tp.width, cy - tp.height / 2 + 1));
    }

    final hollow = note.duration >= 2;
    final headPaint = Paint()..color = _noteColor;
    if (hollow) {
      headPaint
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5;
    }
    canvas.save();
    canvas.translate(cx, cy);
    canvas.rotate(-0.2);
    canvas.drawOval(const Rect.fromLTWH(-6, -4.2, 12, 8.4), headPaint);
    canvas.restore();

    if (note.duration < 4) {
      final stemUp = step < 4;
      final stemX = cx + (stemUp ? 6 : -6);
      final stemTopY = cy + (stemUp ? -24 : 24);
      canvas.drawLine(Offset(stemX, cy), Offset(stemX, stemTopY),
          Paint()..color = _noteColor..strokeWidth = 1.3);

      if (note.duration <= 0.5) {
        final flag = Path()
          ..moveTo(stemX, stemTopY)
          ..quadraticBezierTo(
              stemX + (stemUp ? 8 : -8), stemTopY + (stemUp ? 4 : -4), stemX + (stemUp ? 2 : -2), stemTopY + (stemUp ? 13 : -13));
        canvas.drawPath(flag, Paint()..color = _noteColor..style = PaintingStyle.stroke..strokeWidth = 1.6);
      }
    }
  }

  // 정교한 SMuFL 쉼표 글리프 대신 duration별로 단순화된 모양을 그린다.
  void _drawRest(Canvas canvas, double cx, double duration) {
    final midY = _stepToY(4); // 오선 가운데 줄
    final p = Paint()..color = _noteColor;
    if (duration >= 4) {
      canvas.drawRect(Rect.fromLTWH(cx - 6, midY - _gap / 2 - 3, 12, 4), p);
    } else if (duration >= 2) {
      canvas.drawRect(Rect.fromLTWH(cx - 6, midY - 1, 12, 4), p);
    } else if (duration >= 1) {
      final path = Path()
        ..moveTo(cx - 3, midY - 10)
        ..lineTo(cx + 3, midY - 3)
        ..lineTo(cx - 3, midY + 4)
        ..lineTo(cx + 3, midY + 11);
      canvas.drawPath(path, Paint()..color = _noteColor..style = PaintingStyle.stroke..strokeWidth = 1.8);
    } else {
      canvas.drawLine(Offset(cx, midY - 9), Offset(cx - 2, midY + 3), Paint()..color = _noteColor..strokeWidth = 2.2);
      final hook = Path()
        ..moveTo(cx, midY - 9)
        ..quadraticBezierTo(cx + 6, midY - 6, cx + 1, midY);
      canvas.drawPath(hook, Paint()..color = _noteColor..style = PaintingStyle.stroke..strokeWidth = 1.6);
    }
  }

  @override
  bool shouldRepaint(covariant _StaffNotationPainter old) =>
      old.notes != notes || old.highlightIdx != highlightIdx || old.expectedIdx != expectedIdx;
}
