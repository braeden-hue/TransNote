// webpage/js/notation.js 포팅 (renderNotation/renderGrandStaff) — CustomPainter 버전.
import 'package:flutter/material.dart';
import '../data/samples.dart';

const Color _expectedColor = Color(0xFF0076CE);
const Color _zoneDivider = Color(0xFFC5D8EC);

class NotationWidget extends StatelessWidget {
  final List<NoteEvent> notes;
  final String clef;
  final int highlightIdx;
  final int expectedIdx;
  final void Function(int index, NoteEvent note)? onNoteTap;
  final List<Color>? zoneColors;
  final bool hideNoteNames;
  final bool hideNotes;

  const NotationWidget({
    super.key,
    required this.notes,
    this.clef = 'treble',
    this.highlightIdx = -1,
    this.expectedIdx = -1,
    this.onNoteTap,
    this.zoneColors,
    this.hideNoteNames = false,
    this.hideNotes = false,
  });

  static const double unitW = 80;
  static const double cellH = 46;
  static const double zoneH = cellH + 10;
  static const double marginL = 68;
  static const double marginY = 8;
  static const double indicatorH = 18;

  @override
  Widget build(BuildContext context) {
    if (notes.isEmpty) {
      return const Padding(
        padding: EdgeInsets.all(20),
        child: Text('음표 데이터가 없습니다', style: TextStyle(color: Color(0xFF555555))),
      );
    }
    final totalDur = notes.fold<double>(0, (s, n) => s + n.duration);
    final width = totalDur * unitW + marginL + 40;
    const height = indicatorH + zoneH * 3 + marginY * 2;

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: GestureDetector(
        onTapUp: onNoteTap == null ? null : (details) => _handleTap(details.localPosition),
        child: RepaintBoundary(
          child: CustomPaint(
            size: Size(width, height),
            painter: _NotationPainter(
              notes: notes,
              clef: clef,
              highlightIdx: highlightIdx,
              expectedIdx: expectedIdx,
              zoneColors: zoneColors,
              hideNoteNames: hideNoteNames,
              hideNotes: hideNotes,
            ),
          ),
        ),
      ),
    );
  }

  void _handleTap(Offset pos) {
    double x = marginL + 4;
    for (var i = 0; i < notes.length; i++) {
      final w = notes[i].duration * unitW;
      if (pos.dx >= x && pos.dx < x + w) {
        onNoteTap!(i, notes[i]);
        return;
      }
      x += w;
    }
  }
}

class _NotationPainter extends CustomPainter {
  final List<NoteEvent> notes;
  final String clef;
  final int highlightIdx;
  final int expectedIdx;
  final List<Color>? zoneColors;
  final bool hideNoteNames;
  final bool hideNotes;

  _NotationPainter({
    required this.notes,
    required this.clef,
    required this.highlightIdx,
    required this.expectedIdx,
    required this.zoneColors,
    required this.hideNoteNames,
    required this.hideNotes,
  });

  static const double unitW = NotationWidget.unitW;
  static const double cellH = NotationWidget.cellH;
  static const double zoneH = NotationWidget.zoneH;
  static const double marginL = NotationWidget.marginL;
  static const double marginY = NotationWidget.marginY;
  static const double indicatorH = NotationWidget.indicatorH;

  @override
  void paint(Canvas canvas, Size size) {
    final contentY = indicatorH + marginY;

    // ── zone 배경 ──
    for (var z = 0; z < 3; z++) {
      final zy = contentY + z * zoneH;
      final rect = Rect.fromLTWH(marginL, zy, size.width - marginL - 8, zoneH);
      final fill = zoneColors != null
          ? zoneColors![z].withValues(alpha: 0.55)
          : (z % 2 == 0 ? const Color(0xFFF0F6FC) : const Color(0xFFF8FBFF));
      canvas.drawRect(rect, Paint()..color = fill);
      if (z > 0) {
        _drawDashedLine(canvas, Offset(marginL, zy), Offset(size.width - 8, zy),
            Paint()
              ..color = _zoneDivider
              ..strokeWidth = 1.5,
            6, 4);
      }
    }

    // ── x 위치 계산 ──
    final noteX = <double>[];
    final measureXs = <double>[];
    double x = marginL + 4;
    for (final note in notes) {
      if (note.beat == 1) measureXs.add(x);
      noteX.add(x);
      x += note.duration * unitW;
    }

    final mixedClef = hasMixedClef(notes, clef);

    for (var i = 0; i < notes.length; i++) {
      if (hideNotes) continue;
      final note = notes[i];
      final w = note.duration * unitW - 4;
      final effClef = effectiveClef(note, clef);
      final zone = note.isRest ? 1 : pitchToZone(note.pitch, effClef);
      final y = contentY + zone * zoneH + 5;
      const h = cellH;
      final color = Color(beatColor(note.beat));
      final isHL = i == highlightIdx;
      final isExp = i == expectedIdx;
      final nx = noteX[i];

      if (mixedClef) {
        final tint = effClef == 'bass'
            ? const Color(0x1FFFB347)
            : const Color(0x1F6C63FF);
        canvas.drawRRect(
          RRect.fromRectAndRadius(Rect.fromLTWH(nx, y, w, h), const Radius.circular(4)),
          Paint()..color = tint,
        );
      }

      if (note.isRest) {
        final lineColor = isHL ? Colors.white : isExp ? _expectedColor : color;
        _drawDashedLine(canvas, Offset(nx, y + h), Offset(nx + w, y + h),
            Paint()
              ..color = lineColor
              ..strokeWidth = isHL || isExp ? 3 : 2.5
              ..strokeCap = StrokeCap.round,
            5, 4);
        _drawText(canvas, '쉼표', Offset(nx + w / 2, y + h / 2), lineColor,
            fontSize: w < 30 ? 9 : 11);
      } else if (note.isChord) {
        final fill = isHL
            ? color.withValues(alpha: 0.27)
            : isExp
                ? _expectedColor.withValues(alpha: 0.09)
                : (mixedClef ? Colors.white.withValues(alpha: 0.55) : Colors.white);
        final strokeColor = isExp ? _expectedColor : color;
        final strokeW = isHL ? 3.0 : isExp ? 2.5 : 2.0;
        final rrect = RRect.fromRectAndRadius(Rect.fromLTWH(nx, y, w, h), const Radius.circular(5));
        canvas.drawRRect(rrect, Paint()..color = fill);
        canvas.drawRRect(
            rrect,
            Paint()
              ..color = strokeColor
              ..style = PaintingStyle.stroke
              ..strokeWidth = strokeW);

        if (!hideNoteNames) {
          final allNotes = [note.pitch, ...note.chordNotes];
          final lineH = allNotes.length <= 2 ? 14.0 : 12.0;
          final fs = allNotes.length <= 2 ? 11.0 : 9.0;
          final totalTxtH = allNotes.length * lineH;
          final textColor = isHL ? Colors.white : isExp ? _expectedColor : color;
          for (var pi = 0; pi < allNotes.length; pi++) {
            _drawText(
              canvas,
              formatNoteName(allNotes[pi]),
              Offset(nx + w / 2, y + h / 2 - totalTxtH / 2 + lineH * (pi + 0.5)),
              textColor,
              fontSize: fs,
            );
          }
        }
      } else {
        final lineColor = isHL ? Colors.white : isExp ? _expectedColor : color;
        canvas.drawLine(
          Offset(nx, y + h),
          Offset(nx + w, y + h),
          Paint()
            ..color = lineColor
            ..strokeWidth = isHL || isExp ? 3 : 2.5
            ..strokeCap = StrokeCap.round,
        );
        if (!hideNoteNames) {
          final fs = w < 26 ? 10.0 : w < 42 ? 12.0 : 14.0;
          _drawText(canvas, formatNoteName(note.pitch), Offset(nx + w / 2, y + h / 2), lineColor,
              fontSize: fs);
        }
      }

      if (isExp) {
        final arrowCx = nx + w / 2;
        final arrowTip = y - 2;
        final arrowTop = (arrowTip - 13).clamp(2.0, arrowTip);
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromLTWH(arrowCx - 6, arrowTop - 1, 12, arrowTip - arrowTop + 2),
            const Radius.circular(2),
          ),
          Paint()..color = Colors.white.withValues(alpha: 0.85),
        );
        final path = Path()
          ..moveTo(arrowCx - 5, arrowTop)
          ..lineTo(arrowCx + 5, arrowTop)
          ..lineTo(arrowCx, arrowTip)
          ..close();
        canvas.drawPath(path, Paint()..color = _expectedColor);
      }
    }

    // ── 마디 시작 기준점 ──
    final refZone = clef == 'bass' ? 1 : 2;
    final refCY = contentY + refZone * zoneH + zoneH / 2;
    for (final mx in measureXs) {
      canvas.drawCircle(Offset(mx, refCY), 5, Paint()..color = _expectedColor);
    }
  }

  void _drawText(Canvas canvas, String text, Offset center, Color color, {required double fontSize}) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(color: color, fontSize: fontSize, fontWeight: FontWeight.w700),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, Offset(center.dx - tp.width / 2, center.dy - tp.height / 2));
  }

  void _drawDashedLine(Canvas canvas, Offset p1, Offset p2, Paint paint, double dashW, double gapW) {
    final total = (p2 - p1).distance;
    if (total == 0) return;
    final dir = (p2 - p1) / total;
    var covered = 0.0;
    while (covered < total) {
      final segEnd = (covered + dashW).clamp(0.0, total);
      canvas.drawLine(p1 + dir * covered, p1 + dir * segEnd, paint);
      covered += dashW + gapW;
    }
  }

  @override
  bool shouldRepaint(covariant _NotationPainter old) {
    return old.notes != notes ||
        old.highlightIdx != highlightIdx ||
        old.expectedIdx != expectedIdx ||
        old.hideNotes != hideNotes ||
        old.hideNoteNames != hideNoteNames ||
        old.zoneColors != zoneColors ||
        old.clef != clef;
  }
}

class _StaffMeta {
  final String label;
  final Color color;
  const _StaffMeta(this.label, this.color);
}

_StaffMeta _staffLabel(String clef) {
  if (clef == 'treble') return const _StaffMeta('🎵 높은음자리 (Treble)', Color(0xFF0076CE));
  if (clef == 'bass') return const _StaffMeta('🎻 낮은음자리 (Bass)', Color(0xFF5BB8F5));
  return const _StaffMeta('Staff', Color(0xFF7BB8A0));
}

/// webpage/js/notation.js renderGrandStaff() 포팅 — 두 보표(치블/베이스)를 세로로 쌓는다.
class GrandStaffWidget extends StatelessWidget {
  final List<Stave> staves;
  final int highlightIdx;
  final Map<String, int>? expectedIdxByClef;
  final void Function(String clef, int index, NoteEvent note)? onNoteTap;

  const GrandStaffWidget({
    super.key,
    required this.staves,
    this.highlightIdx = -1,
    this.expectedIdxByClef,
    this.onNoteTap,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (final stave in staves) ...[
          Padding(
            padding: const EdgeInsets.only(left: NotationWidget.marginL, bottom: 3, top: 6),
            child: Text(
              _staffLabel(stave.clef).label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                color: _staffLabel(stave.clef).color,
                letterSpacing: 0.4,
              ),
            ),
          ),
          NotationWidget(
            notes: stave.notes,
            clef: stave.clef,
            highlightIdx: highlightIdx,
            expectedIdx: expectedIdxByClef?[stave.clef] ?? -1,
            onNoteTap: onNoteTap == null ? null : (i, n) => onNoteTap!(stave.clef, i, n),
          ),
        ],
      ],
    );
  }
}
