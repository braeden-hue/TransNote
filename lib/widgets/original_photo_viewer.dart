import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';

final _audio = AudioService.instance;

/// 커스텀 악보로 변환하지 않고, 촬영한 오선 원본 이미지 위에서 음표를 탭하면 말풍선으로
/// 음이름(도/레#...)을 보여주는 화면. 모델이 음표별 실제 픽셀 좌표를 출력하지 않으므로
/// (토큰 시퀀스만 출력), approximateNotePositions()의 마디/음표 개수 기반 균등분배
/// 근사치로 탭 위치를 매칭한다(정확한 음표 위치가 아니라 데모용 근사치).
class OriginalPhotoScreen extends StatefulWidget {
  final Uint8List photo;
  final List<ScoreNote> notes;
  final String title;
  const OriginalPhotoScreen({super.key, required this.photo, required this.notes, required this.title});

  @override
  State<OriginalPhotoScreen> createState() => _OriginalPhotoScreenState();
}

class _OriginalPhotoScreenState extends State<OriginalPhotoScreen> {
  late final List<double> _positions = approximateNotePositions(widget.notes);
  int? _tappedIdx;
  Offset? _bubblePos;

  void _onTapUp(TapUpDetails d, BoxConstraints box) {
    if (_positions.isEmpty) return;
    final xFrac = (d.localPosition.dx / box.maxWidth).clamp(0.0, 1.0);
    int best = 0;
    double bestDist = (xFrac - _positions[0]).abs();
    for (var i = 1; i < _positions.length; i++) {
      final dist = (xFrac - _positions[i]).abs();
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    final note = widget.notes[best];
    if (note.isRest) return;
    _audio.unlock();
    _audio.playNote(note.pitch);
    setState(() {
      _tappedIdx = best;
      _bubblePos = d.localPosition;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Text(widget.title, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
      ),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Text('악보 위 음표를 눌러보세요 (근사 위치, 확대/이동 가능)',
                  style: TextStyle(color: Colors.white.withValues(alpha: .55), fontSize: 12)),
            ),
            Expanded(
              child: LayoutBuilder(
                builder: (context, box) {
                  return InteractiveViewer(
                    minScale: 1.0,
                    maxScale: 4.0,
                    child: GestureDetector(
                      onTapUp: (d) => _onTapUp(d, box),
                      child: Stack(
                        children: [
                          Positioned.fill(child: Image.memory(widget.photo, fit: BoxFit.contain)),
                          if (_tappedIdx != null && _bubblePos != null)
                            _NoteBubble(
                              pitch: widget.notes[_tappedIdx!].pitch,
                              position: _bubblePos!,
                            ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NoteBubble extends StatelessWidget {
  final String pitch;
  final Offset position;
  const _NoteBubble({required this.pitch, required this.position});

  @override
  Widget build(BuildContext context) {
    final label = solfegeLabel(pitch);
    return Positioned(
      left: (position.dx - 28).clamp(0.0, double.infinity),
      top: (position.dy - 56).clamp(0.0, double.infinity),
      child: IgnorePointer(
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: const [BoxShadow(color: Colors.black38, blurRadius: 8, offset: Offset(0, 2))],
              ),
              child: Text(label,
                  style: const TextStyle(color: Color(0xFF222222), fontSize: 15, fontWeight: FontWeight.bold)),
            ),
            CustomPaint(size: const Size(12, 7), painter: _BubbleTailPainter()),
          ],
        ),
      ),
    );
  }
}

class _BubbleTailPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final path = Path()
      ..moveTo(size.width / 2 - 6, 0)
      ..lineTo(size.width / 2 + 6, 0)
      ..lineTo(size.width / 2, size.height)
      ..close();
    canvas.drawPath(path, Paint()..color = Colors.white);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
