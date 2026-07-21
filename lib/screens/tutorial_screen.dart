import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../theme/glory_theme.dart';
import '../widgets/notation_widget.dart';
import '../widgets/mini_piano_widget.dart';

final _audio = AudioService.instance;

const _solfege = {
  'C': '도', 'D': '레', 'E': '미', 'F': '파', 'G': '솔', 'A': '라', 'B': '시',
  'F#': '파#', 'A#': '라#',
};

/// 튜토리얼 내내 연습할 한 옥타브 범위의 음(흰 건반+검은 건반). 화면 하단 고정 미니 피아노와 범위가 일치해야 함.
const _matchNotes = ['C4', 'D4', 'E4', 'F#4', 'G4', 'A#4', 'C5'];

/// 시작 화면의 "다소 복잡한 한 마디" — 오선 악보와 커스텀 악보가 반드시 같은 음을 보여줘야 비교가 성립함.
const _introNotes = [
  ScoreNote(pitch: 'C4', duration: 0.5, beat: 1),
  ScoreNote(pitch: 'E4', duration: 0.5, beat: 1),
  ScoreNote(pitch: 'G4', duration: 1, beat: 2),
  ScoreNote(pitch: 'C5', duration: 2, beat: 3),
];

class TutorialScreen extends StatefulWidget {
  const TutorialScreen({super.key});

  @override
  State<TutorialScreen> createState() => _TutorialScreenState();
}

class _TutorialScreenState extends State<TutorialScreen> {
  final _pageCtrl = PageController();
  int _page = 0;

  // ── 하단 고정 피아노: 실시간 음 맞추기 ──────────────────────────────────────
  int _targetIdx = 0;
  String? _pianoNote;
  String? _wrongNote;
  bool _justCorrect = false;

  String get _targetNote => _matchNotes[_targetIdx];

  void _onMatchKeyDown(String note) {
    _audio.unlock();
    _audio.playNote(note);
    if (note == _targetNote) {
      setState(() {
        _wrongNote = null;
        _pianoNote = note;
        _justCorrect = true;
      });
    } else {
      setState(() => _wrongNote = note);
    }
  }

  void _onMatchKeyUp(String note) {
    if (note == _wrongNote) setState(() => _wrongNote = null);
    if (_justCorrect && note == _targetNote) {
      setState(() {
        _justCorrect = false;
        _targetIdx = (_targetIdx + 1) % _matchNotes.length;
        _pianoNote = null;
      });
    }
  }

  @override
  void dispose() {
    _pageCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(context),
            _buildPageIndicator(),
            Expanded(
              child: PageView(
                controller: _pageCtrl,
                onPageChanged: (p) => setState(() => _page = p),
                children: [
                  _buildIntroCompare(),
                  _buildRule1(),
                  _buildRule2(),
                  _buildRule3(),
                ],
              ),
            ),
            if (_page > 0) _buildPianoDock(),
            _buildNavButtons(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 12, 20, 0),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back, color: gloryInk),
            onPressed: () => Navigator.of(context).pop(),
          ),
          const SizedBox(width: 4),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('커스텀 악보 읽는 법',
                    style: TextStyle(color: gloryInk, fontSize: 20, fontWeight: FontWeight.bold)),
                const SizedBox(height: 2),
                Text('3가지 규칙만 알면 누구나 읽을 수 있어요',
                    style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPageIndicator() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: List.generate(4, (i) => AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          margin: const EdgeInsets.symmetric(horizontal: 4),
          width: _page == i ? 24 : 8,
          height: 8,
          decoration: BoxDecoration(
            color: _page == i ? gloryAccent : gloryBorder,
            borderRadius: BorderRadius.circular(4),
          ),
        )),
      ),
    );
  }

  Widget _buildIntroCompare() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: gloryAccent.withAlpha(30),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: gloryAccent, width: 1),
            ),
            child: const Text('시작하기 전에',
                style: TextStyle(color: gloryAccent, fontSize: 12, fontWeight: FontWeight.bold)),
          ),
          const SizedBox(height: 12),
          const Text('악보, 이렇게 복잡했나요?',
              style: TextStyle(color: gloryInk, fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text('오선지 위의 기호를 다 외워야 읽을 수 있는 전통 악보 대신,\n색과 위치만으로 바로 읽을 수 있는 커스텀 악보를 써보세요.',
              style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 14, height: 1.5)),
          const SizedBox(height: 20),
          Text('전통 오선 악보 (같은 한 마디)', style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const _StaffPreview(notes: _introNotes),
          const SizedBox(height: 20),
          Text('커스텀 악보 (같은 한 마디)', style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const NotationWidget(notes: _introNotes),
        ],
      ),
    );
  }

  Widget _buildRule1() {
    return _RulePage(
      number: '규칙 1',
      title: '세로 위치 = 음 높이',
      description: '셀이 위에 있을수록 높은 음이에요.\n3개 구역으로 나뉩니다.',
      child: Column(
        children: [
          const SizedBox(height: 16),
          NotationWidget(
            notes: const [
              ScoreNote(pitch: 'C5', duration: 1, beat: 1),
              ScoreNote(pitch: 'G4', duration: 1, beat: 2),
              ScoreNote(pitch: 'C4', duration: 1, beat: 3),
              ScoreNote(pitch: 'G3', duration: 1, beat: 4),
            ],
          ),
          const SizedBox(height: 16),
          _ZoneLabel(color: const Color(0xFF5BC0EB), label: '높음 (5옥타브+)'),
          _ZoneLabel(color: const Color(0xFF7BC67E), label: '중간 (4옥타브)'),
          _ZoneLabel(color: const Color(0xFFFF6B35), label: '낮음 (3옥타브 이하)'),
        ],
      ),
    );
  }

  Widget _buildRule2() {
    return _RulePage(
      number: '규칙 2',
      title: '셀 너비 = 음 길이',
      description: '셀이 넓을수록 길게 누르는 음이에요.',
      child: Column(
        children: [
          const SizedBox(height: 16),
          NotationWidget(
            notes: const [
              ScoreNote(pitch: 'C4', duration: 0.5, beat: 3),
              ScoreNote(pitch: 'C4', duration: 1, beat: 1),
              ScoreNote(pitch: 'C4', duration: 2, beat: 2),
            ],
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _DurationLabel(label: '8분음표', width: 0.5),
              _DurationLabel(label: '4분음표', width: 1.0),
              _DurationLabel(label: '2분음표', width: 2.0),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRule3() {
    final beatData = [
      (const Color(0xFFFF6B35), '1박'),
      (const Color(0xFF7BC67E), '2박'),
      (const Color(0xFF5BC0EB), '3박'),
      (const Color(0xFFC97FD6), '4박'),
    ];
    return _RulePage(
      number: '규칙 3',
      title: '테두리 색 = 박자 위치',
      description: '색으로 마디 안에서 어느 박자인지 알 수 있어요.\n아래 피아노로 직접 눌러보세요!',
      child: Column(
        children: [
          const SizedBox(height: 20),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            alignment: WrapAlignment.center,
            children: beatData.map((e) => GestureDetector(
              onTap: () {
                _audio.unlock();
                _audio.playNote('C4');
              },
              child: Container(
                width: 72,
                height: 52,
                decoration: BoxDecoration(
                  color: e.$1.withAlpha(30),
                  border: Border.all(color: e.$1, width: 2.5),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(e.$2,
                        style: TextStyle(color: e.$1, fontSize: 15, fontWeight: FontWeight.bold)),
                    Text('C', style: TextStyle(color: e.$1, fontSize: 11)),
                  ],
                ),
              ),
            )).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildPianoDock() {
    return Container(
      decoration: const BoxDecoration(
        color: glorySurface,
        border: Border(top: BorderSide(color: gloryBorder)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 10),
            child: Row(
              children: [
                Text('실시간 음 맞추기',
                    style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 11, fontWeight: FontWeight.bold)),
                const SizedBox(width: 10),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: gloryAccent.withAlpha(25),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: gloryAccent, width: 1),
                  ),
                  child: Text(
                    '다음 음: ${_solfege[_targetNote.substring(0, _targetNote.length - 1)] ?? _targetNote}',
                    style: const TextStyle(color: gloryAccent, fontSize: 11, fontWeight: FontWeight.bold),
                  ),
                ),
                const Spacer(),
                if (_wrongNote != null)
                  const Text('✕ 다시 시도', style: TextStyle(color: Color(0xFFE64545), fontSize: 11, fontWeight: FontWeight.bold))
                else if (_justCorrect)
                  const Text('✓ 정답!', style: TextStyle(color: Color(0xFF3E8F45), fontSize: 11, fontWeight: FontWeight.bold)),
              ],
            ),
          ),
          MiniPianoWidget(
            highlightNote: _pianoNote ?? _targetNote,
            wrongNote: _wrongNote,
            onKeyDown: _onMatchKeyDown,
            onKeyUp: _onMatchKeyUp,
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }

  Widget _buildNavButtons() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          if (_page > 0)
            OutlinedButton(
              onPressed: () => _pageCtrl.previousPage(
                  duration: const Duration(milliseconds: 300), curve: Curves.easeInOut),
              style: gloryOutlinedButtonStyle(),
              child: const Text('← 이전'),
            ),
          const Spacer(),
          if (_page < 3)
            FilledButton(
              onPressed: () => _pageCtrl.nextPage(
                  duration: const Duration(milliseconds: 300), curve: Curves.easeInOut),
              style: gloryFilledButtonStyle(),
              child: const Text('다음 →'),
            ),
        ],
      ),
    );
  }
}

// 음이름(자연음) → 오선 계단 번호. E4(아래 첫째 줄)를 0으로 두고 한 칸(줄→칸)마다 1씩 증가.
const _staffBaseStep = {'C': -2, 'D': -1, 'E': 0, 'F': 1, 'G': 2, 'A': 3, 'B': 4};

class _StaffPreview extends StatelessWidget {
  final List<ScoreNote> notes;
  const _StaffPreview({required this.notes});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 110,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: gloryBorder),
      ),
      child: CustomPaint(painter: _StaffPainter(notes: notes)),
    );
  }
}

class _StaffPainter extends CustomPainter {
  final List<ScoreNote> notes;
  const _StaffPainter({required this.notes});

  @override
  void paint(Canvas canvas, Size size) {
    final linePaint = Paint()
      ..color = const Color(0xFF6E6259)
      ..strokeWidth = 1.2;
    final gap = size.height / 8;
    final staffTop = gap * 2;
    for (int i = 0; i < 5; i++) {
      final y = staffTop + i * gap;
      canvas.drawLine(Offset(28, y), Offset(size.width - 16, y), linePaint);
    }
    final bottomLineY = staffTop + 4 * gap; // E4, 오선 맨 아래 줄
    double stepToY(int step) => bottomLineY - step * (gap / 2);

    final totalDur = notes.fold<double>(0, (a, n) => a + n.duration);
    final usableW = size.width - 28 - 16 - 24;
    double x = 28 + 16;

    for (final note in notes) {
      final w = totalDur == 0 ? 0.0 : (note.duration / totalDur) * usableW;
      final cx = x + w / 2;
      final oct = int.parse(note.pitch[note.pitch.length - 1]);
      final namePart = note.pitch.substring(0, note.pitch.length - 1);
      final sharp = namePart.endsWith('#');
      final letter = sharp ? namePart.substring(0, 1) : namePart;
      final step = _staffBaseStep[letter]! + 7 * (oct - 4);
      final cy = stepToY(step);

      // 보표를 벗어난 음은 덧줄(ledger line)을 그림
      if (step < 0) {
        for (int s = -2; s >= step; s -= 2) {
          final ly = stepToY(s);
          canvas.drawLine(Offset(cx - 9, ly), Offset(cx + 9, ly), linePaint);
        }
      } else if (step > 8) {
        for (int s = 10; s <= step; s += 2) {
          final ly = stepToY(s);
          canvas.drawLine(Offset(cx - 9, ly), Offset(cx + 9, ly), linePaint);
        }
      }

      // 임시표(#)
      if (sharp) {
        final tp = TextPainter(
          text: const TextSpan(
            text: '♯',
            style: TextStyle(color: Color(0xFF222222), fontSize: 15, fontWeight: FontWeight.bold),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        tp.paint(canvas, Offset(cx - 10 - tp.width, cy - tp.height / 2 + 1));
      }

      // 음표 머리: 2박 이상은 속이 빈 온음표/2분음표 스타일
      final hollow = note.duration >= 2;
      final headPaint = Paint()..color = const Color(0xFF222222);
      if (hollow) {
        headPaint
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.6;
      }
      canvas.save();
      canvas.translate(cx, cy);
      canvas.rotate(-0.2);
      canvas.drawOval(const Rect.fromLTWH(-6.5, -4.6, 13, 9.2), headPaint);
      canvas.restore();

      // 기둥
      final stemX = cx + 6;
      final stemTopY = cy - 26;
      canvas.drawLine(Offset(stemX, cy), Offset(stemX, stemTopY), Paint()
        ..color = const Color(0xFF222222)
        ..strokeWidth = 1.4);

      // 8분음표 이하는 꼬리(flag) 표시
      if (note.duration <= 0.5) {
        final flag = Path()
          ..moveTo(stemX, stemTopY)
          ..quadraticBezierTo(stemX + 9, stemTopY + 4, stemX + 2, stemTopY + 14);
        canvas.drawPath(
          flag,
          Paint()
            ..color = const Color(0xFF222222)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.8,
        );
      }

      x += w;
    }
  }

  @override
  bool shouldRepaint(covariant _StaffPainter oldDelegate) => oldDelegate.notes != notes;
}

class _RulePage extends StatelessWidget {
  final String number, title, description;
  final Widget child;
  const _RulePage({required this.number, required this.title, required this.description, required this.child});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: gloryAccent.withAlpha(30),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: gloryAccent, width: 1),
            ),
            child: Text(number,
                style: const TextStyle(color: gloryAccent, fontSize: 12, fontWeight: FontWeight.bold)),
          ),
          const SizedBox(height: 12),
          Text(title,
              style: const TextStyle(color: gloryInk, fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(description,
              style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 14, height: 1.5)),
          child,
        ],
      ),
    );
  }
}

class _ZoneLabel extends StatelessWidget {
  final Color color;
  final String label;
  const _ZoneLabel({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(width: 16, height: 16,
              decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(3))),
          const SizedBox(width: 8),
          Text(label, style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 13)),
        ],
      ),
    );
  }
}

class _DurationLabel extends StatelessWidget {
  final String label;
  final double width;
  const _DurationLabel({required this.label, required this.width});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: width * 40,
          height: 30,
          decoration: BoxDecoration(
            color: glorySurface,
            border: Border.all(color: gloryAccent, width: 2),
            borderRadius: BorderRadius.circular(4),
          ),
        ),
        const SizedBox(height: 4),
        Text(label, style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 11)),
      ],
    );
  }
}
