import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../theme/glory_theme.dart';
import '../utils/orientation_lock.dart';
import '../widgets/notation_widget.dart';
import '../widgets/mini_piano_widget.dart';
import '../widgets/octave_zone_piano.dart';

final _audio = AudioService.instance;
const _pageCount = 8;

const _beatColors = [
  Color(0xFFFF6B35), Color(0xFF7BC67E), Color(0xFF5BC0EB), Color(0xFFC97FD6),
];

const _solfege = {
  'C': '도', 'C#': '도#', 'D': '레', 'D#': '레#', 'E': '미', 'F': '파', 'F#': '파#',
  'G': '솔', 'G#': '솔#', 'A': '라', 'A#': '라#', 'B': '시',
};

String _solfegeOf(String pitch) => _solfege[pitch.substring(0, pitch.length - 1)] ?? pitch;

const _introNotes = [
  ScoreNote(pitch: 'C4', duration: 0.5, beat: 1),
  ScoreNote(pitch: 'E4', duration: 0.5, beat: 1),
  ScoreNote(pitch: 'G4', duration: 1, beat: 2),
  ScoreNote(pitch: 'C5', duration: 2, beat: 3),
];

// 규칙1: 6옥+ / 5옥 / 4옥 이하 3구역을 실제로 모두 지나가는 예시.
const _rule1Notes = [
  ScoreNote(pitch: 'C6', duration: 1, beat: 1),
  ScoreNote(pitch: 'G5', duration: 1, beat: 2),
  ScoreNote(pitch: 'E4', duration: 1, beat: 3),
  ScoreNote(pitch: 'C4', duration: 1, beat: 4),
];

// 규칙2+3: 서로 다른 음 3개 + 서로 다른 길이 + 서로 다른 박자 색.
const _rule23Notes = [
  ScoreNote(pitch: 'G4', duration: 0.5, beat: 1),
  ScoreNote(pitch: 'C5', duration: 1, beat: 2),
  ScoreNote(pitch: 'E4', duration: 2, beat: 3),
];

const _chordNote = ScoreNote(pitch: 'C4', duration: 2, beat: 1, chordNotes: ['E4']);

(String, String) _spanOf(List<String> pitches) {
  var lo = pitches.first, hi = pitches.first;
  for (final p in pitches) {
    if (pitchOrder(p) < pitchOrder(lo)) lo = p;
    if (pitchOrder(p) > pitchOrder(hi)) hi = p;
  }
  return (lo, hi);
}

class TutorialScreen extends StatefulWidget {
  const TutorialScreen({super.key});

  @override
  State<TutorialScreen> createState() => _TutorialScreenState();
}

class _TutorialScreenState extends State<TutorialScreen> {
  final _pageCtrl = PageController();
  int _page = 0;

  @override
  void initState() {
    super.initState();
    lockLandscape();
  }

  @override
  void dispose() {
    _pageCtrl.dispose();
    lockPortrait();
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
            Expanded(
              child: PageView(
                controller: _pageCtrl,
                onPageChanged: (p) => setState(() => _page = p),
                children: const [
                  _IntroComparePage(),
                  _RuleZeroPage(),
                  _Rule1Page(),
                  _Rule23Page(),
                  _ChordRulePage(),
                  _TestPage(index: 1, targetNote: 'E4'),
                  _TestPage(index: 2, targetNote: 'G#4'),
                  _TestPage(index: 3, targetNote: 'C5'),
                ],
              ),
            ),
            _buildNavButtons(),
          ],
        ),
      ),
    );
  }

  // 가로모드는 세로 여백이 부족하므로 헤더 + 페이지 인디케이터를 한 줄로 압축.
  Widget _buildHeader(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 6, 16, 4),
      child: Row(
        children: [
          IconButton(
            icon: Icon(Icons.arrow_back, color: gloryInk),
            onPressed: () => Navigator.of(context).pop(),
            visualDensity: VisualDensity.compact,
          ),
          const SizedBox(width: 2),
          Text('커스텀 악보 읽는 법',
              style: TextStyle(color: gloryInk, fontSize: 15, fontWeight: FontWeight.bold)),
          const Spacer(),
          Row(
            children: List.generate(_pageCount, (i) => AnimatedContainer(
              duration: const Duration(milliseconds: 250),
              margin: const EdgeInsets.symmetric(horizontal: 2.5),
              width: _page == i ? 18 : 6,
              height: 6,
              decoration: BoxDecoration(
                color: _page == i ? gloryAccent : gloryBorder,
                borderRadius: BorderRadius.circular(4),
              ),
            )),
          ),
        ],
      ),
    );
  }

  Widget _buildNavButtons() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
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
          if (_page < _pageCount - 1)
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

// ── 공용 페이지 프레임 ────────────────────────────────────────────────────────

// 가로모드 화면은 세로 공간이 좁고 폭이 넓으므로, 왼쪽(칩/제목/설명) + 오른쪽(실제 인터랙션
// 콘텐츠)의 2컬럼으로 배치한다. 모든 _XxxPage가 이 프레임만 감싸 쓰므로 여기 한 곳만 고치면
// 튜토리얼 8페이지 전체에 적용된다.
class _PageFrame extends StatelessWidget {
  final String? chip;
  final String title;
  final String description;
  final Widget child;
  const _PageFrame({this.chip, required this.title, required this.description, required this.child});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 210,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                if (chip != null) ...[
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: gloryAccent.withAlpha(30),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: gloryAccent, width: 1),
                    ),
                    child: Text(chip!,
                        style: TextStyle(color: gloryAccent, fontSize: 12, fontWeight: FontWeight.bold)),
                  ),
                  const SizedBox(height: 8),
                ],
                Text(title, style: TextStyle(color: gloryInk, fontSize: 17, fontWeight: FontWeight.bold)),
                const SizedBox(height: 4),
                Text(description, style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 12, height: 1.35)),
              ],
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: SingleChildScrollView(child: child),
          ),
        ],
      ),
    );
  }
}

// ── 인트로: 오선보 vs 커스텀 악보 비교 (피아노 없음) ──────────────────────────

class _IntroComparePage extends StatelessWidget {
  const _IntroComparePage();

  @override
  Widget build(BuildContext context) {
    return _PageFrame(
      chip: '시작하기 전에',
      title: '악보, 이렇게 복잡했나요?',
      description: '오선지 위의 기호를 다 외워야 읽을 수 있는 전통 악보 대신,\n색과 위치만으로 바로 읽을 수 있는 커스텀 악보를 써보세요.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('전통 오선 악보 (같은 한 마디)',
              style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 11.5, fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          const _StaffPreview(notes: _introNotes),
          const SizedBox(height: 14),
          Text('커스텀 악보 (같은 한 마디)',
              style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 11.5, fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          const NotationWidget(notes: _introNotes),
        ],
      ),
    );
  }
}

// ── 규칙 0: 흰/검은 건반과 12음 매칭 ──────────────────────────────────────────

class _RuleZeroPage extends StatelessWidget {
  const _RuleZeroPage();

  @override
  Widget build(BuildContext context) {
    return _PageFrame(
      chip: '규칙 0',
      title: '먼저, 건반부터 익혀요',
      description: '흰 건반 7개 + 검은 건반 5개 = 한 옥타브 12음. 눌러서 소리를 들어보세요.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 8),
          MiniPianoWidget(
            lowNote: 'C4',
            highNote: 'C5',
            detailed: true,
            onKeyDown: (note) {
              _audio.unlock();
              _audio.playNote(note);
            },
          ),
          const SizedBox(height: 14),
          Text('흰 건반은 음이름 그대로(도레미파솔라시), 검은 건반은 낮은 음부터 순서대로 1~5 숫자로 표시해요.',
              style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12, height: 1.4)),
        ],
      ),
    );
  }
}

// ── 규칙 1: 세로 위치 = 음 높이 ────────────────────────────────────────────────

class _Rule1Page extends StatelessWidget {
  const _Rule1Page();

  @override
  Widget build(BuildContext context) {
    return _PageFrame(
      chip: '규칙 1',
      title: '세로 위치 = 음 높이',
      description: '셀이 위에 있을수록 높은 음이에요. 3개 구역의 배경색도 함께 봐주세요.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const NotationWidget(notes: _rule1Notes),
          const SizedBox(height: 10),
          const Wrap(
            spacing: 14,
            runSpacing: 4,
            children: [
              _ZoneLabel(color: zoneHighColor, label: '높음 (6옥타브+)'),
              _ZoneLabel(color: zoneMidColor, label: '중간 (5옥타브)'),
              _ZoneLabel(color: zoneLowColor, label: '낮음 (4옥타브 이하)'),
            ],
          ),
          const SizedBox(height: 10),
          Text('88건반 전체 위에 구역을 표시했어요. 반투명한 색이 덮인 부분이 지금 봐야 할 구역이에요.',
              style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 11.5, height: 1.4)),
          const SizedBox(height: 8),
          OctaveZonePianoWidget(
            onKeyDown: (note) {
              _audio.unlock();
              _audio.playNote(note);
            },
          ),
        ],
      ),
    );
  }
}

// ── 규칙 2+3: 셀 너비 = 음 길이 / 테두리 색 = 박자 위치 (통합) ─────────────────

class _Rule23Page extends StatelessWidget {
  const _Rule23Page();

  @override
  Widget build(BuildContext context) {
    return _PageFrame(
      chip: '규칙 2 + 3',
      title: '셀 너비 = 음 길이, 테두리 색 = 박자',
      description: '넓을수록 길게, 테두리 색으로 마디 안 몇 박자인지 알 수 있어요.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const NotationWidget(notes: _rule23Notes),
          const SizedBox(height: 10),
          Wrap(
            spacing: 14,
            runSpacing: 4,
            children: [
              for (int i = 0; i < 4; i++)
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(width: 12, height: 12,
                        decoration: BoxDecoration(color: _beatColors[i], borderRadius: BorderRadius.circular(3))),
                    const SizedBox(width: 5),
                    Text('${i + 1}박', style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 12)),
                  ],
                ),
            ],
          ),
          const SizedBox(height: 10),
          _NotePracticeDock(notes: _rule23Notes.map((n) => n.pitch).toList()),
        ],
      ),
    );
  }
}

// ── 규칙 4: 화음 (두 음을 동시에) ─────────────────────────────────────────────

class _ChordRulePage extends StatelessWidget {
  const _ChordRulePage();

  @override
  Widget build(BuildContext context) {
    final chordPitches = [_chordNote.pitch, ..._chordNote.chordNotes];
    final span = _spanOf(chordPitches);
    return _PageFrame(
      chip: '규칙 4',
      title: '화음 = 두 음을 동시에',
      description: '세로로 겹쳐 쓴 셀은 두 음을 같이 눌러요. 아래 피아노에서 두 건반이 함께 강조돼요.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const NotationWidget(notes: [_chordNote]),
          const SizedBox(height: 14),
          Text('실제로 눌러보기', style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 11.5, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          MiniPianoWidget(
            lowNote: span.$1,
            highNote: span.$2,
            highlight: chordPitches.toSet(),
            onKeyDown: (note) {
              _audio.unlock();
              _audio.playNote(note);
            },
          ),
        ],
      ),
    );
  }
}

// ── 테스트 3장면: 힌트 없이 정답 건반 찾기 ────────────────────────────────────

class _TestPage extends StatefulWidget {
  final int index;
  final String targetNote;
  const _TestPage({required this.index, required this.targetNote});

  @override
  State<_TestPage> createState() => _TestPageState();
}

class _TestPageState extends State<_TestPage> {
  String? _wrong;
  bool _correct = false;

  void _onDown(String note) {
    _audio.unlock();
    _audio.playNote(note);
    if (note == widget.targetNote) {
      setState(() {
        _wrong = null;
        _correct = true;
      });
    } else {
      setState(() {
        _wrong = note;
        _correct = false;
      });
    }
  }

  void _onUp(String note) {
    if (note == _wrong) setState(() => _wrong = null);
  }

  @override
  Widget build(BuildContext context) {
    final span = _spanOf([widget.targetNote, 'C4', 'C5']);
    return _PageFrame(
      chip: '테스트 ${widget.index} / 3',
      title: '어느 건반을 눌러야 할까요?',
      description: '힌트 없이, 아래 커스텀 악보가 가리키는 음을 피아노에서 직접 찾아보세요.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          NotationWidget(notes: [ScoreNote(pitch: widget.targetNote, duration: 2, beat: 1)]),
          const SizedBox(height: 10),
          Row(
            children: [
              if (_wrong != null)
                const Text('✕ 다시 시도해보세요', style: TextStyle(color: Color(0xFFE64545), fontSize: 12.5, fontWeight: FontWeight.bold))
              else if (_correct)
                const Text('✓ 정답이에요!', style: TextStyle(color: Color(0xFF3E8F45), fontSize: 12.5, fontWeight: FontWeight.bold))
              else
                Text('건반을 눌러 확인해보세요', style: TextStyle(color: gloryInk.withValues(alpha: .4), fontSize: 12.5)),
            ],
          ),
          const SizedBox(height: 8),
          MiniPianoWidget(
            lowNote: span.$1,
            highNote: span.$2,
            wrongNote: _wrong,
            highlight: _correct ? {widget.targetNote} : const {},
            onKeyDown: _onDown,
            onKeyUp: _onUp,
          ),
        ],
      ),
    );
  }
}

// ── 실시간 음 맞추기 독(dock) — 해당 페이지 예시 음을 그대로 순서대로 연주 ─────

class _NotePracticeDock extends StatefulWidget {
  final List<String> notes;
  const _NotePracticeDock({required this.notes});

  @override
  State<_NotePracticeDock> createState() => _NotePracticeDockState();
}

class _NotePracticeDockState extends State<_NotePracticeDock> {
  int _idx = 0;
  String? _pianoNote;
  String? _wrong;
  bool _justCorrect = false;

  String get _target => widget.notes[_idx];

  void _onDown(String note) {
    _audio.unlock();
    _audio.playNote(note);
    if (note == _target) {
      setState(() {
        _wrong = null;
        _pianoNote = note;
        _justCorrect = true;
      });
    } else {
      setState(() => _wrong = note);
    }
  }

  void _onUp(String note) {
    if (note == _wrong) setState(() => _wrong = null);
    if (_justCorrect && note == _target) {
      setState(() {
        _justCorrect = false;
        _idx = (_idx + 1) % widget.notes.length;
        _pianoNote = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final span = _spanOf(widget.notes);
    return Container(
      decoration: BoxDecoration(
        color: glorySurface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: gloryBorder),
      ),
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 10),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('실시간으로 연주해보기',
                  style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 11, fontWeight: FontWeight.bold)),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                decoration: BoxDecoration(
                  color: gloryAccent.withAlpha(25),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: gloryAccent, width: 1),
                ),
                child: Text('다음 음: ${_solfegeOf(_target)}',
                    style: TextStyle(color: gloryAccent, fontSize: 10.5, fontWeight: FontWeight.bold)),
              ),
              const Spacer(),
              if (_wrong != null)
                const Text('✕', style: TextStyle(color: Color(0xFFE64545), fontSize: 14, fontWeight: FontWeight.bold))
              else if (_justCorrect)
                const Text('✓', style: TextStyle(color: Color(0xFF3E8F45), fontSize: 14, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 6),
          MiniPianoWidget(
            lowNote: span.$1,
            highNote: span.$2,
            highlight: {_pianoNote ?? _target},
            wrongNote: _wrong,
            onKeyDown: _onDown,
            onKeyUp: _onUp,
          ),
        ],
      ),
    );
  }
}

// ── 보조 위젯 ─────────────────────────────────────────────────────────────────

class _ZoneLabel extends StatelessWidget {
  final Color color;
  final String label;
  const _ZoneLabel({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(width: 14, height: 14,
            decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(3))),
        const SizedBox(width: 6),
        Text(label, style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 12)),
      ],
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
