import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../widgets/notation_widget.dart';
import '../widgets/piano_widget.dart';

final _audio = AudioService.instance;

class TutorialScreen extends StatefulWidget {
  const TutorialScreen({super.key});

  @override
  State<TutorialScreen> createState() => _TutorialScreenState();
}

class _TutorialScreenState extends State<TutorialScreen> {
  final _pageCtrl = PageController();
  int _page = 0;
  int _hlIdx = -1;
  String? _pianoNote;
  final _previewNotes = samples[0].notes.sublist(0, 7);

  static const _ruleColor = Color(0xFF5BC0EB);

  @override
  void dispose() {
    _pageCtrl.dispose();
    super.dispose();
  }

  void _onNoteTap(int idx, ScoreNote note) {
    _audio.unlock();
    _audio.playNote(note.pitch);
    setState(() {
      _hlIdx = idx;
      _pianoNote = note.pitch;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0d0d1f),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildPageIndicator(),
            Expanded(
              child: PageView(
                controller: _pageCtrl,
                onPageChanged: (p) => setState(() => _page = p),
                children: [
                  _buildRule1(),
                  _buildRule2(),
                  _buildRule3(),
                  _buildInteractive(),
                ],
              ),
            ),
            _buildNavButtons(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('커스텀 악보 읽는 법',
              style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
          SizedBox(height: 4),
          Text('3가지 규칙만 알면 누구나 읽을 수 있어요',
              style: TextStyle(color: Color(0xFF6060a0), fontSize: 13)),
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
            color: _page == i ? _ruleColor : const Color(0xFF333360),
            borderRadius: BorderRadius.circular(4),
          ),
        )),
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
      description: '색으로 마디 안에서 어느 박자인지 알 수 있어요.\n셀을 눌러 소리를 들어보세요!',
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

  Widget _buildInteractive() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('체험해보기',
              style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          const Text('악보의 음표를 눌러보세요. 피아노 건반에 표시됩니다.',
              style: TextStyle(color: Color(0xFF6060a0), fontSize: 12)),
          const SizedBox(height: 16),
          NotationWidget(
            notes: _previewNotes,
            highlightIdx: _hlIdx,
            onNoteTap: _onNoteTap,
          ),
          const SizedBox(height: 8),
          if (_pianoNote != null)
            Center(
              child: Text(
                '현재 음: $_pianoNote',
                style: const TextStyle(color: Color(0xFFFF6B35), fontSize: 13, fontWeight: FontWeight.bold),
              ),
            ),
          const SizedBox(height: 8),
          PianoWidget(
            highlightNote: _pianoNote,
            onKeyTap: (note) {
              _audio.unlock();
              _audio.playNote(note);
              setState(() => _pianoNote = note);
            },
          ),
          const SizedBox(height: 24),
          const Text('기본 음 체험 (클릭)',
              style: TextStyle(color: Color(0xFF8080b0), fontSize: 12)),
          const SizedBox(height: 10),
          _buildNoteButtons(),
        ],
      ),
    );
  }

  Widget _buildNoteButtons() {
    final noteData = [
      ('C4', '도', const Color(0xFFFF6B35)),
      ('D4', '레', const Color(0xFF7BC67E)),
      ('E4', '미', const Color(0xFF5BC0EB)),
      ('F4', '파', const Color(0xFFC97FD6)),
      ('G4', '솔', const Color(0xFFFF6B35)),
      ('A4', '라', const Color(0xFF7BC67E)),
      ('B4', '시', const Color(0xFF5BC0EB)),
    ];
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: noteData.map((e) => GestureDetector(
        onTap: () {
          _audio.unlock();
          _audio.playNote(e.$1);
          setState(() => _pianoNote = e.$1);
        },
        child: Container(
          width: 58,
          height: 64,
          decoration: BoxDecoration(
            color: e.$3.withAlpha(25),
            border: Border.all(color: e.$3, width: 2),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(e.$1.substring(0, e.$1.length - 1),
                  style: TextStyle(color: e.$3, fontSize: 16, fontWeight: FontWeight.bold)),
              Text(e.$2, style: TextStyle(color: e.$3.withAlpha(180), fontSize: 11)),
            ],
          ),
        ),
      )).toList(),
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
              style: OutlinedButton.styleFrom(foregroundColor: _ruleColor, side: const BorderSide(color: Color(0xFF333360))),
              child: const Text('← 이전'),
            ),
          const Spacer(),
          if (_page < 3)
            FilledButton(
              onPressed: () => _pageCtrl.nextPage(
                  duration: const Duration(milliseconds: 300), curve: Curves.easeInOut),
              style: FilledButton.styleFrom(backgroundColor: _ruleColor),
              child: Text(_page == 2 ? '체험하기 →' : '다음 →'),
            ),
        ],
      ),
    );
  }
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
              color: const Color(0xFF5BC0EB).withAlpha(40),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: const Color(0xFF5BC0EB), width: 1),
            ),
            child: Text(number,
                style: const TextStyle(color: Color(0xFF5BC0EB), fontSize: 12, fontWeight: FontWeight.bold)),
          ),
          const SizedBox(height: 12),
          Text(title,
              style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(description,
              style: const TextStyle(color: Color(0xFF8080b0), fontSize: 14, height: 1.5)),
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
          Text(label, style: const TextStyle(color: Color(0xFF8080b0), fontSize: 13)),
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
            color: const Color(0xFF16162e),
            border: Border.all(color: const Color(0xFF5BC0EB), width: 2),
            borderRadius: BorderRadius.circular(4),
          ),
        ),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(color: Color(0xFF8080b0), fontSize: 11)),
      ],
    );
  }
}
