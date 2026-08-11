import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../widgets/notation_widget.dart';

const _treblePitchZoneColors = [Color(0xFF0076CE), Color(0xFF3A9EE0), Color(0xFF999999)];

class _TutorialStep {
  final String title;
  final String body;
  final List<NoteEvent> demoNotes;
  final List<Color>? zoneColors;
  final bool hideNoteNames;

  const _TutorialStep({
    required this.title,
    required this.body,
    required this.demoNotes,
    this.zoneColors,
    this.hideNoteNames = false,
  });
}

final _steps = <_TutorialStep>[
  _TutorialStep(
    title: '규칙 1 · 세로 위치 = 음높이',
    body: '보표는 위·중간·아래 3개 존으로 나뉘어요. 음이 높을수록 위쪽 존에, '
        '낮을수록 아래쪽 존에 나타납니다. 조표나 오선을 몰라도 색과 위치만 보면 돼요.',
    zoneColors: _treblePitchZoneColors,
    hideNoteNames: true,
    demoNotes: [
      NoteEvent(pitch: 'C6', duration: 1, beat: 1),
      NoteEvent(pitch: 'G5', duration: 1, beat: 2),
      NoteEvent(pitch: 'C5', duration: 1, beat: 3),
      NoteEvent(pitch: 'G4', duration: 1, beat: 4),
      NoteEvent(pitch: 'C4', duration: 1, beat: 1),
    ],
  ),
  const _TutorialStep(
    title: '규칙 2 · 가로 폭 = 음길이',
    body: '음표가 차지하는 칸의 너비가 곧 음의 길이예요. 4분음표는 1칸, '
        '2분음표는 2칸, 온음표는 4칸을 차지합니다.',
    demoNotes: [
      NoteEvent(pitch: 'C4', duration: 1, beat: 1),
      NoteEvent(pitch: 'D4', duration: 2, beat: 2),
      NoteEvent(pitch: 'E4', duration: 1, beat: 1),
      NoteEvent(pitch: 'F4', duration: 4, beat: 2),
    ],
  ),
  const _TutorialStep(
    title: '규칙 3 · 화음',
    body: '여러 음을 동시에 눌러야 하는 화음은 사각 박스로 표시되고, 안에 '
        '음 이름이 세로로 쌓여요. 흰건반은 알파벳, 검은건반은 숫자(1~5)로 읽습니다.',
    demoNotes: [
      NoteEvent(pitch: 'C4', duration: 2, beat: 1, chordNotes: ['E4', 'G4']),
      NoteEvent(pitch: 'D4', duration: 2, beat: 3, chordNotes: ['F4', 'A4']),
    ],
  ),
];

class TutorialScreen extends StatefulWidget {
  final VoidCallback? onFinish;
  const TutorialScreen({super.key, this.onFinish});

  @override
  State<TutorialScreen> createState() => _TutorialScreenState();
}

class _TutorialScreenState extends State<TutorialScreen> {
  final _pageController = PageController();
  int _page = 0;

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void _goTo(int page) {
    _pageController.animateToPage(page,
        duration: const Duration(milliseconds: 280), curve: Curves.easeOut);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: PageView.builder(
            controller: _pageController,
            itemCount: _steps.length,
            onPageChanged: (p) => setState(() => _page = p),
            itemBuilder: (context, i) {
              final step = _steps[i];
              return SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(step.title, style: Theme.of(context).textTheme.headlineSmall),
                    const SizedBox(height: 12),
                    Text(step.body, style: Theme.of(context).textTheme.bodyLarge),
                    const SizedBox(height: 24),
                    NotationWidget(
                      notes: step.demoNotes,
                      zoneColors: step.zoneColors,
                      hideNoteNames: step.hideNoteNames,
                    ),
                  ],
                ),
              );
            },
          ),
        ),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(
            _steps.length,
            (i) => Container(
              margin: const EdgeInsets.symmetric(horizontal: 4),
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: i == _page ? const Color(0xFF0076CE) : Colors.grey[300],
              ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              if (_page > 0)
                TextButton(onPressed: () => _goTo(_page - 1), child: const Text('이전')),
              const Spacer(),
              if (_page < _steps.length - 1)
                FilledButton(onPressed: () => _goTo(_page + 1), child: const Text('다음'))
              else
                FilledButton(
                  onPressed: widget.onFinish,
                  child: const Text('악보 체험하러 가기'),
                ),
            ],
          ),
        ),
      ],
    );
  }
}
