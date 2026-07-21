import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../theme/glory_theme.dart';
import '../widgets/notation_widget.dart';
import '../widgets/piano_widget.dart';

final _audio = AudioService.instance;

class ScoreScreen extends StatefulWidget {
  const ScoreScreen({super.key});

  @override
  State<ScoreScreen> createState() => _ScoreScreenState();
}

class _ScoreScreenState extends State<ScoreScreen> {
  SampleScore? _selected;
  bool _practiceMode = false;

  // ── 감상 모드 ────────────────────────────────────────────────────────────
  int _hlIdx = -1;
  String? _pianoNote;

  // ── 연주 모드 ────────────────────────────────────────────────────────────
  int _expectedIdx = 0;
  int _practiceHlIdx = -1;
  String? _wrongNote;
  bool _done = false;

  final _trebleCtrl = ScrollController();
  final _bassCtrl   = ScrollController();
  bool _syncing = false;

  @override
  void initState() {
    super.initState();
    _trebleCtrl.addListener(_syncFromTreble);
    _bassCtrl.addListener(_syncFromBass);
  }

  void _syncFromTreble() {
    if (_syncing || !_bassCtrl.hasClients) return;
    _syncing = true;
    _bassCtrl.jumpTo(_trebleCtrl.offset);
    _syncing = false;
  }

  void _syncFromBass() {
    if (_syncing || !_trebleCtrl.hasClients) return;
    _syncing = true;
    _trebleCtrl.jumpTo(_bassCtrl.offset);
    _syncing = false;
  }

  @override
  void dispose() {
    _trebleCtrl.dispose();
    _bassCtrl.dispose();
    super.dispose();
  }

  void _selectSample(SampleScore s) {
    setState(() {
      _selected = s;
      _practiceMode = false;
      _hlIdx = -1;
      _pianoNote = null;
      _resetPractice();
    });
  }

  void _resetPractice() {
    _expectedIdx = 0;
    _practiceHlIdx = -1;
    _wrongNote = null;
    _done = false;
  }

  void _setMode(bool practice) {
    setState(() {
      _practiceMode = practice;
      _hlIdx = -1;
      _pianoNote = null;
      _resetPractice();
    });
  }

  void _onNoteTap(int idx, ScoreNote note) {
    _audio.unlock();
    _audio.playNote(note.pitch);
    setState(() {
      _hlIdx = idx;
      _pianoNote = note.pitch;
    });
  }

  void _onPracticeKeyDown(String note) {
    if (_selected == null || _done) return;
    _audio.unlock();
    _audio.playNote(note);
    final expected = _selected!.notes[_expectedIdx];
    if (note == expected.pitch) {
      setState(() {
        _wrongNote = null;
        _practiceHlIdx = _expectedIdx;
      });
    } else {
      setState(() => _wrongNote = note);
    }
  }

  void _onPracticeKeyUp(String note) {
    if (note == _wrongNote) setState(() => _wrongNote = null);
    if (_selected != null && !_done) {
      final expected = _selected!.notes[_expectedIdx];
      if (note == expected.pitch && _practiceHlIdx == _expectedIdx) {
        final next = _expectedIdx + 1;
        setState(() {
          if (next >= _selected!.notes.length) {
            _done = true;
            _practiceHlIdx = -1;
          } else {
            _expectedIdx = next;
            _practiceHlIdx = -1;
          }
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: _selected == null ? _buildGrid(context) : _buildDetail(),
      ),
    );
  }

  Widget _buildGrid(BuildContext context) {
    return CustomScrollView(
      slivers: [
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(8, 12, 20, 12),
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
                      const Text('예시 악보 체험',
                          style: TextStyle(color: gloryInk, fontSize: 20, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 2),
                      Text('샘플 악보를 감상하거나 직접 연주해보세요',
                          style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        SliverPadding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          sliver: SliverGrid(
            delegate: SliverChildBuilderDelegate(
              (ctx, i) => _SampleCard(
                sample: samples[i],
                onTap: () => _selectSample(samples[i]),
              ),
              childCount: samples.length,
            ),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
              childAspectRatio: 1.4,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildDetail() {
    final s = _selected!;
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.fromLTRB(4, 8, 12, 10),
          decoration: const BoxDecoration(
            color: glorySurface,
            border: Border(bottom: BorderSide(color: gloryBorder)),
          ),
          child: Column(
            children: [
              Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back, color: gloryInk),
                    onPressed: () => setState(() {
                      _selected = null;
                      _practiceMode = false;
                    }),
                  ),
                  Expanded(
                    child: Text('${s.emoji} ${s.title}',
                        style: const TextStyle(color: gloryInk, fontSize: 15, fontWeight: FontWeight.bold),
                        maxLines: 1, overflow: TextOverflow.ellipsis),
                  ),
                  if (_practiceMode)
                    IconButton(
                      icon: Icon(Icons.refresh, color: gloryInk.withValues(alpha: .5)),
                      tooltip: '처음부터',
                      onPressed: () => setState(_resetPractice),
                    ),
                ],
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 4, 12, 0),
                child: _ModeToggle(practiceMode: _practiceMode, onChanged: _setMode),
              ),
            ],
          ),
        ),
        Expanded(
          child: _practiceMode
              ? (_done ? _buildDoneView() : _buildPracticeView(s))
              : _buildListenView(s),
        ),
        Container(
          decoration: const BoxDecoration(
            color: glorySurface,
            border: Border(top: BorderSide(color: gloryBorder, width: 1)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
                child: _practiceMode ? _buildPracticeBadgeRow(s) : _buildListenBadgeRow(),
              ),
              PianoWidget(
                highlightNote: _practiceMode
                    ? (_done ? null : (_practiceHlIdx >= 0 && _practiceHlIdx < s.notes.length ? s.notes[_practiceHlIdx].pitch : null))
                    : _pianoNote,
                wrongNote: _practiceMode ? _wrongNote : null,
                onKeyDown: _practiceMode ? _onPracticeKeyDown : null,
                onKeyUp: _practiceMode ? _onPracticeKeyUp : null,
                onKeyTap: _practiceMode
                    ? (_) {}
                    : (note) {
                        _audio.unlock();
                        _audio.playNote(note);
                        setState(() => _pianoNote = note);
                      },
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildListenBadgeRow() {
    return Row(
      children: [
        Text('피아노',
            style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 11, fontWeight: FontWeight.bold)),
        const SizedBox(width: 8),
        if (_pianoNote != null)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: const Color(0xFFFF4444).withAlpha(30),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFFFF4444), width: 1),
            ),
            child: Text(_pianoNote!,
                style: const TextStyle(color: Color(0xFFFF4444), fontSize: 11, fontWeight: FontWeight.bold)),
          ),
      ],
    );
  }

  Widget _buildPracticeBadgeRow(SampleScore s) {
    final total = s.notes.length;
    return Row(
      children: [
        Text('피아노 건반',
            style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 11, fontWeight: FontWeight.bold)),
        const SizedBox(width: 10),
        if (!_done && _expectedIdx < total) _ExpectedBadge(pitch: s.notes[_expectedIdx].pitch),
        if (_wrongNote != null) ...[
          const SizedBox(width: 8),
          _WrongBadge(pitch: _wrongNote!),
        ],
        const Spacer(),
        Text('${_done ? total : _expectedIdx} / $total',
            style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12)),
      ],
    );
  }

  Widget _buildListenView(SampleScore s) {
    return SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('음표를 눌러보세요',
                style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12)),
            const SizedBox(height: 8),
            NotationWidget(
              notes: s.notes,
              highlightIdx: _hlIdx,
              onNoteTap: _onNoteTap,
              timeSignature: s.timeSignature,
              scrollController: _trebleCtrl,
            ),
            if (_pianoNote != null) ...[
              const SizedBox(height: 12),
              _NoteInfoBar(pitch: _pianoNote!),
            ],
            const SizedBox(height: 12),
            _buildBeatLegend(),
          ],
        ),
      ),
    );
  }

  Widget _buildPracticeView(SampleScore s) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.arrow_upward, color: Color(0xFFC99400), size: 14),
              const SizedBox(width: 4),
              Text('금색 화살표 음을 피아노에서 찾아 누르세요',
                  style: TextStyle(color: gloryInk.withValues(alpha: .55), fontSize: 12)),
            ],
          ),
          if (_wrongNote != null) ...[
            const SizedBox(height: 4),
            Row(
              children: [
                const Icon(Icons.close, color: Color(0xFFFF4444), size: 14),
                const SizedBox(width: 4),
                Text('$_wrongNote 는 틀린 음입니다',
                    style: const TextStyle(color: Color(0xFFE64545), fontSize: 12)),
              ],
            ),
          ],
          const SizedBox(height: 8),
          NotationWidget(
            notes: s.notes,
            highlightIdx: _practiceHlIdx,
            expectedIdx: _expectedIdx,
            timeSignature: s.timeSignature,
            scrollController: _trebleCtrl,
          ),
        ],
      ),
    );
  }

  Widget _buildDoneView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text('🎉', style: TextStyle(fontSize: 64)),
          const SizedBox(height: 16),
          const Text('연주 완료!',
              style: TextStyle(color: gloryInk, fontSize: 28, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text('${_selected!.title} 를 완주했습니다',
              style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 15)),
          const SizedBox(height: 32),
          FilledButton.icon(
            onPressed: () => setState(_resetPractice),
            icon: const Icon(Icons.refresh),
            label: const Text('다시 연주하기'),
            style: gloryFilledButtonStyle(background: const Color(0xFF7BC67E)),
          ),
          const SizedBox(height: 12),
          OutlinedButton(
            onPressed: () => _setMode(false),
            style: gloryOutlinedButtonStyle(),
            child: const Text('감상하기로 전환'),
          ),
        ],
      ),
    );
  }

  Widget _buildBeatLegend() {
    final beats = [
      (const Color(0xFFFF6B35), '1박'),
      (const Color(0xFF7BC67E), '2박'),
      (const Color(0xFF5BC0EB), '3박'),
      (const Color(0xFFC97FD6), '4박'),
    ];
    return Row(
      children: beats.map((e) => Padding(
        padding: const EdgeInsets.only(right: 12),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 12, height: 12,
              decoration: BoxDecoration(color: e.$1, borderRadius: BorderRadius.circular(2)),
            ),
            const SizedBox(width: 4),
            Text(e.$2, style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 11)),
          ],
        ),
      )).toList(),
    );
  }
}

class _ModeToggle extends StatelessWidget {
  final bool practiceMode;
  final ValueChanged<bool> onChanged;
  const _ModeToggle({required this.practiceMode, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(color: gloryBg, borderRadius: BorderRadius.circular(12)),
      child: Row(
        children: [
          Expanded(child: _segment('감상하기', !practiceMode, () => onChanged(false))),
          Expanded(child: _segment('연주하기', practiceMode, () => onChanged(true))),
        ],
      ),
    );
  }

  Widget _segment(String label, bool selected, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(vertical: 8),
        decoration: BoxDecoration(
          color: selected ? gloryInk : Colors.transparent,
          borderRadius: BorderRadius.circular(9),
        ),
        alignment: Alignment.center,
        child: Text(
          label,
          style: TextStyle(
            color: selected ? gloryBg : gloryInk.withValues(alpha: .5),
            fontSize: 13,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}

class _SampleCard extends StatelessWidget {
  final SampleScore sample;
  final VoidCallback onTap;
  const _SampleCard({required this.sample, required this.onTap});

  static const _colors = [
    Color(0xFFFF6B35), Color(0xFF7BC67E), Color(0xFF5BC0EB),
    Color(0xFFC97FD6), Color(0xFFFFD700),
  ];

  @override
  Widget build(BuildContext context) {
    final color = _colors[samples.indexOf(sample) % _colors.length];
    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          color: glorySurface,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [BoxShadow(color: gloryInk.withValues(alpha: .06), blurRadius: 10, offset: const Offset(0, 4))],
        ),
        child: Stack(
          children: [
            Positioned(
              bottom: 0, left: 0, right: 0,
              child: ClipRRect(
                borderRadius: const BorderRadius.vertical(bottom: Radius.circular(16)),
                child: SizedBox(
                  height: 6,
                  child: Row(
                    children: sample.notes.take(12).map((n) {
                      final c = Color(beatColorValues[n.beat] ?? 0xFF888888);
                      return Expanded(child: Container(color: c.withAlpha(120)));
                    }).toList(),
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(sample.emoji, style: const TextStyle(fontSize: 28)),
                  const SizedBox(height: 6),
                  Text(sample.title,
                      style: const TextStyle(color: gloryInk, fontSize: 13, fontWeight: FontWeight.bold),
                      maxLines: 2, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 4),
                  Text('${sample.notes.length}음 · ${sample.tempo}BPM',
                      style: TextStyle(color: color.withAlpha(220), fontSize: 11)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NoteInfoBar extends StatelessWidget {
  final String pitch;
  const _NoteInfoBar({required this.pitch});

  static const _noteLabels = {
    'C': '도', 'D': '레', 'E': '미', 'F': '파',
    'G': '솔', 'A': '라', 'B': '시',
    'C#': '도#', 'D#': '레#', 'F#': '파#', 'G#': '솔#', 'A#': '라#',
  };

  @override
  Widget build(BuildContext context) {
    final name = pitch.substring(0, pitch.length - 1);
    final oct = pitch[pitch.length - 1];
    final solfege = _noteLabels[name] ?? name;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: glorySurface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: gloryBorder),
      ),
      child: Row(
        children: [
          Text(pitch,
              style: const TextStyle(color: gloryAccent, fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(width: 12),
          Text(solfege,
              style: const TextStyle(color: gloryInk, fontSize: 16)),
          const SizedBox(width: 8),
          Text('($oct옥타브)',
              style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12)),
          const Spacer(),
          const Icon(Icons.touch_app, color: gloryAccent, size: 16),
          const SizedBox(width: 4),
          const Text('건반에 표시됨',
              style: TextStyle(color: gloryAccent, fontSize: 11)),
        ],
      ),
    );
  }
}

class _ExpectedBadge extends StatelessWidget {
  final String pitch;
  const _ExpectedBadge({required this.pitch});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: const Color(0xFFC99400).withAlpha(25),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFC99400), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('▲ ', style: TextStyle(color: Color(0xFFC99400), fontSize: 10)),
          Text(pitch,
              style: const TextStyle(
                  color: Color(0xFFC99400), fontSize: 11, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}

class _WrongBadge extends StatelessWidget {
  final String pitch;
  const _WrongBadge({required this.pitch});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: const Color(0xFFFF4444).withAlpha(25),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFF4444), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('✕ ', style: TextStyle(color: Color(0xFFFF4444), fontSize: 10)),
          Text(pitch,
              style: const TextStyle(
                  color: Color(0xFFFF4444), fontSize: 11, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
