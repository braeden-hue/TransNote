import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
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
  int _hlIdx = -1;
  String? _pianoNote;

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
      _hlIdx = -1;
      _pianoNote = null;
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0d0d1f),
      body: SafeArea(
        child: _selected == null ? _buildGrid() : _buildNotationView(),
      ),
    );
  }

  Widget _buildGrid() {
    return CustomScrollView(
      slivers: [
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text('악보 선택',
                    style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
                SizedBox(height: 4),
                Text('샘플 악보를 골라 커스텀 표기법으로 확인하세요',
                    style: TextStyle(color: Color(0xFF6060a0), fontSize: 13)),
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

  Widget _buildNotationView() {
    final s = _selected!;
    return Column(
      children: [
        // Top bar
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          color: const Color(0xFF12122a),
          child: Row(
            children: [
              IconButton(
                icon: const Icon(Icons.arrow_back, color: Colors.white),
                onPressed: () => setState(() {
                  _selected = null;
                  _hlIdx = -1;
                  _pianoNote = null;
                }),
              ),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('${s.emoji} ${s.title}',
                        style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
                    Text('${s.tempo} BPM · ${s.timeSignature[0]}/${s.timeSignature[1]}박자 · ${s.notes.length}음',
                        style: const TextStyle(color: Color(0xFF6060a0), fontSize: 11)),
                  ],
                ),
              ),
            ],
          ),
        ),
        // Notation area
        Expanded(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('음표를 눌러보세요',
                      style: TextStyle(color: Color(0xFF6060a0), fontSize: 12)),
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
          ),
        ),
        // Piano at bottom
        Container(
          decoration: const BoxDecoration(
            color: Color(0xFF0a0a1a),
            border: Border(top: BorderSide(color: Color(0xFF252550), width: 1)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
                child: Row(
                  children: [
                    const Text('피아노',
                        style: TextStyle(color: Color(0xFF6060a0), fontSize: 11, fontWeight: FontWeight.bold)),
                    const SizedBox(width: 8),
                    if (_pianoNote != null)
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: const Color(0xFFFF4444).withAlpha(40),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: const Color(0xFFFF4444), width: 1),
                        ),
                        child: Text(_pianoNote!,
                            style: const TextStyle(color: Color(0xFFFF4444), fontSize: 11, fontWeight: FontWeight.bold)),
                      ),
                  ],
                ),
              ),
              PianoWidget(
                highlightNote: _pianoNote,
                onKeyTap: (note) {
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
            Text(e.$2, style: const TextStyle(color: Color(0xFF6060a0), fontSize: 11)),
          ],
        ),
      )).toList(),
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
          color: const Color(0xFF12122a),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFF252550), width: 1),
        ),
        child: Stack(
          children: [
            // Mini beat preview at bottom
            Positioned(
              bottom: 0, left: 0, right: 0,
              child: ClipRRect(
                borderRadius: const BorderRadius.vertical(bottom: Radius.circular(12)),
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
                      style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.bold),
                      maxLines: 2, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 4),
                  Text('${sample.notes.length}음 · ${sample.tempo}BPM',
                      style: TextStyle(color: color.withAlpha(180), fontSize: 11)),
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
        color: const Color(0xFF1a1a35),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF252550)),
      ),
      child: Row(
        children: [
          Text(pitch,
              style: const TextStyle(color: Color(0xFFFF6B35), fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(width: 12),
          Text(solfege,
              style: const TextStyle(color: Colors.white, fontSize: 16)),
          const SizedBox(width: 8),
          Text('($oct옥타브)',
              style: const TextStyle(color: Color(0xFF6060a0), fontSize: 12)),
          const Spacer(),
          const Icon(Icons.touch_app, color: Color(0xFF5BC0EB), size: 16),
          const SizedBox(width: 4),
          const Text('건반에 표시됨',
              style: TextStyle(color: Color(0xFF5BC0EB), fontSize: 11)),
        ],
      ),
    );
  }
}
