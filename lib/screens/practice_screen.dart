import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../widgets/notation_widget.dart';
import '../widgets/piano_widget.dart';

final _audio = AudioService.instance;

class PracticeScreen extends StatefulWidget {
  const PracticeScreen({super.key});

  @override
  State<PracticeScreen> createState() => _PracticeScreenState();
}

class _PracticeScreenState extends State<PracticeScreen> {
  SampleScore? _score;
  int _expectedIdx = 0;  // 다음에 눌러야 할 음 인덱스
  int _highlightIdx = -1; // 방금 맞힌 음 인덱스 (잠깐 강조)
  String? _wrongNote;     // 틀리게 누르고 있는 건반 음 이름
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

  void _selectScore(SampleScore s) {
    setState(() {
      _score = s;
      _expectedIdx = 0;
      _highlightIdx = -1;
      _wrongNote = null;
      _done = false;
    });
  }

  void _reset() {
    setState(() {
      _expectedIdx = 0;
      _highlightIdx = -1;
      _wrongNote = null;
      _done = false;
    });
  }

  // 건반 누름
  void _onKeyDown(String note) {
    if (_score == null || _done) return;
    _audio.unlock();
    _audio.playNote(note);

    final expectedNote = _score!.notes[_expectedIdx];
    if (note == expectedNote.pitch) {
      // ── 정답 ─────────────────────────────────────────────────────────────
      setState(() {
        _wrongNote = null;
        _highlightIdx = _expectedIdx;
      });
    } else {
      // ── 오답 — wrongNote 세팅 → PianoWidget이 빨간 깜빡임 ──────────────
      setState(() => _wrongNote = note);
    }
  }

  // 건반 뗌
  void _onKeyUp(String note) {
    if (note == _wrongNote) {
      setState(() => _wrongNote = null);
    }
    // 정답 건반을 뗄 때 다음 음으로 진행
    if (_score != null && !_done) {
      final expectedNote = _score!.notes[_expectedIdx];
      if (note == expectedNote.pitch && _highlightIdx == _expectedIdx) {
        final next = _expectedIdx + 1;
        if (next >= _score!.notes.length) {
          setState(() {
            _done = true;
            _highlightIdx = -1;
          });
        } else {
          setState(() {
            _expectedIdx = next;
            _highlightIdx = -1;
          });
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0d0d1f),
      body: SafeArea(
        child: _score == null ? _buildSelection() : _buildPractice(),
      ),
    );
  }

  // ── 악보 선택 화면 ────────────────────────────────────────────────────────
  Widget _buildSelection() {
    return CustomScrollView(
      slivers: [
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text('연주 체험',
                    style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
                SizedBox(height: 4),
                Text('악보를 선택하고 피아노 건반으로 연주해보세요',
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
                onTap: () => _selectScore(samples[i]),
              ),
              childCount: samples.length,
            ),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
              childAspectRatio: 1.35,
            ),
          ),
        ),
      ],
    );
  }

  // ── 연주 체험 화면 ────────────────────────────────────────────────────────
  Widget _buildPractice() {
    final s = _score!;
    final isLastNote = _expectedIdx >= s.notes.length;
    final totalNotes = s.notes.length;
    final progress = isLastNote ? 1.0 : _expectedIdx / totalNotes;

    return Column(
      children: [
        // ── 상단 바 ────────────────────────────────────────────────────────
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          color: const Color(0xFF12122a),
          child: Row(
            children: [
              IconButton(
                icon: const Icon(Icons.arrow_back, color: Colors.white),
                onPressed: () => setState(() => _score = null),
              ),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('${s.emoji} ${s.title}',
                        style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 4),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(3),
                      child: LinearProgressIndicator(
                        value: progress,
                        backgroundColor: const Color(0xFF252550),
                        color: const Color(0xFF7BC67E),
                        minHeight: 5,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Text(
                '${_done ? totalNotes : _expectedIdx} / $totalNotes',
                style: const TextStyle(color: Color(0xFF6060a0), fontSize: 12),
              ),
              IconButton(
                icon: const Icon(Icons.refresh, color: Color(0xFF6060a0)),
                tooltip: '처음부터',
                onPressed: _reset,
              ),
            ],
          ),
        ),

        // ── 악보 영역 ─────────────────────────────────────────────────────
        Expanded(
          child: _done ? _buildDoneView() : _buildScoreView(s),
        ),

        // ── 피아노 ────────────────────────────────────────────────────────
        Container(
          decoration: const BoxDecoration(
            color: Color(0xFF0a0a1a),
            border: Border(top: BorderSide(color: Color(0xFF252550), width: 1)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
                child: Row(
                  children: [
                    const Text('피아노 건반',
                        style: TextStyle(color: Color(0xFF6060a0), fontSize: 11, fontWeight: FontWeight.bold)),
                    const SizedBox(width: 10),
                    if (!_done && _expectedIdx < s.notes.length)
                      _ExpectedBadge(pitch: s.notes[_expectedIdx].pitch),
                    if (_wrongNote != null) ...[
                      const SizedBox(width: 8),
                      _WrongBadge(pitch: _wrongNote!),
                    ],
                  ],
                ),
              ),
              PianoWidget(
                highlightNote: _done
                    ? null
                    : (_highlightIdx >= 0 && _highlightIdx < s.notes.length
                        ? s.notes[_highlightIdx].pitch
                        : null),
                wrongNote: _wrongNote,
                onKeyDown: _onKeyDown,
                onKeyUp: _onKeyUp,
                onKeyTap: (_) {},
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildScoreView(SampleScore s) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.arrow_upward, color: Color(0xFFFFD700), size: 14),
              const SizedBox(width: 4),
              const Text('금색 화살표 음을 피아노에서 찾아 누르세요',
                  style: TextStyle(color: Color(0xFF8080b0), fontSize: 12)),
            ],
          ),
          if (_wrongNote != null) ...[
            const SizedBox(height: 4),
            Row(
              children: [
                const Icon(Icons.close, color: Color(0xFFFF4444), size: 14),
                const SizedBox(width: 4),
                Text('$_wrongNote 는 틀린 음입니다',
                    style: const TextStyle(color: Color(0xFFFF6666), fontSize: 12)),
              ],
            ),
          ],
          const SizedBox(height: 8),
          NotationWidget(
            notes: s.notes,
            highlightIdx: _highlightIdx,
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
              style: TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text('${_score!.title} 를 완주했습니다',
              style: const TextStyle(color: Color(0xFF6060a0), fontSize: 15)),
          const SizedBox(height: 32),
          FilledButton.icon(
            onPressed: _reset,
            icon: const Icon(Icons.refresh),
            label: const Text('다시 연주하기'),
            style: FilledButton.styleFrom(backgroundColor: const Color(0xFF7BC67E)),
          ),
          const SizedBox(height: 12),
          OutlinedButton(
            onPressed: () => setState(() => _score = null),
            style: OutlinedButton.styleFrom(
                foregroundColor: const Color(0xFF5BC0EB),
                side: const BorderSide(color: Color(0xFF333360))),
            child: const Text('다른 악보 선택'),
          ),
        ],
      ),
    );
  }
}

// ── 보조 위젯 ─────────────────────────────────────────────────────────────────

class _ExpectedBadge extends StatelessWidget {
  final String pitch;
  const _ExpectedBadge({required this.pitch});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: const Color(0xFFFFD700).withAlpha(30),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFFD700), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('▲ ', style: TextStyle(color: Color(0xFFFFD700), fontSize: 10)),
          Text(pitch,
              style: const TextStyle(
                  color: Color(0xFFFFD700), fontSize: 11, fontWeight: FontWeight.bold)),
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
        color: const Color(0xFFFF4444).withAlpha(30),
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
            Positioned(
              bottom: 0, left: 0, right: 0,
              child: ClipRRect(
                borderRadius: const BorderRadius.vertical(bottom: Radius.circular(12)),
                child: SizedBox(
                  height: 5,
                  child: Row(
                    children: sample.notes.take(12).map((n) {
                      final c = Color(beatColorValues[n.beat] ?? 0xFF888888);
                      return Expanded(child: Container(color: c.withAlpha(100)));
                    }).toList(),
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(sample.emoji, style: const TextStyle(fontSize: 26)),
                  const SizedBox(height: 5),
                  Text(sample.title,
                      style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                      maxLines: 2, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 3),
                  Text('${sample.notes.length}음 · ${sample.tempo}BPM',
                      style: TextStyle(color: color.withAlpha(180), fontSize: 10)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
