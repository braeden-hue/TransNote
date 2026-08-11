import 'dart:async';
import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../widgets/notation_widget.dart';
import '../widgets/piano_widget.dart';

class PracticeScreen extends StatefulWidget {
  const PracticeScreen({super.key});

  @override
  State<PracticeScreen> createState() => _PracticeScreenState();
}

class _PracticeScreenState extends State<PracticeScreen> {
  Sample _sample = samples.first;
  final _pianoController = PianoController();
  late List<NoteEvent> _sequence;
  int _index = 0;
  final Set<String> _chordPressed = {};
  String _feedback = '';

  // trans-note.vercel.app "연주하기" 패널 참고 — ▶ 미리듣기(자동 재생) → Wait Mode(직접
  // 연주)로 이어지는 흐름.
  bool _isPreviewing = false;
  int _previewIdx = -1;
  Timer? _previewTimer;
  int _bpm = 90;
  bool _accompanimentMode = false;
  bool _showNoteNames = true;

  // 반주 모드 — 미리듣기(▶) 중에만 왼손(bass) 라인을 오른손과 별개 타이밍으로 같이
  // 재생한다. Wait Mode(사용자가 직접 연주)는 박자가 사용자 속도에 맞춰 늘어나므로
  // 반주를 동기화할 방법이 없어 오른손 연습만 그대로 유지한다.
  Timer? _bassTimer;
  int _bassIdx = 0;

  @override
  void initState() {
    super.initState();
    _loadSample(_sample);
  }

  @override
  void dispose() {
    _previewTimer?.cancel();
    _bassTimer?.cancel();
    _pianoController.dispose();
    super.dispose();
  }

  Stave? get _bassStave {
    final staves = _sample.staves;
    if (staves == null) return null;
    for (final s in staves) {
      if (s.clef == 'bass') return s;
    }
    return null;
  }

  void _loadSample(Sample s) {
    _stopPreview();
    setState(() {
      _sample = s;
      _sequence = s.notes;
      _index = 0;
      _chordPressed.clear();
      _feedback = '';
      _skipRests();
    });
  }

  void _skipRests() {
    while (_index < _sequence.length && _sequence[_index].isRest) {
      _index++;
    }
  }

  bool get _completed => _index >= _sequence.length;

  List<String> get _currentTargets {
    if (_completed) return const [];
    final n = _sequence[_index];
    return [n.pitch, ...n.chordNotes];
  }

  void _onKeyPress(String pitch) {
    if (_isPreviewing || _completed) return;
    audioService.playNote(pitch);
    final targets = _currentTargets;
    final note = _sequence[_index];

    if (note.isChord) {
      if (targets.contains(pitch)) {
        _chordPressed.add(pitch);
        _pianoController.flashCorrect(pitch);
        if (_chordPressed.containsAll(targets)) {
          _advance();
        } else {
          setState(
            () => _feedback = '✓ (${_chordPressed.length}/${targets.length})',
          );
        }
      } else {
        _chordPressed.clear();
        _pianoController.flashWrong(pitch);
        setState(() => _feedback = '✕ 다시 시도해보세요');
      }
    } else {
      if (pitch == targets.first) {
        _pianoController.flashCorrect(pitch);
        _advance();
      } else {
        _pianoController.flashWrong(pitch);
        setState(() => _feedback = '✕ 다시 시도해보세요');
      }
    }
  }

  void _advance() {
    setState(() {
      _chordPressed.clear();
      _index++;
      _skipRests();
      _feedback = _completed ? '🎉 완료했어요!' : '✓';
    });
  }

  void _togglePreview() {
    if (_isPreviewing) {
      _stopPreview();
    } else {
      _startPreview();
    }
  }

  void _startPreview() {
    setState(() {
      _isPreviewing = true;
      _previewIdx = 0;
    });
    _stepPreview();
    final bass = _bassStave;
    if (_accompanimentMode && bass != null) {
      _bassIdx = 0;
      _stepBassPreview(bass);
    }
  }

  void _stepPreview() {
    if (_previewIdx >= _sequence.length) {
      _stopPreview();
      return;
    }
    final note = _sequence[_previewIdx];
    if (!note.isRest) {
      audioService.playNote(note.pitch);
      for (final c in note.chordNotes) {
        audioService.playNote(c);
      }
      _pianoController.flashCorrect(note.pitch);
    }
    final beatMs = 60000 / _bpm;
    final delayMs = (beatMs * note.duration).round().clamp(80, 4000);
    _previewTimer = Timer(Duration(milliseconds: delayMs), () {
      if (!mounted) return;
      setState(() => _previewIdx++);
      _stepPreview();
    });
  }

  // 오른손(_stepPreview)과 별개 타이밍으로 왼손(bass)을 같은 BPM으로 재생 — 화면에
  // 표시되는 악보는 오른손 것 그대로라 왼손은 소리만 나고 하이라이트는 안 움직인다.
  void _stepBassPreview(Stave bass) {
    if (!_isPreviewing || _bassIdx >= bass.notes.length) return;
    final note = bass.notes[_bassIdx];
    if (!note.isRest) {
      audioService.playNote(note.pitch);
      for (final c in note.chordNotes) {
        audioService.playNote(c);
      }
    }
    final beatMs = 60000 / _bpm;
    final delayMs = (beatMs * note.duration).round().clamp(80, 4000);
    _bassTimer = Timer(Duration(milliseconds: delayMs), () {
      if (!mounted) return;
      _bassIdx++;
      _stepBassPreview(bass);
    });
  }

  void _stopPreview() {
    _previewTimer?.cancel();
    _bassTimer?.cancel();
    if (!mounted) return;
    setState(() {
      _isPreviewing = false;
      _previewIdx = -1;
    });
  }

  void _skipToStart() {
    _stopPreview();
    setState(() {
      _index = 0;
      _chordPressed.clear();
      _feedback = '';
      _skipRests();
    });
  }

  void _skipCurrentNote() {
    if (_isPreviewing) {
      _previewTimer?.cancel();
      setState(() => _previewIdx++);
      _stepPreview();
      return;
    }
    if (!_completed) _advance();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          height: 56,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            itemCount: samples.length,
            separatorBuilder: (context, i) => const SizedBox(width: 8),
            itemBuilder: (context, i) {
              final s = samples[i];
              final selected = s.id == _sample.id;
              return ChoiceChip(
                label: Text('${s.emoji} ${s.title}'),
                selected: selected,
                onSelected: (_) => _loadSample(s),
              );
            },
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(
            children: [
              Text(
                _completed ? '완료!' : '진행: ${_index + 1} / ${_sequence.length}',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
              const Spacer(),
              Text(
                _feedback,
                style: const TextStyle(fontWeight: FontWeight.w700),
              ),
              IconButton(
                icon: const Icon(Icons.replay),
                tooltip: '다시 시작',
                onPressed: () => _loadSample(_sample),
              ),
            ],
          ),
        ),
        _PlaybackControls(
          isPreviewing: _isPreviewing,
          bpm: _bpm,
          accompanimentMode: _accompanimentMode,
          accompanimentAvailable: _bassStave != null,
          showNoteNames: _showNoteNames,
          onSkipToStart: _skipToStart,
          onTogglePreview: _togglePreview,
          onSkipNote: _skipCurrentNote,
          onStop: _isPreviewing ? _stopPreview : null,
          onBpmChanged: (v) => setState(() => _bpm = v),
          onAccompanimentChanged: (v) {
            setState(() => _accompanimentMode = v);
            final bass = _bassStave;
            if (_isPreviewing && bass != null) {
              if (v) {
                _bassIdx = 0;
                _stepBassPreview(bass);
              } else {
                _bassTimer?.cancel();
              }
            }
          },
          onShowNoteNamesChanged: (v) => setState(() => _showNoteNames = v),
        ),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: NotationWidget(
              notes: _sequence,
              highlightIdx: _isPreviewing ? _previewIdx : -1,
              expectedIdx: _isPreviewing || _completed ? -1 : _index,
              hideNoteNames: !_showNoteNames,
            ),
          ),
        ),
        PianoDock(
          child: PianoWidget(
            controller: _pianoController,
            onKeyPress: _onKeyPress,
            expectedPitches: _isPreviewing || _completed
                ? const {}
                : _currentTargets.toSet(),
          ),
        ),
      ],
    );
  }
}

class _PlaybackControls extends StatelessWidget {
  final bool isPreviewing;
  final int bpm;
  final bool accompanimentMode;
  final bool accompanimentAvailable;
  final bool showNoteNames;
  final VoidCallback onSkipToStart;
  final VoidCallback onTogglePreview;
  final VoidCallback onSkipNote;
  final VoidCallback? onStop;
  final ValueChanged<int> onBpmChanged;
  final ValueChanged<bool> onAccompanimentChanged;
  final ValueChanged<bool> onShowNoteNamesChanged;

  const _PlaybackControls({
    required this.isPreviewing,
    required this.bpm,
    required this.accompanimentMode,
    required this.accompanimentAvailable,
    required this.showNoteNames,
    required this.onSkipToStart,
    required this.onTogglePreview,
    required this.onSkipNote,
    required this.onStop,
    required this.onBpmChanged,
    required this.onAccompanimentChanged,
    required this.onShowNoteNamesChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.fromLTRB(12, 4, 12, 8),
      color: Theme.of(context).colorScheme.surfaceContainerLow,
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Column(
          children: [
            Text(
              isPreviewing
                  ? '▶ 미리듣기 재생 중'
                  : '① ▶ 로 먼저 들어보고 → ② 정확한 건반을 눌러 직접 연주해보세요',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  icon: const Icon(Icons.skip_previous),
                  tooltip: '처음으로',
                  onPressed: onSkipToStart,
                ),
                IconButton.filled(
                  icon: Icon(isPreviewing ? Icons.pause : Icons.play_arrow),
                  tooltip: isPreviewing ? '일시정지' : '미리듣기',
                  onPressed: onTogglePreview,
                ),
                IconButton(
                  icon: const Icon(Icons.skip_next),
                  tooltip: '다음 음으로 건너뛰기',
                  onPressed: onSkipNote,
                ),
                IconButton(
                  icon: const Icon(Icons.stop),
                  tooltip: '정지',
                  onPressed: onStop,
                ),
              ],
            ),
            Row(
              children: [
                const Text('빠르기', style: TextStyle(fontSize: 12)),
                Expanded(
                  child: Slider(
                    min: 40,
                    max: 200,
                    divisions: 32,
                    value: bpm.toDouble(),
                    label: '$bpm BPM',
                    onChanged: (v) => onBpmChanged(v.round()),
                  ),
                ),
                SizedBox(
                  width: 56,
                  child: Text('$bpm BPM', style: const TextStyle(fontSize: 12)),
                ),
              ],
            ),
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text(
                '🎼 반주 모드 (왼손 자동)',
                style: TextStyle(fontSize: 13),
              ),
              subtitle: accompanimentAvailable
                  ? const Text('미리듣기(▶) 중에 왼손 반주가 같이 들려요', style: TextStyle(fontSize: 11))
                  : const Text('이 곡은 왼손 반주가 없어요', style: TextStyle(fontSize: 11)),
              value: accompanimentMode && accompanimentAvailable,
              onChanged: accompanimentAvailable ? onAccompanimentChanged : null,
            ),
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('음이름 표시', style: TextStyle(fontSize: 13)),
              value: showNoteNames,
              onChanged: onShowNoteNamesChanged,
            ),
          ],
        ),
      ),
    );
  }
}
