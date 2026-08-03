import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../services/auto_player.dart';
import '../theme/glory_theme.dart';
import '../utils/orientation_lock.dart';
import '../widgets/notation_widget.dart';
import '../widgets/original_photo_viewer.dart';
import '../widgets/piano_widget.dart';
import '../widgets/play_mode_toggle.dart';
import '../widgets/staff_notation_widget.dart';
import 'collection_screen.dart' show ScannedEntry;
import 'playlist_screen.dart';

final _audio = AudioService.instance;

/// 스캔된 악보 한 건의 재생/열람 화면. HomeTab(최근 스캔)과 CollectionScreen(Score Lab
/// 탭)/PlaylistsTab 양쪽에서 공용으로 열기 때문에 최상위 공개 클래스로 분리해뒀다.
/// allEntries/playlists는 AppShell이 들고 있는 리스트를 그대로(참조로) 받는다 -- 이 화면
/// 자체는 다시 그려지지 않아도, "플레이리스트에 추가" 시트를 열 때마다 그 시점의 최신
/// 상태를 읽을 수 있어야 하기 때문(리스트 mutate는 참조가 바뀌지 않으므로 그대로 반영됨).
class EntryDetailScreen extends StatefulWidget {
  final ScannedEntry entry;
  final List<ScannedEntry> allEntries;
  final List<Playlist> playlists;
  final void Function(Playlist playlist, ScannedEntry entry) onToggleInPlaylist;
  final ValueChanged<Playlist> onCreatePlaylist;
  const EntryDetailScreen({
    super.key,
    required this.entry,
    required this.allEntries,
    required this.playlists,
    required this.onToggleInPlaylist,
    required this.onCreatePlaylist,
  });

  @override
  State<EntryDetailScreen> createState() => _EntryDetailScreenState();
}

class _EntryDetailScreenState extends State<EntryDetailScreen> {
  int _trebleHlIdx = -1;
  int _bassHlIdx = -1;
  String? _pianoNote;

  final _trebleCtrl = ScrollController();
  final _bassCtrl = ScrollController();
  bool _syncingScroll = false;

  AutoPlayer? _player;
  bool _playing = false;

  bool _bassOnly = false;
  bool _customMode = true;

  @override
  void initState() {
    super.initState();
    _trebleCtrl.addListener(_syncFromTreble);
    _bassCtrl.addListener(_syncFromBass);
    lockLandscape();
  }

  // 대보표 두 줄이 같은 타임라인을 공유하므로, 한쪽을 스와이프하면 다른 쪽도 같은
  // x축 위치로 따라가야 두 줄이 어긋나 보이지 않는다(ScoreScreen의 트레블/베이스 동기화와 동일 패턴).
  void _syncFromTreble() {
    if (_syncingScroll || !_bassCtrl.hasClients) return;
    _syncingScroll = true;
    _bassCtrl.jumpTo(_trebleCtrl.offset.clamp(0.0, _bassCtrl.position.maxScrollExtent));
    _syncingScroll = false;
  }

  void _syncFromBass() {
    if (_syncingScroll || !_trebleCtrl.hasClients) return;
    _syncingScroll = true;
    _trebleCtrl.jumpTo(_bassCtrl.offset.clamp(0.0, _trebleCtrl.position.maxScrollExtent));
    _syncingScroll = false;
  }

  @override
  void dispose() {
    _player?.dispose();
    _trebleCtrl.dispose();
    _bassCtrl.dispose();
    lockPortrait();
    super.dispose();
  }

  void _togglePlay() {
    if (_playing) {
      _player?.stop();
      setState(() => _playing = false);
      return;
    }
    final s = widget.entry.preview;
    _player = AutoPlayer(
      treble: s.notes,
      bass: s.bassNotes,
      tempo: s.tempo,
      onTick: (ti, bi) {
        if (!mounted) return;
        if (ti != _trebleHlIdx && !_bassOnly) _audio.playNote(s.notes[ti].pitch);
        if (bi >= 0 && bi != _bassHlIdx && s.bassNotes != null) _audio.playNote(s.bassNotes![bi].pitch);
        setState(() {
          _trebleHlIdx = ti;
          _bassHlIdx = bi;
          _pianoNote = s.notes[ti].pitch;
        });
      },
      onDone: () {
        if (!mounted) return;
        setState(() => _playing = false);
      },
    );
    _audio.unlock();
    setState(() => _playing = true);
    _player!.start();
  }

  // 왼손 반주만 모드로 재생 중일 때는 사용자가 직접 멜로디를 같이 연주할 수 있도록
  // 자동재생을 멈추지 않는다(반주는 계속 흐르게 두고, 탭한 음만 추가로 재생).
  void _onTrebleTap(int idx, ScoreNote note) {
    if (_bassOnly && _playing) {
      _audio.unlock();
      _audio.playNote(note.pitch);
      return;
    }
    _player?.stop();
    _playing = false;
    _audio.unlock();
    _audio.playNote(note.pitch);
    setState(() {
      _trebleHlIdx = idx;
      _bassHlIdx = -1;
      _pianoNote = note.pitch;
    });
  }

  void _onBassTap(int idx, ScoreNote note) {
    if (_bassOnly && _playing) {
      _audio.unlock();
      _audio.playNote(note.pitch);
      return;
    }
    _player?.stop();
    _playing = false;
    _audio.unlock();
    _audio.playNote(note.pitch);
    setState(() {
      _bassHlIdx = idx;
      _trebleHlIdx = -1;
      _pianoNote = note.pitch;
    });
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.entry.preview;
    final bassNotes = s.bassNotes;
    final isGrand = bassNotes != null && bassNotes.isNotEmpty;
    return Scaffold(
      backgroundColor: gloryBg,
      appBar: AppBar(
        backgroundColor: gloryBg,
        elevation: 0,
        iconTheme: IconThemeData(color: gloryInk),
        title: Text('${s.emoji} ${widget.entry.title}', style: TextStyle(color: gloryInk, fontWeight: FontWeight.w600)),
        actions: [
          if (widget.entry.photo != null)
            IconButton(
              icon: Icon(Icons.image_outlined, color: gloryInk),
              tooltip: '원본 악보로 보기',
              onPressed: () {
                _player?.stop();
                setState(() => _playing = false);
                Navigator.of(context).push(MaterialPageRoute(
                  builder: (_) => OriginalPhotoScreen(
                    photo: widget.entry.photo!,
                    notes: s.notes,
                    title: widget.entry.title,
                  ),
                ));
              },
            ),
          IconButton(
            icon: Icon(Icons.playlist_add, color: gloryInk),
            tooltip: '플레이리스트에 추가',
            onPressed: () => showAddToPlaylistSheet(
              context,
              entry: widget.entry,
              allEntries: widget.allEntries,
              playlists: widget.playlists,
              onToggle: widget.onToggleInPlaylist,
              onCreatePlaylist: widget.onCreatePlaylist,
            ),
          ),
          IconButton(
            icon: Icon(_playing ? Icons.pause_circle_filled : Icons.play_circle_fill,
                color: gloryAccent, size: 30),
            tooltip: _playing ? '일시정지' : '연주하기',
            onPressed: _togglePlay,
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: const Color(0xFF7BC67E).withAlpha(30),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: const Color(0xFF7BC67E), width: 1),
                      ),
                      child: const Text('변환 악보 · 데모',
                          style: TextStyle(color: Color(0xFF3E8F45), fontSize: 12, fontWeight: FontWeight.bold)),
                    ),
                    const SizedBox(height: 12),
                    Text('학습 중인 OMR 모델이 준비되면, 촬영한 사진이 바로 아래와 같은 커스텀 악보로 변환돼요.',
                        style: TextStyle(color: gloryInk.withValues(alpha: .45), fontSize: 11.5, height: 1.4)),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 12,
                      runSpacing: 8,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        _CustomModeToggle(
                          customMode: _customMode,
                          onChanged: (v) => setState(() => _customMode = v),
                        ),
                        if (isGrand)
                          PlayModeToggle(
                            bassOnly: _bassOnly,
                            enabled: !_playing,
                            onChanged: (v) => setState(() => _bassOnly = v),
                          ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Text('음표를 눌러보세요',
                        style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12)),
                    const SizedBox(height: 8),
                    _customMode
                        ? NotationWidget(
                            notes: s.notes,
                            clef: 'treble',
                            highlightIdx: _trebleHlIdx,
                            timeSignature: s.timeSignature,
                            scrollController: _trebleCtrl,
                            onNoteTap: _onTrebleTap,
                          )
                        : StaffNotationWidget(
                            notes: s.notes,
                            clef: 'treble',
                            highlightIdx: _trebleHlIdx,
                            timeSignature: s.timeSignature,
                            scrollController: _trebleCtrl,
                            onNoteTap: _onTrebleTap,
                          ),
                    if (isGrand) ...[
                      const SizedBox(height: 8),
                      _customMode
                          ? NotationWidget(
                              notes: bassNotes,
                              clef: 'bass',
                              highlightIdx: _bassHlIdx,
                              scrollController: _bassCtrl,
                              onNoteTap: _onBassTap,
                            )
                          : StaffNotationWidget(
                              notes: bassNotes,
                              clef: 'bass',
                              highlightIdx: _bassHlIdx,
                              scrollController: _bassCtrl,
                              onNoteTap: _onBassTap,
                            ),
                    ],
                  ],
                ),
              ),
            ),
            Container(
              decoration: BoxDecoration(
                color: glorySurface,
                border: Border(top: BorderSide(color: gloryBorder, width: 1)),
              ),
              child: PianoWidget(
                highlightNote: _pianoNote,
                highlightNote2: (bassNotes != null && _bassHlIdx >= 0 && _bassHlIdx < bassNotes.length)
                    ? bassNotes[_bassHlIdx].pitch
                    : null,
                onKeyTap: (note) {
                  if (_bassOnly && _playing) {
                    _audio.unlock();
                    _audio.playNote(note);
                    return;
                  }
                  _player?.stop();
                  _playing = false;
                  _audio.unlock();
                  _audio.playNote(note);
                  setState(() {
                    _pianoNote = note;
                    _trebleHlIdx = -1;
                    _bassHlIdx = -1;
                  });
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// 커스텀 악보(색칠 셀 표기) ↔ 오선 악보(StaffNotationWidget) 전환 -- score_screen.dart의
// _ModeToggle(감상하기/연주하기)와 같은 스타일.
class _CustomModeToggle extends StatelessWidget {
  final bool customMode;
  final ValueChanged<bool> onChanged;
  const _CustomModeToggle({required this.customMode, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(color: gloryBg, borderRadius: BorderRadius.circular(12)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _segment('커스텀 악보', customMode, () => onChanged(true)),
          _segment('오선 악보', !customMode, () => onChanged(false)),
        ],
      ),
    );
  }

  Widget _segment(String label, bool selected, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? gloryInk : Colors.transparent,
          borderRadius: BorderRadius.circular(9),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: selected ? gloryBg : gloryInk.withValues(alpha: .5),
            fontSize: 12.5,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}
