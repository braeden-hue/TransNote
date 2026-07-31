import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../services/auto_player.dart';
import '../theme/glory_theme.dart';
import '../widgets/notation_widget.dart';
import '../widgets/piano_widget.dart';
import 'collection_screen.dart';

final _audio = AudioService.instance;

/// 여러 스캔 결과를 사용자가 이름 붙여 묶은 모음. 음악 플레이리스트처럼 "이어서 재생" 가능.
class Playlist {
  String name;
  final List<ScannedEntry> entries;
  final DateTime createdAt;
  Playlist({required this.name, required this.entries, required this.createdAt});
}

class CreatePlaylistScreen extends StatefulWidget {
  final List<ScannedEntry> entries;
  final ValueChanged<Playlist> onCreate;
  const CreatePlaylistScreen({super.key, required this.entries, required this.onCreate});

  @override
  State<CreatePlaylistScreen> createState() => _CreatePlaylistScreenState();
}

class _CreatePlaylistScreenState extends State<CreatePlaylistScreen> {
  final _nameCtrl = TextEditingController();
  final Set<int> _selected = {};

  @override
  void dispose() {
    _nameCtrl.dispose();
    super.dispose();
  }

  void _submit() {
    final name = _nameCtrl.text.trim();
    if (name.isEmpty || _selected.isEmpty) return;
    final chosen = _selected.map((i) => widget.entries[i]).toList();
    widget.onCreate(Playlist(name: name, entries: chosen, createdAt: DateTime.now()));
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final canSubmit = _nameCtrl.text.trim().isNotEmpty && _selected.isNotEmpty;
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 20, 0),
              child: Row(
                children: [
                  FrostedCircleButton(icon: Icons.arrow_back, onTap: () => Navigator.of(context).pop()),
                  const SizedBox(width: 12),
                  Text('새 플레이리스트',
                      style: TextStyle(color: gloryInk, fontSize: 18, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 12),
              child: TextField(
                controller: _nameCtrl,
                style: TextStyle(color: gloryInk),
                onChanged: (_) => setState(() {}),
                decoration: InputDecoration(
                  hintText: '플레이리스트 이름 (예: 연습곡 모음)',
                  hintStyle: TextStyle(color: gloryMuted),
                  filled: true,
                  fillColor: glorySurface,
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                children: [
                  Text('포함할 악보 선택', style: TextStyle(color: gloryMuted, fontSize: 12.5)),
                  const Spacer(),
                  Text('${_selected.length}개 선택됨', style: TextStyle(color: gloryAccent, fontSize: 12.5, fontWeight: FontWeight.w600)),
                ],
              ),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
                itemCount: widget.entries.length,
                itemBuilder: (context, i) {
                  final e = widget.entries[i];
                  final selected = _selected.contains(i);
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: Material(
                      color: glorySurface,
                      borderRadius: BorderRadius.circular(14),
                      child: InkWell(
                        borderRadius: BorderRadius.circular(14),
                        onTap: () => setState(() {
                          if (selected) {
                            _selected.remove(i);
                          } else {
                            _selected.add(i);
                          }
                        }),
                        child: Padding(
                          padding: const EdgeInsets.all(14),
                          child: Row(
                            children: [
                              Icon(selected ? Icons.check_circle : Icons.circle_outlined,
                                  color: selected ? gloryAccent : gloryMuted, size: 22),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text('${e.preview.emoji} ${e.title}',
                                    style: TextStyle(color: gloryInk, fontSize: 14, fontWeight: FontWeight.w600),
                                    maxLines: 1, overflow: TextOverflow.ellipsis),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
              child: SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: canSubmit ? _submit : null,
                  style: FilledButton.styleFrom(
                    backgroundColor: gloryAccent,
                    disabledBackgroundColor: glorySurface,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                  ),
                  child: Text('만들기',
                      style: TextStyle(color: canSubmit ? Colors.white : gloryMuted, fontWeight: FontWeight.bold)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class PlaylistDetailScreen extends StatelessWidget {
  final Playlist playlist;
  const PlaylistDetailScreen({super.key, required this.playlist});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 20, 0),
              child: Row(
                children: [
                  FrostedCircleButton(icon: Icons.arrow_back, onTap: () => Navigator.of(context).pop()),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(playlist.name,
                        style: TextStyle(color: gloryInk, fontSize: 18, fontWeight: FontWeight.bold),
                        maxLines: 1, overflow: TextOverflow.ellipsis),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            GestureDetector(
              onTap: () => Navigator.of(context).push(MaterialPageRoute(
                builder: (_) => PlaylistPlayerScreen(playlist: playlist),
              )),
              child: Container(
                width: 84,
                height: 84,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: gloryAccent,
                  boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .5), blurRadius: 20, spreadRadius: 2)],
                ),
                child: const Icon(Icons.play_arrow, color: Colors.white, size: 40),
              ),
            ),
            const SizedBox(height: 8),
            Text('이어서 재생', style: TextStyle(color: gloryMuted, fontSize: 12.5)),
            const SizedBox(height: 24),
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                itemCount: playlist.entries.length,
                itemBuilder: (context, i) => EntryTile(entry: playlist.entries[i], onTap: () {}),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 플레이리스트에 담긴 여러 악보를 하나로 이어붙여 자동 재생. 대보표는 모든 악보가
/// bassNotes를 갖고 있을 때만 두 줄로 재생하고(섞여 있으면 트레블만 이어붙임 -- 한쪽만
/// 대보표면 마디 정렬이 어긋나 보이는 걸 피하기 위한 단순화), 템포는 첫 곡 기준.
class PlaylistPlayerScreen extends StatefulWidget {
  final Playlist playlist;
  const PlaylistPlayerScreen({super.key, required this.playlist});

  @override
  State<PlaylistPlayerScreen> createState() => _PlaylistPlayerScreenState();
}

class _PlaylistPlayerScreenState extends State<PlaylistPlayerScreen> {
  late final List<ScoreNote> _joinedTreble;
  late final List<ScoreNote>? _joinedBass;
  late final List<int> _entryStartIdx; // 각 entry가 시작하는 _joinedTreble 인덱스
  int _hlIdx = -1;
  int _bassHlIdx = -1;
  String? _pianoNote;
  bool _playing = false;
  bool _done = false;
  AutoPlayer? _player;
  final _trebleCtrl = ScrollController();
  final _bassCtrl = ScrollController();
  bool _syncingScroll = false;

  @override
  void initState() {
    super.initState();
    final treble = <ScoreNote>[];
    final bass = <ScoreNote>[];
    final starts = <int>[];
    final allGrand = widget.playlist.entries.every((e) => e.preview.bassNotes != null && e.preview.bassNotes!.isNotEmpty);
    for (final e in widget.playlist.entries) {
      starts.add(treble.length);
      treble.addAll(e.preview.notes);
      if (allGrand) bass.addAll(e.preview.bassNotes!);
    }
    _joinedTreble = treble;
    _joinedBass = allGrand ? bass : null;
    _entryStartIdx = starts;
    _trebleCtrl.addListener(_syncFromTreble);
    _bassCtrl.addListener(_syncFromBass);
  }

  void _syncFromTreble() {
    if (_syncingScroll || _joinedBass == null || !_bassCtrl.hasClients) return;
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

  int _currentEntryIdx() {
    var idx = 0;
    for (var i = 0; i < _entryStartIdx.length; i++) {
      if (_hlIdx >= _entryStartIdx[i]) idx = i;
    }
    return idx;
  }

  void _togglePlay() {
    if (_playing) {
      _player?.stop();
      setState(() => _playing = false);
      return;
    }
    if (_done) {
      setState(() {
        _done = false;
        _hlIdx = -1;
        _bassHlIdx = -1;
      });
    }
    final tempo = widget.playlist.entries.first.preview.tempo;
    _player = AutoPlayer(
      treble: _joinedTreble,
      bass: _joinedBass,
      tempo: tempo,
      onTick: (ti, bi) {
        if (!mounted) return;
        if (ti != _hlIdx) _audio.playNote(_joinedTreble[ti].pitch);
        if (bi >= 0 && bi != _bassHlIdx && _joinedBass != null) _audio.playNote(_joinedBass[bi].pitch);
        setState(() {
          _hlIdx = ti;
          _bassHlIdx = bi;
          _pianoNote = _joinedTreble[ti].pitch;
        });
      },
      onDone: () {
        if (!mounted) return;
        setState(() {
          _playing = false;
          _done = true;
        });
      },
    );
    _audio.unlock();
    setState(() => _playing = true);
    _player!.start();
  }

  @override
  void dispose() {
    _player?.dispose();
    _trebleCtrl.dispose();
    _bassCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final entries = widget.playlist.entries;
    final curName = entries.isEmpty ? '' : entries[_currentEntryIdx()].title;
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 20, 0),
              child: Row(
                children: [
                  FrostedCircleButton(icon: Icons.arrow_back, onTap: () {
                    _player?.stop();
                    Navigator.of(context).pop();
                  }),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(widget.playlist.name,
                            style: TextStyle(color: gloryInk, fontSize: 16, fontWeight: FontWeight.bold),
                            maxLines: 1, overflow: TextOverflow.ellipsis),
                        if (_hlIdx >= 0)
                          Text('재생 중: $curName', style: TextStyle(color: gloryMuted, fontSize: 11.5)),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    NotationWidget(
                      notes: _joinedTreble,
                      clef: 'treble',
                      highlightIdx: _hlIdx,
                      scrollController: _trebleCtrl,
                    ),
                    if (_joinedBass != null) ...[
                      const SizedBox(height: 8),
                      NotationWidget(
                        notes: _joinedBass,
                        clef: 'bass',
                        highlightIdx: _bassHlIdx,
                        scrollController: _bassCtrl,
                      ),
                    ],
                  ],
                ),
              ),
            ),
            if (_done)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text('재생 완료', style: TextStyle(color: gloryMuted, fontSize: 12.5)),
              ),
            Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: GestureDetector(
                onTap: _togglePlay,
                child: Container(
                  width: 68,
                  height: 68,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: gloryAccent,
                    boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .5), blurRadius: 20, spreadRadius: 2)],
                  ),
                  child: Icon(_playing ? Icons.pause : Icons.play_arrow, color: Colors.white, size: 32),
                ),
              ),
            ),
            Container(
              decoration: BoxDecoration(color: glorySurface, border: Border(top: BorderSide(color: gloryInk.withValues(alpha: .15)))),
              child: PianoWidget(highlightNote: _pianoNote),
            ),
          ],
        ),
      ),
    );
  }
}
