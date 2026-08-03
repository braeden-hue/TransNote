import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../services/auto_player.dart';
import '../theme/glory_theme.dart';
import '../utils/orientation_lock.dart';
import '../widgets/notation_widget.dart';
import '../widgets/piano_widget.dart';
import '../widgets/play_mode_toggle.dart';
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
  // "이 악보를 새 플레이리스트로" 진입(플레이리스트에 추가 시트)에서 넘어올 때, 그 악보를
  // 미리 선택해둔 채로 열기 위한 값 -- 일반 진입(악보 모음집 화면)에서는 비워둠.
  final int? preselectedIndex;
  const CreatePlaylistScreen({super.key, required this.entries, required this.onCreate, this.preselectedIndex});

  @override
  State<CreatePlaylistScreen> createState() => _CreatePlaylistScreenState();
}

class _CreatePlaylistScreenState extends State<CreatePlaylistScreen> {
  final _nameCtrl = TextEditingController();
  late final Set<int> _selected = {if (widget.preselectedIndex != null) widget.preselectedIndex!};

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

/// 스캔/변환된 악보 하나를 기존 플레이리스트에 추가하거나(체크 표시로 포함 여부 표시),
/// 그 자리에서 새 플레이리스트를 만들 수 있는 바텀시트. Mume 킷의 "add_song_to_playlist"
/// 화면 컨셉 참고. 탭 하나당 즉시 반영 후 시트를 닫는다(멀티 액션 중 상태를 계속
/// 띄워두는 대신 단순한 단일 액션 + 스낵바 확인으로 유지 -- 시트는 별도 라우트라 부모의
/// setState로 자동 다시 그려지지 않기 때문).
void showAddToPlaylistSheet(
  BuildContext context, {
  required ScannedEntry entry,
  required List<ScannedEntry> allEntries,
  required List<Playlist> playlists,
  required void Function(Playlist playlist, ScannedEntry entry) onToggle,
  required ValueChanged<Playlist> onCreatePlaylist,
}) {
  showModalBottomSheet(
    context: context,
    backgroundColor: gloryBg,
    shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
    builder: (sheetContext) {
      return SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('플레이리스트에 추가', style: TextStyle(color: gloryInk, fontSize: 17, fontWeight: FontWeight.w800)),
              const SizedBox(height: 4),
              Text(entry.title, style: TextStyle(color: gloryMuted, fontSize: 12.5)),
              const SizedBox(height: 16),
              Material(
                color: Colors.transparent,
                child: InkWell(
                  borderRadius: BorderRadius.circular(14),
                  onTap: () {
                    Navigator.of(sheetContext).pop();
                    Navigator.of(context).push(MaterialPageRoute(
                      builder: (_) => CreatePlaylistScreen(
                        entries: allEntries,
                        onCreate: onCreatePlaylist,
                        preselectedIndex: allEntries.indexOf(entry),
                      ),
                    ));
                  },
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Row(
                      children: [
                        Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(shape: BoxShape.circle, gradient: gloryGradient),
                          child: const Icon(Icons.add, color: Colors.white, size: 20),
                        ),
                        const SizedBox(width: 14),
                        Text('새 플레이리스트로 만들기',
                            style: TextStyle(color: gloryInk, fontSize: 14.5, fontWeight: FontWeight.w700)),
                      ],
                    ),
                  ),
                ),
              ),
              if (playlists.isNotEmpty) ...[
                const Divider(height: 24),
                ConstrainedBox(
                  constraints: const BoxConstraints(maxHeight: 320),
                  child: ListView(
                    shrinkWrap: true,
                    children: playlists.map((p) {
                      final included = p.entries.contains(entry);
                      return Material(
                        color: Colors.transparent,
                        child: InkWell(
                          borderRadius: BorderRadius.circular(14),
                          onTap: () {
                            onToggle(p, entry);
                            Navigator.of(sheetContext).pop();
                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                              content: Text(included ? '${p.name}에서 제거했어요' : '${p.name}에 추가했어요'),
                              duration: const Duration(seconds: 2),
                            ));
                          },
                          child: Padding(
                            padding: const EdgeInsets.symmetric(vertical: 8),
                            child: Row(
                              children: [
                                Container(
                                  width: 40,
                                  height: 40,
                                  decoration: BoxDecoration(color: glorySurface, borderRadius: BorderRadius.circular(12)),
                                  child: Icon(Icons.queue_music_rounded, color: gloryAccent, size: 20),
                                ),
                                const SizedBox(width: 14),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(p.name,
                                          style: TextStyle(color: gloryInk, fontSize: 14.5, fontWeight: FontWeight.w700),
                                          maxLines: 1, overflow: TextOverflow.ellipsis),
                                      Text('악보 ${p.entries.length}개', style: TextStyle(color: gloryMuted, fontSize: 11.5)),
                                    ],
                                  ),
                                ),
                                Icon(
                                  included ? Icons.check_circle : Icons.circle_outlined,
                                  color: included ? gloryAccent : gloryMuted,
                                  size: 22,
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ],
            ],
          ),
        ),
      );
    },
  );
}

// mume_app(참고용 프로토타입)의 playlist_detail_screen.dart 디자인을 참고해 큰 커버
// 블록 + 셔플/재생 버튼 + 이름변경/삭제/곡 추가/곡 제거를 진짜 데이터에 연결해 이식.
class PlaylistDetailScreen extends StatefulWidget {
  final Playlist playlist;
  final List<ScannedEntry> allEntries;
  final VoidCallback onDeleted;
  const PlaylistDetailScreen({
    super.key,
    required this.playlist,
    required this.allEntries,
    required this.onDeleted,
  });

  @override
  State<PlaylistDetailScreen> createState() => _PlaylistDetailScreenState();
}

class _PlaylistDetailScreenState extends State<PlaylistDetailScreen> {
  Playlist get _playlist => widget.playlist;

  void _play({required bool shuffle}) {
    final entries = List<ScannedEntry>.from(_playlist.entries);
    if (shuffle) entries.shuffle();
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => PlaylistPlayerScreen(
        playlist: Playlist(name: _playlist.name, entries: entries, createdAt: _playlist.createdAt),
      ),
    ));
  }

  Future<void> _rename() async {
    final ctrl = TextEditingController(text: _playlist.name);
    final newName = await showDialog<String>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: glorySurface,
        title: Text('이름 변경', style: TextStyle(color: gloryInk)),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          style: TextStyle(color: gloryInk),
          decoration: const InputDecoration(border: OutlineInputBorder()),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(dialogContext), child: const Text('취소')),
          FilledButton(
            onPressed: () => Navigator.pop(dialogContext, ctrl.text.trim()),
            style: gloryFilledButtonStyle(),
            child: const Text('저장'),
          ),
        ],
      ),
    );
    if (newName == null || newName.isEmpty) return;
    setState(() => _playlist.name = newName);
  }

  Future<void> _confirmDelete() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: glorySurface,
        title: Text('플레이리스트 삭제', style: TextStyle(color: gloryInk)),
        content: Text('"${_playlist.name}"을(를) 삭제할까요? 스캔한 악보 자체는 지워지지 않아요.',
            style: TextStyle(color: gloryMuted)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(dialogContext, false), child: const Text('취소')),
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, true),
            child: const Text('삭제', style: TextStyle(color: Color(0xFFE05252))),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    if (!mounted) return;
    Navigator.of(context).pop();
    widget.onDeleted();
  }

  void _showMenu() {
    showModalBottomSheet(
      context: context,
      backgroundColor: gloryBg,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
      builder: (sheetContext) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: Icon(Icons.edit_outlined, color: gloryInk),
              title: Text('이름 변경', style: TextStyle(color: gloryInk)),
              onTap: () {
                Navigator.pop(sheetContext);
                _rename();
              },
            ),
            ListTile(
              leading: const Icon(Icons.delete_outline, color: Color(0xFFE05252)),
              title: const Text('플레이리스트 삭제', style: TextStyle(color: Color(0xFFE05252))),
              onTap: () {
                Navigator.pop(sheetContext);
                _confirmDelete();
              },
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _addEntries() async {
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => AddEntriesToPlaylistScreen(playlist: _playlist, allEntries: widget.allEntries),
    ));
    setState(() {});
  }

  void _removeEntry(ScannedEntry entry) {
    setState(() => _playlist.entries.remove(entry));
  }

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
                  const Spacer(),
                  FrostedCircleButton(icon: Icons.more_horiz, onTap: _showMenu),
                ],
              ),
            ),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(24, 8, 24, 20),
                children: [
                  Container(
                    height: 170,
                    decoration: BoxDecoration(gradient: gloryGradient, borderRadius: BorderRadius.circular(24)),
                    child: const Icon(Icons.queue_music_rounded, color: Colors.white, size: 56),
                  ),
                  const SizedBox(height: 18),
                  Text(_playlist.name,
                      textAlign: TextAlign.center,
                      style: TextStyle(color: gloryInk, fontSize: 20, fontWeight: FontWeight.w800)),
                  const SizedBox(height: 4),
                  Text('악보 ${_playlist.entries.length}개',
                      textAlign: TextAlign.center, style: TextStyle(color: gloryMuted, fontSize: 12.5)),
                  const SizedBox(height: 18),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: _playlist.entries.isEmpty ? null : () => _play(shuffle: true),
                          style: gloryFilledButtonStyle(),
                          icon: const Icon(Icons.shuffle, size: 18),
                          label: const Text('섞어서 재생'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _playlist.entries.isEmpty ? null : () => _play(shuffle: false),
                          style: gloryOutlinedButtonStyle(),
                          icon: const Icon(Icons.play_arrow, size: 20),
                          label: const Text('이어서 재생'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 22),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('곡 목록', style: TextStyle(color: gloryInk, fontSize: 14, fontWeight: FontWeight.w800)),
                      GestureDetector(
                        onTap: _addEntries,
                        child: Container(
                          width: 32,
                          height: 32,
                          decoration: BoxDecoration(gradient: gloryGradient, shape: BoxShape.circle),
                          child: const Icon(Icons.add, color: Colors.white, size: 18),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  if (_playlist.entries.isEmpty)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 24),
                      child: Text('아직 곡이 없어요. + 버튼으로 추가해보세요.',
                          textAlign: TextAlign.center, style: TextStyle(color: gloryMuted, fontSize: 12.5)),
                    )
                  else
                    for (final e in _playlist.entries)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Row(
                          children: [
                            Expanded(child: EntryTile(entry: e, onTap: () {})),
                            IconButton(
                              icon: Icon(Icons.close, color: gloryMuted, size: 18),
                              tooltip: '목록에서 제거',
                              onPressed: () => _removeEntry(e),
                            ),
                          ],
                        ),
                      ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 기존 플레이리스트에 스캔한 악보를 더 추가하는 화면 -- CreatePlaylistScreen과 비슷하지만
/// "새로 만들기"가 아니라 "이미 있는 플레이리스트에 얹기"라서 이미 담긴 곡은 목록에서
/// 제외하고, 다중 선택 후 곧바로 playlist.entries에 추가한다.
class AddEntriesToPlaylistScreen extends StatefulWidget {
  final Playlist playlist;
  final List<ScannedEntry> allEntries;
  const AddEntriesToPlaylistScreen({super.key, required this.playlist, required this.allEntries});

  @override
  State<AddEntriesToPlaylistScreen> createState() => _AddEntriesToPlaylistScreenState();
}

class _AddEntriesToPlaylistScreenState extends State<AddEntriesToPlaylistScreen> {
  final Set<ScannedEntry> _selected = {};

  late final List<ScannedEntry> _candidates =
      widget.allEntries.where((e) => !widget.playlist.entries.contains(e)).toList();

  void _submit() {
    widget.playlist.entries.addAll(_selected);
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final canSubmit = _selected.isNotEmpty;
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
                  Text('곡 추가', style: TextStyle(color: gloryInk, fontSize: 18, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
            Expanded(
              child: _candidates.isEmpty
                  ? Center(
                      child: Text('추가할 수 있는 악보가 없어요.\n먼저 새 악보를 스캔해보세요.',
                          textAlign: TextAlign.center, style: TextStyle(color: gloryMuted, fontSize: 13)),
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                      itemCount: _candidates.length,
                      itemBuilder: (context, i) {
                        final e = _candidates[i];
                        final selected = _selected.contains(e);
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Material(
                            color: glorySurface,
                            borderRadius: BorderRadius.circular(14),
                            child: InkWell(
                              borderRadius: BorderRadius.circular(14),
                              onTap: () => setState(() {
                                if (selected) {
                                  _selected.remove(e);
                                } else {
                                  _selected.add(e);
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
                  child: Text('추가하기 (${_selected.length})',
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
  bool _bassOnly = false;
  AutoPlayer? _player;
  final _trebleCtrl = ScrollController();
  final _bassCtrl = ScrollController();
  bool _syncingScroll = false;

  @override
  void initState() {
    super.initState();
    lockLandscape();
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
        if (ti != _hlIdx && !_bassOnly) _audio.playNote(_joinedTreble[ti].pitch);
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
    lockPortrait();
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
                    if (_joinedBass != null) ...[
                      PlayModeToggle(
                        bassOnly: _bassOnly,
                        enabled: !_playing,
                        onChanged: (v) => setState(() => _bassOnly = v),
                      ),
                      const SizedBox(height: 12),
                    ],
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
              child: PianoWidget(
                highlightNote: _pianoNote,
                onKeyTap: (note) {
                  _audio.unlock();
                  _audio.playNote(note);
                  // 왼손 반주만 모드로 재생 중일 때는 자동재생을 멈추지 않고 탭한 음만 더해
                  // 사용자가 반주 위에 직접 멜로디를 얹어 연주할 수 있게 한다.
                  if (_bassOnly && _playing) return;
                  _player?.stop();
                  setState(() {
                    _playing = false;
                    _pianoNote = note;
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
