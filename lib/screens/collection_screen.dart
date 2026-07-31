import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../services/auto_player.dart';
import '../theme/glory_theme.dart';
import '../widgets/notation_widget.dart';
import '../widgets/original_photo_viewer.dart';
import '../widgets/piano_widget.dart';
import 'guided_camera_screen.dart';
import 'playlist_screen.dart';

final _audio = AudioService.instance;

// Mume UI 킷 홈 화면의 "섹션 제목 + 전체보기" 패턴 참고 -- 최근 스캔 목록이 길어지면
// 미리보기는 일부만 보여주고 나머지는 전체보기에서 확인.
const _recentPreviewCount = 5;

class ScannedEntry {
  final String title;
  final DateTime time;
  final SampleScore preview;
  final Uint8List? photo;
  ScannedEntry(this.title, this.time, this.preview, [this.photo]);
}

class CollectionScreen extends StatefulWidget {
  const CollectionScreen({super.key});

  @override
  State<CollectionScreen> createState() => _CollectionScreenState();
}

class _CollectionScreenState extends State<CollectionScreen> with SingleTickerProviderStateMixin {
  final List<ScannedEntry> _entries = [];
  final List<Playlist> _playlists = [];
  bool _capturing = false;
  Uint8List? _photo;
  String? _error;
  late final AnimationController _scanCtrl;

  @override
  void initState() {
    super.initState();
    _scanCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 2200));
  }

  @override
  void dispose() {
    _scanCtrl.dispose();
    super.dispose();
  }

  Future<void> _captureFromGallery() async {
    setState(() => _error = null);
    try {
      final file = await ImagePicker().pickImage(source: ImageSource.gallery, imageQuality: 85);
      if (file == null) return;
      final bytes = await file.readAsBytes();
      await _processPhoto(bytes);
    } catch (e) {
      setState(() => _error = '갤러리를 사용할 수 없어요: $e');
    }
  }

  // 위/아래에 다른 오선이 걸쳐 있는 페이지에서 원하는 대보표 한 세트만 찍기 어렵다는
  // 문제 대응 -- 자동 오선 검출에 맡기지 않고, 촬영 화면의 가이드 박스에 맞춰 프레이밍한
  // 뒤 그 영역만 잘라서 받는다(GuidedCameraScreen).
  Future<void> _captureFromCamera() async {
    setState(() => _error = null);
    final bytes = await Navigator.of(context).push<Uint8List>(
      MaterialPageRoute(builder: (_) => const GuidedCameraScreen()),
    );
    if (bytes == null) return; // 사용자가 취소
    await _processPhoto(bytes);
  }

  Future<void> _processPhoto(Uint8List bytes) async {
    setState(() {
      _photo = bytes;
      _capturing = true;
    });
    await _scanCtrl.forward(from: 0);
    if (!mounted) return;
    final sample = samples[_entries.length % samples.length];
    setState(() {
      _entries.insert(0, ScannedEntry(sample.title, DateTime.now(), sample, _photo));
      _capturing = false;
      _photo = null;
    });
  }

  void _openEntry(ScannedEntry e) {
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => _EntryDetailScreen(entry: e)));
  }

  void _createPlaylist() {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => CreatePlaylistScreen(
        entries: _entries,
        onCreate: (playlist) => setState(() => _playlists.insert(0, playlist)),
      ),
    ));
  }

  void _openPlaylist(Playlist p) {
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => PlaylistDetailScreen(playlist: p)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: Stack(
        children: [
          SafeArea(child: _buildBody(context)),
          if (!_capturing)
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: SafeArea(top: false, child: _buildActionRow()),
            ),
          if (_capturing) _buildScanOverlay(),
        ],
      ),
    );
  }

  Widget _buildActionRow() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20, top: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          _ImportBadge(
            icon: Icons.photo_library_outlined,
            color: const Color(0xFF3EA06B),
            label: '갤러리',
            onTap: _captureFromGallery,
          ),
          const SizedBox(width: 40),
          GestureDetector(
            onTap: _captureFromCamera,
            child: Container(
              width: 68,
              height: 68,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: gloryAccent,
                boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .5), blurRadius: 20, spreadRadius: 2)],
              ),
              child: const Icon(Icons.camera_alt, color: Colors.white, size: 28),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBody(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 20, 0),
          child: Row(
            children: [
              FrostedCircleButton(icon: Icons.arrow_back, onTap: () => Navigator.of(context).pop()),
              const SizedBox(width: 12),
              Text('악보 모음집',
                  style: TextStyle(color: gloryInk, fontSize: 18, fontWeight: FontWeight.bold)),
            ],
          ),
        ),
        Expanded(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 160),
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: glorySurface,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('악보를 촬영해서\n바로 정리해보세요',
                              style: TextStyle(color: gloryInk, fontSize: 18, fontWeight: FontWeight.bold, height: 1.3)),
                          const SizedBox(height: 6),
                          Text('촬영한 사진은 커스텀 악보로 자동 변환돼요.',
                              style: TextStyle(color: gloryMuted, fontSize: 12)),
                        ],
                      ),
                    ),
                    Icon(Icons.description_outlined, color: gloryAccent.withValues(alpha: .7), size: 44),
                  ],
                ),
              ),
              if (_error != null) ...[
                const SizedBox(height: 12),
                Text(_error!, style: const TextStyle(color: Color(0xFFEE7777), fontSize: 12)),
              ],
              const SizedBox(height: 24),
              Text('내 플레이리스트',
                  style: TextStyle(color: gloryInk, fontSize: 14, fontWeight: FontWeight.bold)),
              const SizedBox(height: 12),
              SizedBox(
                height: 108,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  children: [
                    _NewPlaylistCard(onTap: _entries.isEmpty ? null : _createPlaylist),
                    for (final p in _playlists)
                      _PlaylistCard(playlist: p, onTap: () => _openPlaylist(p)),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              Row(
                children: [
                  Text('최근 스캔',
                      style: TextStyle(color: gloryInk, fontSize: 14, fontWeight: FontWeight.bold)),
                  const Spacer(),
                  if (_entries.length > _recentPreviewCount)
                    GestureDetector(
                      onTap: () => Navigator.of(context).push(
                        MaterialPageRoute(builder: (_) => _AllEntriesScreen(entries: _entries, onOpen: _openEntry)),
                      ),
                      child: Text('전체보기',
                          style: TextStyle(color: gloryAccent, fontSize: 12.5, fontWeight: FontWeight.w600)),
                    ),
                ],
              ),
              const SizedBox(height: 12),
              if (_entries.isEmpty)
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(color: glorySurface, borderRadius: BorderRadius.circular(16)),
                  child: Column(
                    children: [
                      Icon(Icons.camera_alt_outlined, color: gloryMuted, size: 28),
                      const SizedBox(height: 10),
                      Text('아직 스캔한 악보가 없어요.\n아래 카메라 버튼으로 촬영해보세요.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: gloryMuted, fontSize: 12.5, height: 1.4)),
                    ],
                  ),
                )
              else
                ..._entries.take(_recentPreviewCount).map((e) => EntryTile(entry: e, onTap: () => _openEntry(e))),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildScanOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black,
        child: Column(
          children: [
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      if (_photo != null) Image.memory(_photo!, fit: BoxFit.cover),
                      Container(color: Colors.black.withValues(alpha: .35)),
                      const _CornerGuide(),
                      AnimatedBuilder(
                        animation: _scanCtrl,
                        builder: (_, _) => Align(
                          alignment: Alignment(0, -1 + 2 * _scanCtrl.value),
                          child: Container(
                            height: 3,
                            margin: const EdgeInsets.symmetric(horizontal: 20),
                            decoration: BoxDecoration(
                              color: gloryAccent,
                              boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .8), blurRadius: 12, spreadRadius: 1)],
                            ),
                          ),
                        ),
                      ),
                      Positioned(
                        top: 12,
                        left: 12,
                        right: 12,
                        child: Row(
                          children: [
                            FrostedCircleButton(
                              icon: Icons.close,
                              alwaysLight: true,
                              onTap: () {
                                _scanCtrl.stop();
                                setState(() {
                                  _capturing = false;
                                  _photo = null;
                                });
                              },
                            ),
                            Expanded(
                              child: Center(
                                child: AnimatedBuilder(
                                  animation: _scanCtrl,
                                  builder: (_, _) => Container(
                                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                                    decoration: BoxDecoration(
                                      color: Colors.white.withValues(alpha: .12),
                                      borderRadius: BorderRadius.circular(20),
                                    ),
                                    child: Text('스캔 중... ${(_scanCtrl.value * 100).round()}%',
                                        style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600)),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 40),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }
}

class _CornerGuide extends StatelessWidget {
  const _CornerGuide();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.all(28),
      child: CustomPaint(painter: _CornerPainter(), child: SizedBox.expand()),
    );
  }
}

class _CornerPainter extends CustomPainter {
  const _CornerPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = Colors.white
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    const len = 26.0;
    // top-left
    canvas.drawLine(Offset.zero, const Offset(len, 0), p);
    canvas.drawLine(Offset.zero, const Offset(0, len), p);
    // top-right
    canvas.drawLine(Offset(size.width, 0), Offset(size.width - len, 0), p);
    canvas.drawLine(Offset(size.width, 0), Offset(size.width, len), p);
    // bottom-left
    canvas.drawLine(Offset(0, size.height), Offset(len, size.height), p);
    canvas.drawLine(Offset(0, size.height), Offset(0, size.height - len), p);
    // bottom-right
    canvas.drawLine(Offset(size.width, size.height), Offset(size.width - len, size.height), p);
    canvas.drawLine(Offset(size.width, size.height), Offset(size.width, size.height - len), p);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class EntryTile extends StatelessWidget {
  final ScannedEntry entry;
  final VoidCallback onTap;
  const EntryTile({super.key, required this.entry, required this.onTap});

  String _fmtTime(DateTime t) {
    final h = t.hour.toString().padLeft(2, '0');
    final m = t.minute.toString().padLeft(2, '0');
    return '오늘 · $h:$m';
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Material(
        color: glorySurface,
        borderRadius: BorderRadius.circular(14),
        child: InkWell(
          borderRadius: BorderRadius.circular(14),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: gloryAccent.withValues(alpha: .18),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(Icons.music_note, color: gloryAccent, size: 20),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('${entry.title} (스캔)',
                          style: TextStyle(color: gloryInk, fontSize: 14, fontWeight: FontWeight.w600),
                          maxLines: 1, overflow: TextOverflow.ellipsis),
                      const SizedBox(height: 2),
                      Text(_fmtTime(entry.time), style: TextStyle(color: gloryMuted, fontSize: 11.5)),
                    ],
                  ),
                ),
                Container(
                  width: 28,
                  height: 28,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: gloryInk.withValues(alpha: .4)),
                  ),
                  child: Icon(Icons.arrow_forward, color: gloryMuted, size: 14),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _ImportBadge extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String label;
  final VoidCallback onTap;
  const _ImportBadge({required this.icon, required this.color, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: color.withValues(alpha: .18),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(height: 6),
          Text(label, style: TextStyle(color: gloryMuted, fontSize: 11)),
        ],
      ),
    );
  }
}

// Mume UI 킷 홈 화면의 "Artists"/"Most Played" 가로 스크롤 둥근 카드 패턴 참고.
class _PlaylistCard extends StatelessWidget {
  final Playlist playlist;
  final VoidCallback onTap;
  const _PlaylistCard({required this.playlist, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 12),
      child: Material(
        color: glorySurface,
        borderRadius: BorderRadius.circular(16),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onTap,
          child: Container(
            width: 130,
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: gloryAccent.withValues(alpha: .2),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(Icons.queue_music, color: gloryAccent, size: 20),
                ),
                const SizedBox(height: 10),
                Text(playlist.name,
                    style: TextStyle(color: gloryInk, fontSize: 13, fontWeight: FontWeight.w600),
                    maxLines: 1, overflow: TextOverflow.ellipsis),
                const SizedBox(height: 2),
                Text('악보 ${playlist.entries.length}개', style: TextStyle(color: gloryMuted, fontSize: 11)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _NewPlaylistCard extends StatelessWidget {
  final VoidCallback? onTap;
  const _NewPlaylistCard({required this.onTap});

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null;
    return Padding(
      padding: const EdgeInsets.only(right: 12),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onTap,
          child: Container(
            width: 130,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: gloryInk.withValues(alpha: enabled ? .4 : .2), width: 1.4),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.add, color: enabled ? gloryInk : gloryInk.withValues(alpha: .4), size: 24),
                const SizedBox(height: 10),
                Text('새 플레이리스트',
                    maxLines: 1, overflow: TextOverflow.ellipsis,
                    style: TextStyle(color: enabled ? gloryInk : gloryInk.withValues(alpha: .4), fontSize: 13, fontWeight: FontWeight.w600)),
                const SizedBox(height: 2),
                Text(enabled ? '스캔한 악보 모아 만들기' : '스캔한 악보가 필요해요',
                    maxLines: 1, overflow: TextOverflow.ellipsis,
                    style: TextStyle(color: gloryInk.withValues(alpha: enabled ? .7 : .35), fontSize: 10.5)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class FrostedCircleButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  // 촬영 스캔 오버레이(항상 검은 배경, 카메라 뷰파인더 느낌)에서만 true -- 앱 테마와
  // 무관하게 흰색 프로스티드 유지. 그 외(일반 화면 헤더)는 기본값(false)으로 테마色 따라감.
  final bool alwaysLight;
  const FrostedCircleButton({super.key, required this.icon, required this.onTap, this.alwaysLight = false});

  @override
  Widget build(BuildContext context) {
    final fg = alwaysLight ? Colors.white : gloryInk;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: fg.withValues(alpha: .12),
        ),
        child: Icon(icon, color: fg, size: 20),
      ),
    );
  }
}

class _AllEntriesScreen extends StatelessWidget {
  final List<ScannedEntry> entries;
  final ValueChanged<ScannedEntry> onOpen;
  const _AllEntriesScreen({required this.entries, required this.onOpen});

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
                  Text('스캔한 악보 전체',
                      style: TextStyle(color: gloryInk, fontSize: 18, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
                children: entries.map((e) => EntryTile(entry: e, onTap: () => onOpen(e))).toList(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _EntryDetailScreen extends StatefulWidget {
  final ScannedEntry entry;
  const _EntryDetailScreen({required this.entry});

  @override
  State<_EntryDetailScreen> createState() => _EntryDetailScreenState();
}

class _EntryDetailScreenState extends State<_EntryDetailScreen> {
  int _trebleHlIdx = -1;
  int _bassHlIdx = -1;
  String? _pianoNote;

  final _trebleCtrl = ScrollController();
  final _bassCtrl = ScrollController();
  bool _syncingScroll = false;

  AutoPlayer? _player;
  bool _playing = false;

  @override
  void initState() {
    super.initState();
    _trebleCtrl.addListener(_syncFromTreble);
    _bassCtrl.addListener(_syncFromBass);
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
        if (ti != _trebleHlIdx) _audio.playNote(s.notes[ti].pitch);
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

  void _onTrebleTap(int idx, ScoreNote note) {
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
              tooltip: '원본 이미지로 보기',
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
                      child: const Text('인식 완료 (데모)',
                          style: TextStyle(color: Color(0xFF3E8F45), fontSize: 12, fontWeight: FontWeight.bold)),
                    ),
                    const SizedBox(height: 12),
                    Text('모델 학습이 아직 진행 중이라, 지금은 촬영 사진 대신 샘플 악보로 결과를 보여드려요.',
                        style: TextStyle(color: gloryInk.withValues(alpha: .45), fontSize: 11.5, height: 1.4)),
                    const SizedBox(height: 16),
                    Text('음표를 눌러보세요',
                        style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12)),
                    const SizedBox(height: 8),
                    NotationWidget(
                      notes: s.notes,
                      clef: 'treble',
                      highlightIdx: _trebleHlIdx,
                      timeSignature: s.timeSignature,
                      scrollController: _trebleCtrl,
                      onNoteTap: _onTrebleTap,
                    ),
                    if (isGrand) ...[
                      const SizedBox(height: 8),
                      NotationWidget(
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
                onKeyTap: (note) {
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
