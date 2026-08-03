import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import 'collection_screen.dart';
import 'playlist_screen.dart';

enum _PlaylistSort { recentlyAdded, name }

/// 하단 탭 "플레이리스트" -- Mume_Modified_UI_Kit_PNG 25번(playlists) 참고, 정렬 토글은
/// mume_app(참고용 프로토타입)의 playlists_screen.dart 디자인 패턴을 실제 데이터(createdAt)에
/// 연결해 이식.
class PlaylistsTab extends StatefulWidget {
  final List<ScannedEntry> entries;
  final List<Playlist> playlists;
  final ValueChanged<Playlist> onAddPlaylist;
  final ValueChanged<Playlist> onDeletePlaylist;
  const PlaylistsTab({
    super.key,
    required this.entries,
    required this.playlists,
    required this.onAddPlaylist,
    required this.onDeletePlaylist,
  });

  @override
  State<PlaylistsTab> createState() => _PlaylistsTabState();
}

class _PlaylistsTabState extends State<PlaylistsTab> {
  _PlaylistSort _sort = _PlaylistSort.recentlyAdded;

  void _createPlaylist(BuildContext context) {
    if (widget.entries.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('먼저 악보를 스캔해야 플레이리스트를 만들 수 있어요.')),
      );
      return;
    }
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => CreatePlaylistScreen(entries: widget.entries, onCreate: widget.onAddPlaylist),
    ));
  }

  void _openPlaylist(BuildContext context, Playlist p) {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => PlaylistDetailScreen(
        playlist: p,
        allEntries: widget.entries,
        onDeleted: () => widget.onDeletePlaylist(p),
      ),
    ));
  }

  List<Playlist> get _sortedPlaylists {
    final list = List<Playlist>.from(widget.playlists);
    switch (_sort) {
      case _PlaylistSort.recentlyAdded:
        list.sort((a, b) => b.createdAt.compareTo(a.createdAt));
      case _PlaylistSort.name:
        list.sort((a, b) => a.name.compareTo(b.name));
    }
    return list;
  }

  void _toggleSort() {
    setState(() {
      _sort = _sort == _PlaylistSort.recentlyAdded ? _PlaylistSort.name : _PlaylistSort.recentlyAdded;
    });
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
        children: [
          Text('플레이리스트', style: TextStyle(color: gloryInk, fontSize: 24, fontWeight: FontWeight.w800)),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('${widget.playlists.length}개의 플레이리스트', style: TextStyle(color: gloryMuted, fontSize: 13)),
              if (widget.playlists.isNotEmpty)
                GestureDetector(
                  onTap: _toggleSort,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(_sort == _PlaylistSort.recentlyAdded ? '최근 추가순' : '이름순',
                          style: TextStyle(color: gloryAccent, fontSize: 12.5, fontWeight: FontWeight.w700)),
                      Icon(Icons.swap_vert, color: gloryAccent, size: 18),
                    ],
                  ),
                ),
            ],
          ),
          const SizedBox(height: 16),
          Material(
            color: Colors.transparent,
            child: InkWell(
              borderRadius: BorderRadius.circular(16),
              onTap: () => _createPlaylist(context),
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Row(
                  children: [
                    Container(
                      width: 48,
                      height: 48,
                      decoration: BoxDecoration(shape: BoxShape.circle, gradient: gloryGradient),
                      child: const Icon(Icons.add, color: Colors.white, size: 24),
                    ),
                    const SizedBox(width: 14),
                    Text('새 플레이리스트 만들기',
                        style: TextStyle(color: gloryInk, fontSize: 15, fontWeight: FontWeight.w700)),
                  ],
                ),
              ),
            ),
          ),
          const Divider(height: 28),
          if (widget.playlists.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 24),
              child: Column(
                children: [
                  Icon(Icons.queue_music_rounded, color: gloryMuted, size: 28),
                  const SizedBox(height: 10),
                  Text('아직 만든 플레이리스트가 없어요.',
                      style: TextStyle(color: gloryMuted, fontSize: 13)),
                ],
              ),
            )
          else
            ..._sortedPlaylists.map((p) => _PlaylistRow(
                  playlist: p,
                  onTap: () => _openPlaylist(context, p),
                )),
        ],
      ),
    );
  }
}

class _PlaylistRow extends StatelessWidget {
  final Playlist playlist;
  final VoidCallback onTap;
  const _PlaylistRow({required this.playlist, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 10),
          child: Row(
            children: [
              Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(color: glorySurface, borderRadius: BorderRadius.circular(14)),
                child: Icon(Icons.queue_music_rounded, color: gloryAccent, size: 24),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(playlist.name,
                        style: TextStyle(color: gloryInk, fontSize: 15, fontWeight: FontWeight.w700),
                        maxLines: 1, overflow: TextOverflow.ellipsis),
                    const SizedBox(height: 2),
                    Text('악보 ${playlist.entries.length}개', style: TextStyle(color: gloryMuted, fontSize: 12)),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: gloryMuted),
            ],
          ),
        ),
      ),
    );
  }
}
