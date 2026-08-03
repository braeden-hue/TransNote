import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import '../theme/theme_controller.dart';
import 'collection_screen.dart';
import 'entry_detail_screen.dart';
import 'home_tab.dart';
import 'playlist_screen.dart';
import 'playlists_tab.dart';
import 'settings_screen.dart';

/// Mume_Modified_UI_Kit_PNG의 하단 5탭(홈/즐겨찾기/촬영/플레이리스트/설정) 셸을 참고한
/// 앱 루트 -- 이번 작업 범위에서 즐겨찾기/검색/아티스트 등은 제외했으므로 4탭(홈/촬영/
/// 플레이리스트/설정)으로 축소. 스캔한 악보(entries)와 플레이리스트는 홈 탭("최근 스캔")과
/// 촬영 탭이 함께 봐야 해서 여기서 소유하고 콜백으로 내려준다.
class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _tab = 0;
  final List<ScannedEntry> _entries = [];
  final List<Playlist> _playlists = [];

  void _addEntry(ScannedEntry e) => setState(() => _entries.insert(0, e));
  void _addPlaylist(Playlist p) => setState(() => _playlists.insert(0, p));
  void _deletePlaylist(Playlist p) => setState(() => _playlists.remove(p));

  void _toggleInPlaylist(Playlist playlist, ScannedEntry entry) {
    setState(() {
      if (playlist.entries.contains(entry)) {
        playlist.entries.remove(entry);
      } else {
        playlist.entries.add(entry);
      }
    });
  }

  void _openEntry(ScannedEntry e) {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => EntryDetailScreen(
        entry: e,
        allEntries: _entries,
        playlists: _playlists,
        onToggleInPlaylist: _toggleInPlaylist,
        onCreatePlaylist: _addPlaylist,
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    // gloryBg 등은 ThemeController 값을 직접 읽는 plain getter라 InheritedWidget처럼
    // 자동으로 하위 트리를 다시 그리게 하지 않는다 -- Navigator로 이미 push된 화면들은
    // MaterialApp이 다시 빌드돼도 자동으로 rebuild되지 않으므로(콜백 프로퍼티가 그대로면
    // Element가 재사용됨), 탭 4개를 실제로 소유한 이 지점에서 직접 구독해 다크모드 토글이
    // 하단 탭 전체에 반영되게 한다.
    return ValueListenableBuilder<bool>(
      valueListenable: ThemeController.instance.isDark,
      builder: (context, isDark, _) {
        final tabs = [
          HomeTab(entries: _entries, onOpenEntry: _openEntry),
          CollectionScreen(entries: _entries, onAddEntry: _addEntry, onOpenEntry: _openEntry),
          PlaylistsTab(
            entries: _entries,
            playlists: _playlists,
            onAddPlaylist: _addPlaylist,
            onDeletePlaylist: _deletePlaylist,
          ),
          SettingsScreen(),
        ];
        return Scaffold(
          backgroundColor: gloryBg,
          body: IndexedStack(index: _tab, children: tabs),
          bottomNavigationBar: _MumeBottomNav(current: _tab, onTap: (i) => setState(() => _tab = i)),
        );
      },
    );
  }
}

class _NavItem {
  final IconData icon;
  final String label;
  const _NavItem(this.icon, this.label);
}

const _navItems = [
  _NavItem(Icons.home_rounded, '홈'),
  _NavItem(Icons.camera_alt_rounded, '촬영'),
  _NavItem(Icons.queue_music_rounded, '플레이리스트'),
  _NavItem(Icons.settings_rounded, '설정'),
];

class _MumeBottomNav extends StatelessWidget {
  final int current;
  final ValueChanged<int> onTap;
  const _MumeBottomNav({required this.current, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      minimum: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Container(
          height: 68,
          padding: const EdgeInsets.symmetric(horizontal: 8),
          decoration: BoxDecoration(
            color: gloryNavBg,
            borderRadius: BorderRadius.circular(28),
            boxShadow: [BoxShadow(color: gloryInk.withValues(alpha: .1), blurRadius: 24, offset: const Offset(0, 10))],
          ),
          child: Row(
            children: List.generate(_navItems.length, (i) {
              final item = _navItems[i];
              final selected = i == current;
              final isCamera = item.icon == Icons.camera_alt_rounded;
              return Expanded(
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTap: () => onTap(i),
                  child: isCamera
                      ? Center(
                          child: Container(
                            width: 46,
                            height: 46,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              gradient: gloryGradient,
                              boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .45), blurRadius: 14, offset: const Offset(0, 5))],
                            ),
                            child: Icon(item.icon, color: Colors.white, size: 22),
                          ),
                        )
                      : Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(item.icon, color: selected ? gloryAccent : gloryMuted, size: 22),
                            const SizedBox(height: 3),
                            Text(item.label,
                                style: TextStyle(
                                  color: selected ? gloryAccent : gloryMuted,
                                  fontSize: 10,
                                  fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                                )),
                          ],
                        ),
                ),
              );
            }),
          ),
        ),
      ),
    );
  }
}
