import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import 'glory_music_data.dart';
import 'now_playing_screen.dart';
import 'widgets/glory_bottom_nav.dart';

class GloryLibraryScreen extends StatefulWidget {
  const GloryLibraryScreen({super.key});

  @override
  State<GloryLibraryScreen> createState() => _GloryLibraryScreenState();
}

class _GloryLibraryScreenState extends State<GloryLibraryScreen> {
  final _pageController = PageController(viewportFraction: 0.8);
  int _albumIndex = 0;

  @override
  void initState() {
    super.initState();
    _pageController.addListener(() {
      final next = _pageController.page?.round() ?? 0;
      if (next != _albumIndex) setState(() => _albumIndex = next);
    });
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final album = galleryAlbums[_albumIndex];
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(24, 8, 20, 0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Glory Music',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600, color: gloryInk),
                  ),
                  Icon(Icons.more_horiz, color: gloryInk),
                ],
              ),
            ),
            SizedBox(
              height: 150,
              child: PageView.builder(
                controller: _pageController,
                itemCount: galleryAlbums.length,
                itemBuilder: (context, i) => _AlbumCard(album: galleryAlbums[i], focused: i == _albumIndex),
              ),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ListView.separated(
                padding: const EdgeInsets.fromLTRB(24, 8, 24, 16),
                itemCount: album.tracks.length,
                separatorBuilder: (_, _) => const SizedBox(height: 6),
                itemBuilder: (context, i) {
                  final track = album.tracks[i];
                  final isCurrent = i == album.currentTrackIndex;
                  return _TrackTile(
                    album: album,
                    track: track,
                    isCurrent: isCurrent,
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute(builder: (_) => GloryNowPlayingScreen(album: album, trackIndex: i)),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: const GloryBottomNav(),
    );
  }
}

class _AlbumCard extends StatelessWidget {
  final GloryAlbum album;
  final bool focused;

  const _AlbumCard({required this.album, required this.focused});

  @override
  Widget build(BuildContext context) {
    return AnimatedOpacity(
      opacity: focused ? 1 : 0.35,
      duration: const Duration(milliseconds: 200),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 116,
              height: 116,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(colors: album.gradient, begin: Alignment.topLeft, end: Alignment.bottomRight),
              ),
              child: const Icon(Icons.person, color: Colors.white24, size: 52),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('YEAR', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: gloryInk.withValues(alpha: .5), letterSpacing: 1)),
                  const SizedBox(height: 2),
                  Text('${album.year} ALBUM', style: TextStyle(fontSize: 15, color: gloryInk.withValues(alpha: .5))),
                  const SizedBox(height: 8),
                  Text(album.title, style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600, color: gloryInk), maxLines: 3, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 4),
                  Text(album.artist, style: TextStyle(fontSize: 13, color: gloryInk.withValues(alpha: .4))),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _TrackTile extends StatelessWidget {
  final GloryAlbum album;
  final GloryTrack track;
  final bool isCurrent;
  final VoidCallback onTap;

  const _TrackTile({required this.album, required this.track, required this.isCurrent, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: isCurrent ? Colors.white : Colors.transparent,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
          child: Row(
            children: [
              Container(
                width: 43,
                height: 43,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  gradient: LinearGradient(colors: album.gradient, begin: Alignment.topLeft, end: Alignment.bottomRight),
                ),
                child: Icon(isCurrent ? Icons.pause : Icons.play_arrow, color: Colors.white, size: 20),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Text(
                  track.title,
                  style: const TextStyle(fontSize: 15, color: gloryInk),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (isCurrent) const Icon(Icons.graphic_eq, size: 18, color: gloryInk),
            ],
          ),
        ),
      ),
    );
  }
}
