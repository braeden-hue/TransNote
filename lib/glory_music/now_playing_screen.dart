import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import 'glory_music_data.dart';
import 'widgets/glory_bottom_nav.dart';

class GloryNowPlayingScreen extends StatefulWidget {
  final GloryAlbum album;
  final int trackIndex;

  const GloryNowPlayingScreen({super.key, required this.album, required this.trackIndex});

  @override
  State<GloryNowPlayingScreen> createState() => _GloryNowPlayingScreenState();
}

class _GloryNowPlayingScreenState extends State<GloryNowPlayingScreen> {
  late int _index = widget.trackIndex;
  bool _playing = true;
  bool _liked = false;
  bool _shuffle = false;
  bool _repeat = false;
  double _positionSeconds = 53;

  GloryTrack get _track => widget.album.tracks[_index];

  void _skip(int delta) {
    final count = widget.album.tracks.length;
    setState(() {
      _index = (_index + delta) % count;
      if (_index < 0) _index += count;
      _positionSeconds = 0;
    });
  }

  String _fmt(double seconds) {
    final d = Duration(seconds: seconds.round());
    final m = d.inMinutes;
    final s = d.inSeconds % 60;
    return '$m:${s.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final total = _track.duration.inSeconds.toDouble();
    final position = _positionSeconds.clamp(0, total);

    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back_ios_new, color: gloryInk, size: 20),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                  const Expanded(
                    child: Text(
                      'Now Playing',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600, color: gloryInk),
                    ),
                  ),
                  IconButton(icon: const Icon(Icons.more_horiz, color: gloryInk), onPressed: () {}),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Container(
              width: 218,
              height: 258,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(24),
                gradient: LinearGradient(colors: widget.album.gradient, begin: Alignment.topLeft, end: Alignment.bottomRight),
                boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: .25), blurRadius: 30, offset: const Offset(0, 16))],
              ),
              child: const Icon(Icons.person, color: Colors.white24, size: 96),
            ),
            const SizedBox(height: 28),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Text(
                _track.title,
                textAlign: TextAlign.center,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w500, color: gloryInk),
              ),
            ),
            const SizedBox(height: 8),
            Text(widget.album.artist, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: gloryInk.withValues(alpha: .4))),
            const Spacer(),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  IconButton(
                    icon: Icon(_liked ? Icons.favorite : Icons.favorite_border, color: _liked ? Colors.redAccent : gloryInk.withValues(alpha: .6)),
                    onPressed: () => setState(() => _liked = !_liked),
                  ),
                  IconButton(
                    icon: Icon(Icons.playlist_add, color: gloryInk.withValues(alpha: .6)),
                    onPressed: () {},
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 26),
              child: Column(
                children: [
                  SliderTheme(
                    data: SliderTheme.of(context).copyWith(
                      trackHeight: 6,
                      thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 8),
                      activeTrackColor: gloryInk,
                      inactiveTrackColor: gloryInk.withValues(alpha: .15),
                      thumbColor: gloryInk,
                      overlayShape: SliderComponentShape.noOverlay,
                    ),
                    child: Slider(
                      min: 0,
                      max: total,
                      value: position.toDouble(),
                      onChanged: (v) => setState(() => _positionSeconds = v),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(_fmt(position.toDouble()), style: TextStyle(fontSize: 13, color: gloryInk.withValues(alpha: .7))),
                        Text(_fmt(total), style: TextStyle(fontSize: 13, color: gloryInk.withValues(alpha: .7))),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  IconButton(
                    icon: Icon(Icons.shuffle, color: _shuffle ? gloryAccent : gloryInk.withValues(alpha: .3)),
                    onPressed: () => setState(() => _shuffle = !_shuffle),
                  ),
                  IconButton(
                    icon: const Icon(Icons.skip_previous, color: gloryInk, size: 34),
                    onPressed: () => _skip(-1),
                  ),
                  GestureDetector(
                    onTap: () => setState(() => _playing = !_playing),
                    child: Container(
                      width: 72,
                      height: 72,
                      decoration: const BoxDecoration(shape: BoxShape.circle, color: gloryInk),
                      child: Icon(_playing ? Icons.pause : Icons.play_arrow, color: gloryBg, size: 34),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.skip_next, color: gloryInk, size: 34),
                    onPressed: () => _skip(1),
                  ),
                  IconButton(
                    icon: Icon(Icons.repeat, color: _repeat ? gloryAccent : gloryInk.withValues(alpha: .3)),
                    onPressed: () => setState(() => _repeat = !_repeat),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
      bottomNavigationBar: const GloryBottomNav(),
    );
  }
}
