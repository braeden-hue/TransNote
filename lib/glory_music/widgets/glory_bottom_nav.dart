import 'package:flutter/material.dart';
import '../../theme/glory_theme.dart';

class GloryBottomNav extends StatelessWidget {
  const GloryBottomNav({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 90,
      padding: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: gloryNavBg,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          Icon(Icons.home_outlined, color: gloryInk.withValues(alpha: .3), size: 28),
          Icon(Icons.music_note, color: gloryAccent, size: 28),
          Icon(Icons.search, color: gloryInk.withValues(alpha: .3), size: 28),
          Icon(Icons.radio, color: gloryInk.withValues(alpha: .3), size: 28),
        ],
      ),
    );
  }
}
