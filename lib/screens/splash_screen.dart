import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import 'home_menu_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late final AnimationController _pulseCtrl;
  bool _navigating = false;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 1400))..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  void _enter() {
    if (_navigating) return;
    _navigating = true;
    final size = MediaQuery.of(context).size;
    final center = Offset(size.width / 2, size.height / 2);
    Navigator.of(context).push(
      PageRouteBuilder(
        transitionDuration: const Duration(milliseconds: 550),
        pageBuilder: (_, _, _) => const HomeMenuScreen(),
        transitionsBuilder: (context, animation, _, child) {
          return AnimatedBuilder(
            animation: animation,
            builder: (context, _) => ClipPath(
              clipper: _CircleRevealClipper(fraction: Curves.easeOutCubic.transform(animation.value), center: center),
              child: child,
            ),
            child: child,
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Icon(Icons.more_horiz, color: gloryInk.withValues(alpha: .3)),
                ],
              ),
            ),
            const Spacer(),
            GestureDetector(
              onTap: _enter,
              child: AnimatedBuilder(
                animation: _pulseCtrl,
                builder: (context, child) {
                  final s = 1 + _pulseCtrl.value * 0.03;
                  return Transform.scale(scale: s, child: child);
                },
                child: Container(
                  width: 218,
                  height: 218,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: gloryInk,
                    boxShadow: [BoxShadow(color: gloryInk.withValues(alpha: .25), blurRadius: 36, offset: const Offset(0, 18))],
                  ),
                  child: Icon(Icons.music_note, color: gloryBg, size: 84),
                ),
              ),
            ),
            const SizedBox(height: 32),
            const Text('악보의 대중화 프로젝트',
                textAlign: TextAlign.center,
                style: TextStyle(color: gloryInk, fontSize: 24, fontWeight: FontWeight.w500)),
            const SizedBox(height: 8),
            Text('유규태, 조준성',
                style: TextStyle(color: gloryInk.withValues(alpha: .4), fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 20),
            Text('로고를 눌러 시작하기', style: TextStyle(color: gloryAccent.withValues(alpha: .8), fontSize: 12.5)),
            const Spacer(),
          ],
        ),
      ),
    );
  }
}

class _CircleRevealClipper extends CustomClipper<Path> {
  final double fraction;
  final Offset center;
  const _CircleRevealClipper({required this.fraction, required this.center});

  @override
  Path getClip(Size size) {
    final maxRadius = (Offset(size.width, size.height) - Offset.zero).distance;
    return Path()..addOval(Rect.fromCircle(center: center, radius: maxRadius * fraction));
  }

  @override
  bool shouldReclip(covariant _CircleRevealClipper oldClipper) =>
      oldClipper.fraction != fraction || oldClipper.center != center;
}
