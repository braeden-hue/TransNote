import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import 'walkthrough_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late final AnimationController _loadCtrl;
  bool _navigating = false;

  @override
  void initState() {
    super.initState();
    _loadCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))..repeat();
  }

  @override
  void dispose() {
    _loadCtrl.dispose();
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
        pageBuilder: (_, _, _) => const WalkthroughScreen(),
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
        child: GestureDetector(
          onTap: _enter,
          behavior: HitTestBehavior.opaque,
          child: Column(
            children: [
              const Spacer(flex: 3),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 56,
                    height: 56,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: gloryGradient,
                      boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .35), blurRadius: 24, offset: const Offset(0, 10))],
                    ),
                    child: const Icon(Icons.music_note_rounded, color: Colors.white, size: 30),
                  ),
                  const SizedBox(width: 14),
                  Text('악보의 대중화',
                      style: TextStyle(color: gloryInk, fontSize: 26, fontWeight: FontWeight.w800)),
                ],
              ),
              const SizedBox(height: 10),
              Text('유규태 · 조준성',
                  style: TextStyle(color: gloryMuted, fontSize: 14, fontWeight: FontWeight.w600)),
              const Spacer(flex: 4),
              AnimatedBuilder(
                animation: _loadCtrl,
                builder: (context, _) => Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(5, (i) {
                    final t = (_loadCtrl.value * 5 - i) % 5;
                    final scale = 0.5 + 0.5 * (1 - (t.clamp(0, 1)));
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 3),
                      child: Opacity(
                        opacity: 0.35 + 0.65 * scale,
                        child: Container(
                          width: 8 * scale.clamp(0.5, 1.0),
                          height: 8 * scale.clamp(0.5, 1.0),
                          decoration: BoxDecoration(shape: BoxShape.circle, gradient: gloryGradient),
                        ),
                      ),
                    );
                  }),
                ),
              ),
              const SizedBox(height: 16),
              Text('화면을 탭해 시작하기', style: TextStyle(color: gloryAccent.withValues(alpha: .85), fontSize: 12.5, fontWeight: FontWeight.w600)),
              const Spacer(flex: 3),
            ],
          ),
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
