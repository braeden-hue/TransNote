import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import 'tutorial_screen.dart';
import 'score_screen.dart';
import 'collection_screen.dart';

class HomeMenuScreen extends StatelessWidget {
  const HomeMenuScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 12),
              const Text('무엇을 해볼까요?',
                  style: TextStyle(color: gloryInk, fontSize: 24, fontWeight: FontWeight.bold)),
              const SizedBox(height: 6),
              Text('아래에서 하나를 골라 시작하세요',
                  style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 13)),
              const SizedBox(height: 28),
              _MenuCard(
                icon: Icons.school_outlined,
                title: '튜토리얼',
                subtitle: '커스텀 악보 읽는 법을 3가지 규칙으로 배워요',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const TutorialScreen()),
                ),
              ),
              const SizedBox(height: 16),
              _MenuCard(
                icon: Icons.library_music_outlined,
                title: '예시 악보 체험',
                subtitle: '샘플 악보를 감상하거나 직접 연주해봐요',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const ScoreScreen()),
                ),
              ),
              const SizedBox(height: 16),
              _MenuCard(
                icon: Icons.camera_alt_outlined,
                title: '악보 모음집',
                subtitle: '악보를 촬영해 커스텀 악보로 모아보세요',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const CollectionScreen()),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MenuCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _MenuCard({required this.icon, required this.title, required this.subtitle, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: glorySurface,
      borderRadius: BorderRadius.circular(20),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            boxShadow: [BoxShadow(color: gloryInk.withValues(alpha: .06), blurRadius: 12, offset: const Offset(0, 4))],
          ),
          child: Row(
            children: [
              Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: gloryAccent.withAlpha(28),
                ),
                child: Icon(icon, color: gloryAccent, size: 26),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: const TextStyle(color: gloryInk, fontSize: 17, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 4),
                    Text(subtitle,
                        style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12.5, height: 1.3)),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: gloryInk.withValues(alpha: .3)),
            ],
          ),
        ),
      ),
    );
  }
}
