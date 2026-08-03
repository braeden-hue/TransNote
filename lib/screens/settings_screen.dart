import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import '../theme/theme_controller.dart';

/// 하단 탭 "설정" -- Mume_Modified_UI_Kit_PNG 30번(settings) 참고. 킷의 구독/백업/알림/
/// 언어/공유/FAQ 행은 이번 범위 밖이라 제외하고, 실제로 있는 기능(다크 모드)만 같은
/// 아이콘 리스트 행 스타일로 반영했다.
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
        children: [
          Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(shape: BoxShape.circle, gradient: gloryGradient),
                child: const Icon(Icons.music_note_rounded, color: Colors.white, size: 18),
              ),
              const SizedBox(width: 10),
              Text('설정', style: TextStyle(color: gloryInk, fontSize: 22, fontWeight: FontWeight.w800)),
            ],
          ),
          const SizedBox(height: 24),
          ValueListenableBuilder<bool>(
            valueListenable: ThemeController.instance.isDark,
            builder: (context, isDark, _) => _SettingsRow(
              icon: isDark ? Icons.dark_mode_rounded : Icons.light_mode_rounded,
              title: '다크 모드',
              subtitle: '화면 전체를 어두운 배경으로 전환해요',
              trailing: Switch(
                value: isDark,
                onChanged: ThemeController.instance.setDark,
                activeThumbColor: Colors.white,
                activeTrackColor: gloryAccent,
              ),
              onTap: () => ThemeController.instance.setDark(!isDark),
            ),
          ),
        ],
      ),
    );
  }
}

class _SettingsRow extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Widget trailing;
  final VoidCallback onTap;
  const _SettingsRow({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.trailing,
    required this.onTap,
  });

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
                width: 44,
                height: 44,
                decoration: BoxDecoration(color: glorySurface, borderRadius: BorderRadius.circular(12)),
                child: Icon(icon, color: gloryAccent, size: 22),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: TextStyle(color: gloryInk, fontSize: 15, fontWeight: FontWeight.w700)),
                    const SizedBox(height: 2),
                    Text(subtitle, style: TextStyle(color: gloryMuted, fontSize: 12)),
                  ],
                ),
              ),
              trailing,
            ],
          ),
        ),
      ),
    );
  }
}
