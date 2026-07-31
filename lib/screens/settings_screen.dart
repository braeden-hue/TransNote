import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import '../theme/theme_controller.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      appBar: AppBar(
        backgroundColor: gloryBg,
        elevation: 0,
        iconTheme: IconThemeData(color: gloryInk),
        title: Text('설정', style: TextStyle(color: gloryInk, fontWeight: FontWeight.w600)),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: ValueListenableBuilder<bool>(
            valueListenable: ThemeController.instance.isDark,
            builder: (context, isDark, _) {
              return Material(
                color: glorySurface,
                borderRadius: BorderRadius.circular(16),
                child: SwitchListTile(
                  contentPadding: const EdgeInsets.symmetric(horizontal: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  value: isDark,
                  onChanged: ThemeController.instance.setDark,
                  activeThumbColor: gloryAccent,
                  title: Text('다크 모드', style: TextStyle(color: gloryInk, fontWeight: FontWeight.w600)),
                  subtitle: Text('화면 전체를 어두운 배경으로 전환해요',
                      style: TextStyle(color: gloryInk.withValues(alpha: .5), fontSize: 12.5)),
                  secondary: Icon(isDark ? Icons.dark_mode : Icons.light_mode, color: gloryAccent),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}
