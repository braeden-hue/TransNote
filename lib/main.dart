import 'package:flutter/material.dart';
import 'screens/splash_screen.dart';
import 'theme/glory_theme.dart';
import 'theme/theme_controller.dart';

void main() => runApp(const MusicScoreApp());

class MusicScoreApp extends StatelessWidget {
  const MusicScoreApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: ThemeController.instance.isDark,
      builder: (context, isDark, _) {
        return MaterialApp(
          title: '커스텀 악보',
          debugShowCheckedModeBanner: false,
          themeMode: isDark ? ThemeMode.dark : ThemeMode.light,
          theme: buildGloryThemeData(Brightness.light),
          darkTheme: buildGloryThemeData(Brightness.dark),
          home: const SplashScreen(),
        );
      },
    );
  }
}
