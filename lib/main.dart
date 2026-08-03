import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'screens/splash_screen.dart';
import 'theme/glory_theme.dart';
import 'theme/theme_controller.dart';

// 앱 진입 흐름: SplashScreen(탭) -> WalkthroughScreen(3페이지 온보딩) -> AppShell(하단
// 4탭: 홈/촬영/플레이리스트/설정). Mume_Modified_UI_Kit_PNG 참고, screens/app_shell.dart
// 및 screens/walkthrough_screen.dart 참고.

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // 앱 기본은 세로(portrait) 고정. 피아노 건반이 나오는 화면(TutorialScreen/ScoreScreen/
  // EntryDetailScreen/PlaylistPlayerScreen)은 각 화면이 자체적으로 진입 시 가로로 전환하고
  // 이탈 시 다시 이 기본값(세로)으로 되돌린다 -- lib/utils/orientation_lock.dart 참고.
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  runApp(const MusicScoreApp());
}

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
