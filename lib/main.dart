import 'package:flutter/material.dart';
import 'screens/splash_screen.dart';
import 'theme/glory_theme.dart';

void main() => runApp(const MusicScoreApp());

class MusicScoreApp extends StatelessWidget {
  const MusicScoreApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '커스텀 악보',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: gloryAccent, brightness: Brightness.light),
        scaffoldBackgroundColor: gloryBg,
        useMaterial3: true,
      ),
      home: const SplashScreen(),
    );
  }
}
