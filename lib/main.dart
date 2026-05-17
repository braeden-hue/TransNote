import 'package:flutter/material.dart';
import 'screens/tutorial_screen.dart';
import 'screens/score_screen.dart';
import 'screens/practice_screen.dart';

void main() => runApp(const MusicScoreApp());

class MusicScoreApp extends StatelessWidget {
  const MusicScoreApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '커스텀 악보',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF5BC0EB), brightness: Brightness.dark),
        scaffoldBackgroundColor: const Color(0xFF0d0d1f),
        useMaterial3: true,
      ),
      home: const _AppShell(),
    );
  }
}

class _AppShell extends StatefulWidget {
  const _AppShell();

  @override
  State<_AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<_AppShell> {
  int _tab = 0;

  static const _screens = [
    TutorialScreen(),
    ScoreScreen(),
    PracticeScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0d0d1f),
      body: IndexedStack(index: _tab, children: _screens),
      bottomNavigationBar: NavigationBar(
        backgroundColor: const Color(0xFF0a0a1a),
        indicatorColor: const Color(0xFF5BC0EB).withAlpha(50),
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.school_outlined),
            selectedIcon: Icon(Icons.school, color: Color(0xFF5BC0EB)),
            label: '튜토리얼',
          ),
          NavigationDestination(
            icon: Icon(Icons.library_music_outlined),
            selectedIcon: Icon(Icons.library_music, color: Color(0xFF5BC0EB)),
            label: '악보',
          ),
          NavigationDestination(
            icon: Icon(Icons.piano_outlined),
            selectedIcon: Icon(Icons.piano, color: Color(0xFF5BC0EB)),
            label: '연주',
          ),
        ],
      ),
    );
  }
}
