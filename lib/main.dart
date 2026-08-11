import 'package:flutter/material.dart';
import 'screens/capture_screen.dart';
import 'screens/practice_screen.dart';
import 'screens/project_info_screen.dart';
import 'screens/score_screen.dart';
import 'screens/tutorial_screen.dart';

void main() {
  runApp(const TransNoteApp());
}

class TransNoteApp extends StatelessWidget {
  const TransNoteApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TransNote',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0076CE))),
      home: const _HomeShell(),
    );
  }
}

class _HomeShell extends StatefulWidget {
  const _HomeShell();

  @override
  State<_HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<_HomeShell> {
  int _tab = 0;

  static const _titles = ['튜토리얼', '악보', '연습'];

  void _openCapture() {
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => const CaptureScreen()));
  }

  void _openProjectInfo() {
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => const ProjectInfoScreen()));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset('assets/icon/logo.png', height: 28),
            const SizedBox(width: 8),
            Text('TransNote · ${_titles[_tab]}'),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: '프로젝트 소개',
            onPressed: _openProjectInfo,
          ),
        ],
      ),
      body: IndexedStack(
        index: _tab,
        children: [
          TutorialScreen(onFinish: () => setState(() => _tab = 1)),
          const ScoreScreen(),
          const PracticeScreen(),
        ],
      ),
      // centerFloat: Flutter가 화면 진짜 가로 중앙에 자동으로 띄워준다 — 4번째
      // 칸을 만들어 수동으로 중앙을 맞추려던 것(밀리던 원인)과 달리 계산이 틀어질
      // 일이 없다. 하단 바 위로 살짝 뜨는 기본 여백도 정확히 "가운데보다 살짝 위".
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: FloatingActionButton(
        onPressed: _openCapture,
        tooltip: '악보 촬영',
        child: const Icon(Icons.camera_alt),
      ),
      bottomNavigationBar: BottomAppBar(
        padding: EdgeInsets.zero,
        // 3칸 균등 배치(1:1:1) — 촬영은 위 FAB이 따로 담당하므로 여기엔
        // 튜토리얼/악보/연습만 남는다.
        child: SafeArea(
          child: SizedBox(
            height: 64,
            child: Row(
              children: [
                Expanded(
                  child: _NavIconButton(
                    icon: Icons.school_outlined,
                    label: '튜토리얼',
                    selected: _tab == 0,
                    onTap: () => setState(() => _tab = 0),
                  ),
                ),
                Expanded(
                  child: _NavIconButton(
                    icon: Icons.music_note_outlined,
                    label: '악보',
                    selected: _tab == 1,
                    onTap: () => setState(() => _tab = 1),
                  ),
                ),
                Expanded(
                  child: _NavIconButton(
                    icon: Icons.piano_outlined,
                    label: '연습',
                    selected: _tab == 2,
                    onTap: () => setState(() => _tab = 2),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _NavIconButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _NavIconButton({
    required this.icon,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = selected
        ? Theme.of(context).colorScheme.primary
        : Theme.of(context).colorScheme.onSurfaceVariant;
    return InkWell(
      onTap: onTap,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: color),
          const SizedBox(height: 2),
          Text(label, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}
