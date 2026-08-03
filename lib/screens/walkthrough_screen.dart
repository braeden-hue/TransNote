import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import 'app_shell.dart';

/// Mume_Modified_UI_Kit_PNG의 워크스루(02~04) 3페이지 구성 참고 -- 실제 인물 사진 대신
/// 앱 전역에서 이미 쓰고 있는 그라디언트 원형 아이콘 일러스트로 대체(사진 에셋 없이도
/// 반응형으로 어느 화면 크기에서나 깨지지 않음).
class WalkthroughScreen extends StatefulWidget {
  const WalkthroughScreen({super.key});

  @override
  State<WalkthroughScreen> createState() => _WalkthroughScreenState();
}

class _WalkthroughPage {
  final IconData icon;
  final String headline;
  const _WalkthroughPage({required this.icon, required this.headline});
}

const _pages = [
  _WalkthroughPage(icon: Icons.camera_alt_rounded, headline: '원하는 악보를\n카메라로 촬영하세요'),
  _WalkthroughPage(icon: Icons.auto_awesome_rounded, headline: '촬영한 사진을\n연주하기 쉬운 커스텀 악보로\n바꿔드려요'),
  _WalkthroughPage(icon: Icons.piano_rounded, headline: '커스텀 악보로\n다양한 곡을 연주해보세요'),
];

class _WalkthroughScreenState extends State<WalkthroughScreen> {
  final _pageCtrl = PageController();
  int _page = 0;

  @override
  void dispose() {
    _pageCtrl.dispose();
    super.dispose();
  }

  void _next() {
    if (_page == _pages.length - 1) {
      Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const AppShell()));
      return;
    }
    _pageCtrl.nextPage(duration: const Duration(milliseconds: 320), curve: Curves.easeOutCubic);
  }

  @override
  Widget build(BuildContext context) {
    final isLast = _page == _pages.length - 1;
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView.builder(
                controller: _pageCtrl,
                itemCount: _pages.length,
                onPageChanged: (i) => setState(() => _page = i),
                itemBuilder: (context, i) => _WalkthroughPageView(page: _pages[i]),
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(_pages.length, (i) {
                final active = i == _page;
                return AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  margin: const EdgeInsets.symmetric(horizontal: 4),
                  width: active ? 22 : 8,
                  height: 8,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(4),
                    gradient: active ? gloryGradient : null,
                    color: active ? null : gloryBorder,
                  ),
                );
              }),
            ),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40),
                child: FilledButton(
                  onPressed: _next,
                  style: gloryFilledButtonStyle(),
                  child: Text(isLast ? '시작하기' : '다음'),
                ),
              ),
            ),
            const SizedBox(height: 28),
          ],
        ),
      ),
    );
  }
}

class _WalkthroughPageView extends StatelessWidget {
  final _WalkthroughPage page;
  const _WalkthroughPageView({required this.page});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(32, 24, 32, 8),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 220,
            height: 220,
            child: Stack(
              alignment: Alignment.center,
              children: [
                Positioned(
                  top: 6, left: 10,
                  child: _dot(14, gloryAccent2.withValues(alpha: .5)),
                ),
                Positioned(
                  bottom: 20, right: 4,
                  child: _dot(10, gloryAccent.withValues(alpha: .4)),
                ),
                Positioned(
                  top: 30, right: 20,
                  child: _dot(8, gloryAccent2.withValues(alpha: .35)),
                ),
                Container(
                  width: 176,
                  height: 176,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: gloryGradient,
                    boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .3), blurRadius: 32, offset: const Offset(0, 16))],
                  ),
                  child: Icon(page.icon, color: Colors.white, size: 76),
                ),
              ],
            ),
          ),
          const SizedBox(height: 40),
          Text(
            page.headline,
            textAlign: TextAlign.center,
            style: TextStyle(color: gloryInk, fontSize: 23, fontWeight: FontWeight.w800, height: 1.35),
          ),
        ],
      ),
    );
  }

  Widget _dot(double size, Color color) =>
      Container(width: size, height: size, decoration: BoxDecoration(shape: BoxShape.circle, color: color));
}
