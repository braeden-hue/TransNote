import 'package:flutter/material.dart';

/// 앱 전역에서 쓰는 화면 전환 -- 기본 MaterialPageRoute의 단순 좌우 슬라이드 대신
/// 페이드 + 살짝 위로 슬라이드(맑고 고급스러운 느낌, Mume 킷 참고)를 준다.
/// 사용법: Navigator.of(context).push(gloryPageRoute(builder: (_) => NextScreen()))
/// -- MaterialPageRoute(builder: ...)를 그대로 대체하면 됨(같은 시그니처).
Route<T> gloryPageRoute<T>({required WidgetBuilder builder}) {
  return PageRouteBuilder<T>(
    pageBuilder: (context, animation, secondaryAnimation) => builder(context),
    transitionDuration: const Duration(milliseconds: 320),
    reverseTransitionDuration: const Duration(milliseconds: 260),
    transitionsBuilder: (context, animation, secondaryAnimation, child) {
      final curved = CurvedAnimation(parent: animation, curve: Curves.easeOutCubic, reverseCurve: Curves.easeInCubic);
      return FadeTransition(
        opacity: curved,
        child: SlideTransition(
          position: Tween<Offset>(begin: const Offset(0, .04), end: Offset.zero).animate(curved),
          child: child,
        ),
      );
    },
  );
}
