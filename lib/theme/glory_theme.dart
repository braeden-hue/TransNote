import 'package:flutter/material.dart';
import 'theme_controller.dart';

/// Mume Music Player UI 킷(피그마) 팔레트를 그대로 가져온 앱 전역 색상 -- 라이트/다크
/// 두 세트를 두고 ThemeController.instance.isDark 값에 따라 실시간으로 골라 쓴다. 상수
/// (const)가 아니라 getter라서 화면이 다시 빌드될 때마다 최신 값을 반환하고, MaterialApp을
/// ValueListenableBuilder로 감싸 다크모드 토글 시 전체 트리가 다시 빌드되게 해뒀다
/// (main.dart 참고). 기존 코드가 전부 gloryBg/gloryInk 등을 "상수처럼" 참조하고 있었기
/// 때문에, 이름은 그대로 두고 const -> getter로만 바꿔서 호출부 수정을 최소화했다.
///
/// 2026-07-30: 기존 갈색/크림톤("Glory") 팔레트에서 Mume 킷 원본 오렌지/화이트로 전환
/// (사용자 확인 완료 -- "디자인 그대로 사용"). 변수 이름(glory*)은 하위 호환을 위해 유지.
bool get _dark => ThemeController.instance.isDark.value;

Color get gloryBg => _dark ? const Color(0xFF15120E) : const Color(0xFFFFFFFF);
Color get gloryInk => _dark ? const Color(0xFFFFFFFF) : const Color(0xFF14120E);
Color get gloryAccent => const Color(0xFFFF8A00); // Mume Primary/500, 라이트/다크 공통
Color get glorySurface => _dark ? const Color(0xFF211D17) : const Color(0xFFF7F5F2);
Color get gloryNavBg => _dark ? const Color(0xFF211D17) : Colors.white;
Color get gloryBorder => _dark ? const Color(0xFF322C22) : const Color(0xFFEFEBE6);
Color get gloryMuted => gloryInk.withValues(alpha: .5);

ButtonStyle gloryFilledButtonStyle({Color? background}) => FilledButton.styleFrom(
      backgroundColor: background ?? gloryInk,
      foregroundColor: gloryBg,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      shape: const StadiumBorder(),
    );

ButtonStyle gloryOutlinedButtonStyle() => OutlinedButton.styleFrom(
      foregroundColor: gloryInk,
      side: BorderSide(color: gloryInk, width: 1.4),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      shape: const StadiumBorder(),
    );

/// Mume UI 킷(Design System > Typography)의 H1~H6/Body 스케일을 참고한 타이포그래피
/// 크기 체계. 화면마다 11~28px가 산발적으로 쓰이던 것을 정리하는 기준점 -- 기존 화면의
/// 텍스트 스타일을 일괄 교체하진 않았고, 앞으로 새 화면/수정 시 여기서 골라 쓰면 됨.
/// 색상은 원본 킷의 오렌지 대신 gloryInk를 기본으로 둬서 기존 팔레트와 맞춘다.
TextStyle get gloryH1 => TextStyle(fontSize: 48, fontWeight: FontWeight.bold, height: 1.2, color: gloryInk);
TextStyle get gloryH2 => TextStyle(fontSize: 40, fontWeight: FontWeight.bold, height: 1.2, color: gloryInk);
TextStyle get gloryH3 => TextStyle(fontSize: 32, fontWeight: FontWeight.bold, height: 1.2, color: gloryInk);
TextStyle get gloryH4 => TextStyle(fontSize: 24, fontWeight: FontWeight.bold, height: 1.2, color: gloryInk);
TextStyle get gloryH5 => TextStyle(fontSize: 20, fontWeight: FontWeight.bold, height: 1.2, color: gloryInk);
TextStyle get gloryH6 => TextStyle(fontSize: 18, fontWeight: FontWeight.bold, height: 1.2, color: gloryInk);

TextStyle get gloryBodyXLarge => TextStyle(fontSize: 18, fontWeight: FontWeight.w400, height: 1.4, color: gloryInk);
TextStyle get gloryBodyLarge  => TextStyle(fontSize: 16, fontWeight: FontWeight.w400, height: 1.4, color: gloryInk);
TextStyle get gloryBodyMedium => TextStyle(fontSize: 14, fontWeight: FontWeight.w400, height: 1.4, color: gloryInk);
TextStyle get gloryBodySmall  => TextStyle(fontSize: 12, fontWeight: FontWeight.w400, height: 1.2, color: gloryInk);

/// main.dart의 ThemeData(light/dark)에 쓰는 시드 색 -- 브랜드 액센트는 다크에서도
/// 동일하게(밝은 변형) 유지.
ThemeData buildGloryThemeData(Brightness brightness) {
  final dark = brightness == Brightness.dark;
  return ThemeData(
    colorScheme: ColorScheme.fromSeed(
      seedColor: const Color(0xFFFF8A00), // Mume Primary/500
      brightness: brightness,
    ),
    scaffoldBackgroundColor: dark ? const Color(0xFF15120E) : const Color(0xFFFFFFFF),
    useMaterial3: true,
  );
}
