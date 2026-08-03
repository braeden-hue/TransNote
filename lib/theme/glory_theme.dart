import 'package:flutter/material.dart';
import 'theme_controller.dart';

/// Mume_Modified UI 킷(팀원이 실제로 만든 앱 화면 PNG, Mume_Modified_UI_Kit_PNG/)
/// 팔레트를 스포이드로 추출해 반영한 앱 전역 색상 -- 라이트/다크 두 세트를 두고
/// ThemeController.instance.isDark 값에 따라 실시간으로 골라 쓴다. 상수(const)가 아니라
/// getter라서 화면이 다시 빌드될 때마다 최신 값을 반환하고, MaterialApp을
/// ValueListenableBuilder로 감싸 다크모드 토글 시 전체 트리가 다시 빌드되게 해뒀다
/// (main.dart 참고). 기존 코드가 전부 gloryBg/gloryInk 등을 "상수처럼" 참조하고 있었기
/// 때문에, 이름은 그대로 두고 const -> getter로만 바꿔서 호출부 수정을 최소화했다.
///
/// 2026-08-02: Mume_Modified_UI_Kit_PNG(팀원 수정본)의 실제 색상(블루 그라디언트
/// #1DC1FF -> #146BFF, 배경 #F7FAFF)으로 전환 -- 이전 "Mume 원본 오렌지"는 실제로는
/// 팀원이 아직 반영하기 전 구버전 킷 색상이었음. 변수 이름(glory*)은 하위 호환을 위해 유지.
bool get _dark => ThemeController.instance.isDark.value;

Color get gloryBg => _dark ? const Color(0xFF0A1220) : const Color(0xFFF7FAFF);
Color get gloryInk => _dark ? const Color(0xFFF2F6FF) : const Color(0xFF0B1B36);
Color get gloryAccent => const Color(0xFF146BFF); // Mume Primary(블루), 라이트/다크 공통
Color get gloryAccent2 => const Color(0xFF1DC1FF); // 그라디언트 보조색(시안)
Color get glorySurface => _dark ? const Color(0xFF16202E) : const Color(0xFFEFF4FC);
Color get gloryNavBg => _dark ? const Color(0xFF0F1826) : Colors.white;
Color get gloryBorder => _dark ? const Color(0xFF22304A) : const Color(0xFFE4EEFC);
Color get gloryMuted => gloryInk.withValues(alpha: .5);

/// 로고/카메라 아이콘/스캔 카드 등 Mume 킷의 시안->블루 대각선 그라디언트.
LinearGradient get gloryGradient => LinearGradient(
      begin: Alignment.topLeft,
      end: Alignment.bottomRight,
      colors: [gloryAccent2, gloryAccent],
    );

ButtonStyle gloryFilledButtonStyle({Color? background}) => FilledButton.styleFrom(
      backgroundColor: background ?? gloryAccent,
      foregroundColor: Colors.white,
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
      seedColor: const Color(0xFF146BFF), // Mume Primary(블루)
      brightness: brightness,
    ),
    scaffoldBackgroundColor: dark ? const Color(0xFF0A1220) : const Color(0xFFF7FAFF),
    useMaterial3: true,
  );
}
