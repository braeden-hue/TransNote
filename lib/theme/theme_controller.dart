import 'package:flutter/foundation.dart';

/// 앱 전체 다크모드 상태. glory_theme.dart의 색 getter들이 이 값을 참조하고,
/// main.dart의 MaterialApp이 ValueListenableBuilder로 감싸서 값이 바뀌면 전체 트리를
/// 다시 빌드한다(색상 getter들이 상수가 아니라 매 build마다 재평가되므로 자동 반영됨).
class ThemeController {
  ThemeController._();
  static final instance = ThemeController._();

  final ValueNotifier<bool> isDark = ValueNotifier(false);

  void toggle() => isDark.value = !isDark.value;
  void setDark(bool value) => isDark.value = value;
}
