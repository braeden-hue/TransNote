import 'package:flutter/material.dart';

/// Glory Music UI 킷에서 가져온 앱 전역 팔레트.
const gloryBg = Color(0xFFF0E8E3);
const gloryInk = Color(0xFF222222);
const gloryAccent = Color(0xFFAB5F2B);
const glorySurface = Colors.white;
const gloryNavBg = Color(0xFFDDC9B4);
const gloryBorder = Color(0xFFE4D6C6);

ButtonStyle gloryFilledButtonStyle({Color? background}) => FilledButton.styleFrom(
      backgroundColor: background ?? gloryInk,
      foregroundColor: gloryBg,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      shape: const StadiumBorder(),
    );

ButtonStyle gloryOutlinedButtonStyle() => OutlinedButton.styleFrom(
      foregroundColor: gloryInk,
      side: const BorderSide(color: gloryInk, width: 1.4),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      shape: const StadiumBorder(),
    );
