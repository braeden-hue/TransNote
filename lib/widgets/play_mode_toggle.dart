import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';

/// 대보표(그랜드 스태프) 악보 재생 시 "양손 멜로디"(트레블+베이스 모두 자동 재생) 또는
/// "왼손 반주만"(베이스만 자동 재생, 트레블은 사용자가 직접 건반을 눌러 연주) 중 하나를
/// 고르는 세그먼트 토글. ScoreScreen/EntryDetailScreen/PlaylistPlayerScreen 세 화면에서
/// 동일하게 쓰여서 공용 위젯으로 분리함 -- score_screen.dart의 기존 _ModeToggle과 같은 스타일.
class PlayModeToggle extends StatelessWidget {
  final bool bassOnly;
  final bool enabled;
  final ValueChanged<bool> onChanged;
  const PlayModeToggle({
    super.key,
    required this.bassOnly,
    required this.onChanged,
    this.enabled = true,
  });

  @override
  Widget build(BuildContext context) {
    return Opacity(
      opacity: enabled ? 1.0 : .45,
      child: Container(
        padding: const EdgeInsets.all(3),
        decoration: BoxDecoration(color: gloryBg, borderRadius: BorderRadius.circular(12)),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _segment('양손 멜로디', !bassOnly, false),
            _segment('왼손 반주만', bassOnly, true),
          ],
        ),
      ),
    );
  }

  Widget _segment(String label, bool selected, bool value) {
    return GestureDetector(
      onTap: enabled ? () => onChanged(value) : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? gloryInk : Colors.transparent,
          borderRadius: BorderRadius.circular(9),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: selected ? gloryBg : gloryInk.withValues(alpha: .5),
            fontSize: 12.5,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}
