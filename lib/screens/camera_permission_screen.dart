import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';

/// Mume_Modified_UI_Kit_PNG 43번(camera_permission) 참고 -- 실제 OS 권한 다이얼로그는
/// GuidedCameraScreen이 카메라를 초기화할 때(camera 플러그인이 자동으로) 뜨므로, 이 화면은
/// 그 전에 "왜 카메라가 필요한지"를 설명하는 자체 프리프롬프트 UI다. true를 pop하면 카메라
/// 화면으로 진행, false/null(뒤로가기)이면 취소.
class CameraPermissionScreen extends StatelessWidget {
  const CameraPermissionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(28, 20, 28, 20),
          child: Column(
            children: [
              const Spacer(flex: 2),
              Container(
                width: 116,
                height: 116,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: gloryGradient,
                  boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .3), blurRadius: 28, offset: const Offset(0, 14))],
                ),
                child: const Icon(Icons.camera_alt_rounded, color: Colors.white, size: 52),
              ),
              const SizedBox(height: 28),
              Text('카메라 권한이 필요해요',
                  style: TextStyle(color: gloryInk, fontSize: 21, fontWeight: FontWeight.w800)),
              const SizedBox(height: 12),
              Text(
                '악보를 촬영하려면 카메라 접근을 허용해주세요.\n사진은 분석 준비를 위해 기기에만 저장됩니다.',
                textAlign: TextAlign.center,
                style: TextStyle(color: gloryMuted, fontSize: 13.5, height: 1.5),
              ),
              const Spacer(flex: 3),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: () => Navigator.of(context).pop(true),
                  style: gloryFilledButtonStyle(),
                  child: const Text('카메라 허용'),
                ),
              ),
              const SizedBox(height: 10),
              SizedBox(
                width: double.infinity,
                child: TextButton(
                  onPressed: () => Navigator.of(context).pop(false),
                  style: TextButton.styleFrom(
                    backgroundColor: glorySurface,
                    foregroundColor: gloryMuted,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: const StadiumBorder(),
                  ),
                  child: const Text('나중에'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
