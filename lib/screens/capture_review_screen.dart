import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';

/// Mume_Modified_UI_Kit_PNG 45번(capture_review) 참고. 촬영/크롭된 이미지를 확인하고
/// 다시 찍을지, 이 사진으로 진행할지 고른다. "다시 촬영"은 null을, "이 사진 사용"은
/// photo bytes를 그대로 pop해서 호출부(CollectionScreen)가 재촬영 루프를 돌 수 있게 한다.
class CaptureReviewScreen extends StatelessWidget {
  final Uint8List photo;
  const CaptureReviewScreen({super.key, required this.photo});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(4, 8, 16, 0),
              child: Row(
                children: [
                  IconButton(
                    icon: Icon(Icons.arrow_back, color: gloryInk),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                  Expanded(
                    child: Text('촬영 확인',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: gloryInk, fontSize: 17, fontWeight: FontWeight.w800)),
                  ),
                  const SizedBox(width: 40),
                ],
              ),
            ),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: glorySurface,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(14),
                    child: Container(
                      color: Colors.white,
                      alignment: Alignment.center,
                      child: Image.memory(photo, fit: BoxFit.contain),
                    ),
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 8),
              child: Text('악보가 선명하고 잘리지 않았는지 확인해주세요.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: gloryMuted, fontSize: 12.5)),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
              child: Row(
                children: [
                  Expanded(
                    child: TextButton(
                      onPressed: () => Navigator.of(context).pop(),
                      style: TextButton.styleFrom(
                        backgroundColor: glorySurface,
                        foregroundColor: gloryInk,
                        padding: const EdgeInsets.symmetric(vertical: 15),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                      ),
                      child: const Text('다시 촬영', style: TextStyle(fontWeight: FontWeight.w700)),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton(
                      onPressed: () => Navigator.of(context).pop(photo),
                      style: FilledButton.styleFrom(
                        backgroundColor: gloryAccent,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 15),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                      ),
                      child: const Text('이 사진 사용', style: TextStyle(fontWeight: FontWeight.w700)),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
