import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import '../theme/glory_theme.dart';

/// Mume_Modified_UI_Kit_PNG 46번(OMR_pending) 참고. 실제 OMR 모델이 아직 학습 중이라는
/// 프로젝트의 현재 상태를 그대로 안내하는 화면 -- 킷 원본 문구(모델 준비되면 연결)와
/// 우리 앱의 실제 동작(데모 샘플로 즉시 결과를 보여줌, EntryDetailScreen의 "인식 완료(데모)"
/// 배지와 동일한 맥락)을 맞춰 문구만 조정했다. "변환 결과 보기"를 누르면 true를 pop해서
/// 호출부가 스캔 연출 + 결과 화면으로 이어가게 한다.
class CaptureDoneScreen extends StatelessWidget {
  final Uint8List photo;
  const CaptureDoneScreen({super.key, required this.photo});

  // TEMP/개발용 -- 우리 네이티브 OMR 엔진(ml/omr/engine)은 아직 Android 빌드가 안 되고
  // (CMake/prefab 불일치, CLAUDE.md "Known Gaps" 참고) 모델도 학습 중이라, 지금 당장
  // "진짜 인식"을 보여주고 싶을 때만 쓰는 임시 다리. 우리 앱 안에 Andromr 코드를 전혀
  // 포함/링크하지 않고(AGPL-3.0, homr 기반이라 상업 배포와 라이선스 충돌 -- project.md
  // 참고) 기기에 이미 깔려 있는 Andromr 앱으로 사진만 공유(share) intent로 넘긴다.
  // 반드시 제출/배포용 빌드 전에 제거할 것.
  Future<void> _testWithAndromr(BuildContext context) async {
    final file = XFile.fromData(photo, name: 'score.jpg', mimeType: 'image/jpeg');
    await SharePlus.instance.share(ShareParams(
      files: [file],
      text: 'Andromr로 인식해보세요 (기기에 Andromr 앱이 설치되어 있어야 해요)',
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(4, 8, 20, 0),
              child: Row(
                children: [
                  IconButton(
                    icon: Icon(Icons.arrow_back, color: gloryInk),
                    onPressed: () => Navigator.of(context).pop(false),
                  ),
                  Expanded(
                    child: Text('OMR 변환',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: gloryInk, fontSize: 17, fontWeight: FontWeight.w800)),
                  ),
                  const SizedBox(width: 40),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 20, 24, 20),
                child: Column(
                  children: [
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 36, horizontal: 20),
                      decoration: BoxDecoration(
                        color: glorySurface,
                        borderRadius: BorderRadius.circular(24),
                      ),
                      child: Column(
                        children: [
                          Container(
                            width: 84,
                            height: 84,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              gradient: gloryGradient,
                              boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .3), blurRadius: 24, offset: const Offset(0, 12))],
                            ),
                            child: const Icon(Icons.check_rounded, color: Colors.white, size: 42),
                          ),
                          const SizedBox(height: 20),
                          Text('촬영 완료!',
                              style: TextStyle(color: gloryInk, fontSize: 21, fontWeight: FontWeight.w800)),
                          const SizedBox(height: 10),
                          Text(
                            '학습 중인 실제 OMR 모델 대신,\n지금은 데모 샘플 악보로 변환 결과를 보여드려요.',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: gloryMuted, fontSize: 13, height: 1.5),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 20),
                    Align(
                      alignment: Alignment.centerLeft,
                      child: Text('다음 개발 단계',
                          style: TextStyle(color: gloryInk, fontSize: 14, fontWeight: FontWeight.w800)),
                    ),
                    const SizedBox(height: 10),
                    _RoadmapItem(index: 1, text: 'OMR 추론 API 연결'),
                    _RoadmapItem(index: 2, text: '음표를 커스텀 악보 신호로 변환'),
                    _RoadmapItem(index: 3, text: '박자에 맞춰 내려오는 리듬 노트 생성'),
                    const SizedBox(height: 20),
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: () => _testWithAndromr(context),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: gloryMuted,
                          side: BorderSide(color: gloryBorder),
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                        ),
                        icon: const Icon(Icons.science_outlined, size: 18),
                        label: const Text('Andromr 앱으로 인식 테스트 (임시, 개발용)', style: TextStyle(fontSize: 12.5)),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 20),
              child: SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: () => Navigator.of(context).pop(true),
                  style: gloryFilledButtonStyle(),
                  child: const Text('변환 결과 보기'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _RoadmapItem extends StatelessWidget {
  final int index;
  final String text;
  const _RoadmapItem({required this.index, required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 20,
            height: 20,
            alignment: Alignment.center,
            decoration: BoxDecoration(shape: BoxShape.circle, color: gloryAccent.withValues(alpha: .15)),
            child: Text('$index', style: TextStyle(color: gloryAccent, fontSize: 11, fontWeight: FontWeight.w800)),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text, style: TextStyle(color: gloryInk.withValues(alpha: .75), fontSize: 13, height: 1.4)),
          ),
        ],
      ),
    );
  }
}
