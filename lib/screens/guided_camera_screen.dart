import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../theme/glory_theme.dart';
import 'collection_screen.dart' show FrostedCircleButton;

/// 위/아래에 다른 시스템이 걸쳐 있는 페이지에서 "가운데 원하는 대보표 한 세트"만
/// 인식시키고 싶다는 요청에 대응 -- 오선 검출(detect_staffs)에 맡기는 대신, 촬영
/// 시점에 사용자가 직접 가이드 박스에 맞춰 프레이밍하게 하고 그 영역만 잘라서 넘긴다.
/// 자동 검출의 모호함(주변 오선이 섞여 들어오는 문제)을 촬영 UX로 원천 차단하는 접근.
class GuidedCameraScreen extends StatefulWidget {
  const GuidedCameraScreen({super.key});

  @override
  State<GuidedCameraScreen> createState() => _GuidedCameraScreenState();
}

class _GuidedCameraScreenState extends State<GuidedCameraScreen> {
  CameraController? _controller;
  bool _grandStaff = true;
  bool _capturing = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = '사용 가능한 카메라가 없어요.');
        return;
      }
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(back, ResolutionPreset.high, enableAudio: false);
      await controller.initialize();
      if (!mounted) {
        controller.dispose();
        return;
      }
      setState(() => _controller = controller);
    } catch (e) {
      setState(() => _error = '카메라를 열 수 없어요: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  // 가이드 박스 세로 비율(프레임 기준) -- 대보표(높은음자리+낮은음자리)는 더 크게.
  double get _guideHeightFrac => _grandStaff ? 0.34 : 0.16;
  double get _guideWidthFrac => 0.88;

  Rect _guideRect(Size frame) {
    final h = frame.height * _guideHeightFrac;
    final w = frame.width * _guideWidthFrac;
    return Rect.fromLTWH((frame.width - w) / 2, (frame.height - h) / 2, w, h);
  }

  Future<void> _onCapture() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _capturing) return;
    setState(() => _capturing = true);
    try {
      final file = await controller.takePicture();
      final bytes = await file.readAsBytes();
      final cropped = await _cropToGuide(bytes);
      if (mounted) Navigator.of(context).pop(cropped);
    } catch (e) {
      if (mounted) setState(() => _error = '촬영에 실패했어요: $e');
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  // 캡처된 원본 이미지에 가이드와 동일한 비율(중앙, _guideWidthFrac x _guideHeightFrac)의
  // 영역만 잘라낸다 -- CameraPreview를 controller.value.aspectRatio로 고정해서 보여주므로
  // 화면 가이드 비율이 실제 캡처 이미지 비율에 그대로 대응된다.
  Future<Uint8List> _cropToGuide(Uint8List jpgBytes) async {
    final codec = await ui.instantiateImageCodec(jpgBytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final w = image.width.toDouble();
    final h = image.height.toDouble();
    final rect = _guideRect(Size(w, h));

    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    canvas.drawImageRect(image, rect, Rect.fromLTWH(0, 0, rect.width, rect.height), Paint());
    final picture = recorder.endRecording();
    final cropped = await picture.toImage(rect.width.round(), rect.height.round());
    final byteData = await cropped.toByteData(format: ui.ImageByteFormat.png);
    return byteData!.buffer.asUint8List();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      backgroundColor: const Color(0xFF030A1C),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: Row(
                children: [
                  FrostedCircleButton(
                    icon: Icons.arrow_back,
                    alwaysLight: true,
                    onTap: () => Navigator.of(context).pop(),
                  ),
                  Expanded(
                    child: Text('악보 촬영',
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w800)),
                  ),
                  const SizedBox(width: 40),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _ModeChip(label: '오선 1개', selected: !_grandStaff, onTap: () => setState(() => _grandStaff = false)),
                  const SizedBox(width: 8),
                  _ModeChip(label: '대보표(2개)', selected: _grandStaff, onTap: () => setState(() => _grandStaff = true)),
                ],
              ),
            ),
            Expanded(
              child: Center(
                child: _error != null
                    ? Padding(
                        padding: const EdgeInsets.all(24),
                        child: Text(_error!, style: const TextStyle(color: Colors.white70), textAlign: TextAlign.center),
                      )
                    : (controller == null || !controller.value.isInitialized)
                        ? const CircularProgressIndicator(color: Colors.white)
                        : AspectRatio(
                            aspectRatio: controller.value.aspectRatio,
                            child: LayoutBuilder(
                              builder: (context, constraints) {
                                final frame = Size(constraints.maxWidth, constraints.maxHeight);
                                return Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    CameraPreview(controller),
                                    CustomPaint(
                                      painter: _GuideOverlayPainter(
                                        rect: _guideRect(frame),
                                        grandStaff: _grandStaff,
                                      ),
                                    ),
                                    Positioned(
                                      top: 12,
                                      left: 12,
                                      right: 12,
                                      child: Center(
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                          decoration: BoxDecoration(
                                            color: Colors.black.withValues(alpha: .55),
                                            borderRadius: BorderRadius.circular(20),
                                          ),
                                          child: Text(
                                            _grandStaff ? '대보표(높은음자리+낮은음자리)를 박스 안에 맞춰주세요' : '오선 하나를 박스 안에 맞춰주세요',
                                            style: const TextStyle(color: Colors.white, fontSize: 12.5),
                                            textAlign: TextAlign.center,
                                          ),
                                        ),
                                      ),
                                    ),
                                  ],
                                );
                              },
                            ),
                          ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 20),
              child: GestureDetector(
                onTap: (controller != null && controller.value.isInitialized) ? _onCapture : null,
                child: Container(
                  width: 72,
                  height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white,
                    border: Border.all(color: _capturing ? Colors.grey : gloryAccent2, width: 4),
                  ),
                  child: _capturing
                      ? Padding(
                          padding: const EdgeInsets.all(22),
                          child: CircularProgressIndicator(color: gloryAccent, strokeWidth: 3),
                        )
                      : null,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModeChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _ModeChip({required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? gloryAccent : Colors.white.withValues(alpha: .12),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(label,
            style: TextStyle(
              color: Colors.white,
              fontSize: 12.5,
              fontWeight: selected ? FontWeight.bold : FontWeight.normal,
            )),
      ),
    );
  }
}

class _GuideOverlayPainter extends CustomPainter {
  final Rect rect;
  final bool grandStaff;
  const _GuideOverlayPainter({required this.rect, required this.grandStaff});

  // 오선 5줄 그룹을 targetRect 안에 그린다. 실제 촬영된 오선을 이 가이드 줄에 겹치게
  // 맞추도록 유도해서, 학습 데이터의 오선 간격/기울기 분포에 더 가까운 프레이밍을
  // 자연스럽게 유도한다 -- 인식률에 큰 영향을 주는 건 결국 "오선이 얼마나 수평이고
  // 일정한 간격인가"이므로, 박스 하나만 보여주는 것보다 훨씬 직접적인 가이드.
  void _drawStaffLines(Canvas canvas, Rect target) {
    final linePaint = Paint()
      ..color = Colors.white.withValues(alpha: .85)
      ..strokeWidth = 1.6;
    final gap = target.height / 4;
    for (int i = 0; i < 5; i++) {
      final y = target.top + gap * i;
      canvas.drawLine(Offset(target.left, y), Offset(target.right, y), linePaint);
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    final outer = Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    final inner = Path()..addRRect(RRect.fromRectAndRadius(rect, const Radius.circular(8)));
    final mask = Path.combine(PathOperation.difference, outer, inner);
    canvas.drawPath(mask, Paint()..color = Colors.black.withValues(alpha: .55));

    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(8)),
      Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5,
    );

    // 오선 개수만큼(단일=1세트, 대보표=2세트) 실제 오선 줄 가이드를 그린다.
    if (grandStaff) {
      final trebleBand = Rect.fromLTWH(rect.left, rect.top + rect.height * 0.10,
          rect.width, rect.height * 0.32);
      final bassBand = Rect.fromLTWH(rect.left, rect.top + rect.height * 0.58,
          rect.width, rect.height * 0.32);
      _drawStaffLines(canvas, trebleBand);
      _drawStaffLines(canvas, bassBand);
    } else {
      final band = Rect.fromLTWH(rect.left, rect.top + rect.height * 0.30,
          rect.width, rect.height * 0.40);
      _drawStaffLines(canvas, band);
    }

    const len = 22.0;
    final corner = Paint()
      ..color = gloryAccent2
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;
    final tl = rect.topLeft, tr = rect.topRight, bl = rect.bottomLeft, br = rect.bottomRight;
    canvas.drawLine(tl, tl + const Offset(len, 0), corner);
    canvas.drawLine(tl, tl + const Offset(0, len), corner);
    canvas.drawLine(tr, tr + const Offset(-len, 0), corner);
    canvas.drawLine(tr, tr + const Offset(0, len), corner);
    canvas.drawLine(bl, bl + const Offset(len, 0), corner);
    canvas.drawLine(bl, bl + const Offset(0, -len), corner);
    canvas.drawLine(br, br + const Offset(-len, 0), corner);
    canvas.drawLine(br, br + const Offset(0, -len), corner);
  }

  @override
  bool shouldRepaint(covariant _GuideOverlayPainter old) =>
      old.rect != rect || old.grandStaff != grandStaff;
}
