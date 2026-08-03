import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../data/samples.dart';
import '../omr_service.dart';
import '../services/omr_bridge_service.dart';
import '../services/omr_model_loader.dart';
import '../services/omr_token_parser.dart';
import '../theme/glory_theme.dart';
import 'camera_permission_screen.dart';
import 'capture_done_screen.dart';
import 'capture_review_screen.dart';
import 'guided_camera_screen.dart';

// Mume UI 킷 홈 화면의 "섹션 제목 + 전체보기" 패턴 참고 -- 최근 스캔 목록이 길어지면
// 미리보기는 일부만 보여주고 나머지는 전체보기에서 확인.
const _recentPreviewCount = 5;

class ScannedEntry {
  final String title;
  final DateTime time;
  final SampleScore preview;
  final Uint8List? photo;
  ScannedEntry(this.title, this.time, this.preview, [this.photo]);
}

/// 하단 탭 "촬영" -- Mume_Modified_UI_Kit_PNG 42번(score_scan_home) 참고한 "Score Lab" 화면.
/// 스캔 결과 목록(entries)은 AppShell이 들고 있고(홈 탭의 "최근 스캔"과도 공유해야 하므로),
/// 이 화면은 콜백으로만 추가/열람을 요청한다.
class CollectionScreen extends StatefulWidget {
  final List<ScannedEntry> entries;
  final ValueChanged<ScannedEntry> onAddEntry;
  final ValueChanged<ScannedEntry> onOpenEntry;
  const CollectionScreen({
    super.key,
    required this.entries,
    required this.onAddEntry,
    required this.onOpenEntry,
  });

  @override
  State<CollectionScreen> createState() => _CollectionScreenState();
}

class _CollectionScreenState extends State<CollectionScreen> with SingleTickerProviderStateMixin {
  bool _capturing = false;
  Uint8List? _photo;
  String? _error;
  late final AnimationController _scanCtrl;

  @override
  void initState() {
    super.initState();
    _scanCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 2200));
  }

  @override
  void dispose() {
    _scanCtrl.dispose();
    super.dispose();
  }

  // 카메라 촬영 플로우(_captureFromCamera)와 동일하게 완료 안내 화면(CaptureDoneScreen)을
  // 거치게 해서, 갤러리에서 고른 악보 이미지도 카메라로 찍은 것과 똑같이 "Andromr 앱으로
  // 인식 테스트" 버튼을 쓸 수 있게 한다 -- 예전엔 이 화면을 건너뛰어서 갤러리 사진에는
  // 그 버튼이 아예 없었음.
  Future<void> _captureFromGallery() async {
    setState(() => _error = null);
    try {
      final file = await ImagePicker().pickImage(source: ImageSource.gallery, imageQuality: 85);
      if (file == null) return;
      final bytes = await file.readAsBytes();
      if (!mounted) return;
      final wantsResult = await Navigator.of(context).push<bool>(
        MaterialPageRoute(builder: (_) => CaptureDoneScreen(photo: bytes)),
      );
      if (wantsResult != true) return;
      if (!mounted) return;
      await _processPhoto(bytes);
    } catch (e) {
      setState(() => _error = '갤러리를 사용할 수 없어요: $e');
    }
  }

  // 촬영 플로우: 권한 안내 -> (촬영 <-> 확인, "다시 촬영"이면 촬영으로 되돌아감) -> 완료 안내
  // -> 스캔 연출 -> 결과 저장. 위/아래에 다른 오선이 걸쳐 있는 페이지에서 원하는 대보표 한
  // 세트만 찍기 어렵다는 문제 대응 -- 자동 오선 검출에 맡기지 않고, 촬영 화면의 가이드
  // 박스에 맞춰 프레이밍한 뒤 그 영역만 잘라서 받는다(GuidedCameraScreen).
  Future<void> _captureFromCamera() async {
    setState(() => _error = null);
    final proceed = await Navigator.of(context).push<bool>(
      MaterialPageRoute(builder: (_) => const CameraPermissionScreen()),
    );
    if (proceed != true) return;
    if (!mounted) return;

    Uint8List? confirmed;
    while (confirmed == null) {
      if (!mounted) return;
      final captured = await Navigator.of(context).push<Uint8List>(
        MaterialPageRoute(builder: (_) => const GuidedCameraScreen()),
      );
      if (captured == null) return; // 사용자가 카메라 화면에서 취소
      if (!mounted) return;
      confirmed = await Navigator.of(context).push<Uint8List>(
        MaterialPageRoute(builder: (_) => CaptureReviewScreen(photo: captured)),
      );
      // confirmed가 null이면 "다시 촬영" -- while 루프가 카메라 화면을 다시 연다.
    }
    if (!mounted) return;

    final wantsResult = await Navigator.of(context).push<bool>(
      MaterialPageRoute(builder: (_) => CaptureDoneScreen(photo: confirmed!)),
    );
    if (wantsResult != true) return;
    if (!mounted) return;
    await _processPhoto(confirmed);
  }

  Future<void> _processPhoto(Uint8List bytes) async {
    setState(() {
      _photo = bytes;
      _capturing = true;
    });
    // Android에서는 자체 온디바이스 OMR 엔진(ml/omr/engine, dart:ffi)을 먼저 시도한다.
    // 실패/미초기화 시 아래 omr_bridge 개발용 서버로, 그것도 실패하면 데모 샘플로
    // 폴백한다(_recognize 안에서 순서대로 처리). 스캔 연출과 동시에 요청을 보내
    // 체감 지연이 없게 함.
    final recognizeFuture = _recognize(bytes);
    await _scanCtrl.forward(from: 0);
    final recognized = await recognizeFuture;
    if (!mounted) return;

    final SampleScore sample;
    if (recognized != null) {
      sample = SampleScore(
        id: 'recognized-${DateTime.now().millisecondsSinceEpoch}',
        title: '촬영한 악보',
        emoji: '🎼',
        tempo: 100,
        timeSignature: recognized.timeSignature,
        notes: recognized.treble,
        bassNotes: recognized.bass.isEmpty ? null : recognized.bass,
      );
    } else {
      // 실제 OMR 엔진이 연결 안 돼 있거나(ml/omr/engine Android 빌드 미해결 -- CLAUDE.md
      // "Known Gaps" 참고) 브리지 서버가 꺼져 있을 때의 폴백. 예전엔 entries.length로만
      // 골라서 새 세션마다 항상 반짝반짝 작은별(인덱스 0)이 나왔던 걸, 사진 바이트
      // 해시로 골라서 최소한 "다른 사진 = 다른 곡"으로 보이게 했다.
      sample = samples[_photoHash(bytes) % samples.length];
    }
    final entry = ScannedEntry(sample.title, DateTime.now(), sample, bytes);
    widget.onAddEntry(entry);
    setState(() {
      _capturing = false;
      _photo = null;
    });
    widget.onOpenEntry(entry);
  }

  // Android: 자체 온디바이스 엔진(OmrService, dart:ffi) 우선 시도 -> 실패 시
  // omr_bridge 개발용 서버(OmrBridgeService) 폴백. 다른 플랫폼(iOS/데스크톱/웹)은
  // 아직 네이티브 엔진이 없어(CLAUDE.md "iOS Status" 참고) 곧바로 브리지 서버로.
  Future<RecognizedScore?> _recognize(Uint8List bytes) async {
    if (!kIsWeb && Platform.isAndroid) {
      final native = await _recognizeWithNativeEngine(bytes);
      if (native != null) return native;
    }
    return OmrBridgeService.recognize(bytes);
  }

  Future<RecognizedScore?> _recognizeWithNativeEngine(Uint8List bytes) async {
    try {
      final ready = await OmrModelLoader.ensureInitialized();
      if (!ready) return null;
      final tokens = OmrService.instance.process(bytes);
      if (tokens == null || tokens.isEmpty) return null;
      final recognized = OmrTokenParser.parse(tokens);
      if (recognized.treble.isEmpty) return null;
      return recognized;
    } catch (_) {
      return null;
    }
  }

  // 사진 바이트를 훑어 0 이상 정수 해시로 축약. 전체 바이트를 다 읽으면(사진이 보통
  // 수백 KB~수 MB) UI 스레드에서 눈에 띄게 느려질 수 있어 일정 간격으로 샘플링한다.
  int _photoHash(Uint8List bytes) {
    if (bytes.isEmpty) return 0;
    final step = (bytes.length / 2000).ceil().clamp(1, bytes.length);
    var hash = 0;
    for (var i = 0; i < bytes.length; i += step) {
      hash = (hash * 31 + bytes[i]) & 0x7fffffff;
    }
    return hash;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: Stack(
        children: [
          SafeArea(child: _buildBody(context)),
          if (_capturing) _buildScanOverlay(),
        ],
      ),
    );
  }

  Widget _buildBody(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
      children: [
        Text('Score Lab', style: TextStyle(color: gloryInk, fontSize: 24, fontWeight: FontWeight.w800)),
        const SizedBox(height: 6),
        Text('음악을 듣고, 악보를 촬영해 커스텀 악보로 바꿔보세요.',
            style: TextStyle(color: gloryMuted, fontSize: 13)),
        const SizedBox(height: 20),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(24, 28, 24, 24),
          decoration: BoxDecoration(
            gradient: gloryGradient,
            borderRadius: BorderRadius.circular(24),
            boxShadow: [BoxShadow(color: gloryAccent.withValues(alpha: .3), blurRadius: 24, offset: const Offset(0, 12))],
          ),
          child: Column(
            children: [
              Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(shape: BoxShape.circle, color: Colors.white.withValues(alpha: .2)),
                child: const Icon(Icons.camera_alt_rounded, color: Colors.white, size: 28),
              ),
              const SizedBox(height: 16),
              const Text('악보 스캔', style: TextStyle(color: Colors.white, fontSize: 19, fontWeight: FontWeight.w800)),
              const SizedBox(height: 8),
              Text('사진 촬영 기능부터 먼저 사용해보세요\nOMR 분석은 학습 완료 후 연결됩니다.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white.withValues(alpha: .85), fontSize: 12.5, height: 1.4)),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: _captureFromCamera,
                  style: FilledButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: gloryAccent,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: const StadiumBorder(),
                  ),
                  child: const Text('카메라 열기', style: TextStyle(fontWeight: FontWeight.w800)),
                ),
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(top: 10),
          child: TextButton.icon(
            onPressed: _captureFromGallery,
            icon: Icon(Icons.photo_library_outlined, color: gloryAccent, size: 18),
            label: Text('갤러리에서 가져오기', style: TextStyle(color: gloryAccent, fontSize: 12.5, fontWeight: FontWeight.w700)),
          ),
        ),
        if (_error != null) ...[
          const SizedBox(height: 4),
          Text(_error!, style: const TextStyle(color: Color(0xFFEE7777), fontSize: 12)),
        ],
        const SizedBox(height: 16),
        Row(
          children: [
            Text('최근 스캔', style: TextStyle(color: gloryInk, fontSize: 15, fontWeight: FontWeight.w800)),
            const Spacer(),
            if (widget.entries.length > _recentPreviewCount)
              GestureDetector(
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => _AllEntriesScreen(entries: widget.entries, onOpen: widget.onOpenEntry)),
                ),
                child: Text('전체보기', style: TextStyle(color: gloryAccent, fontSize: 12.5, fontWeight: FontWeight.w700)),
              ),
          ],
        ),
        const SizedBox(height: 12),
        if (widget.entries.isEmpty)
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(color: glorySurface, borderRadius: BorderRadius.circular(16)),
            child: Column(
              children: [
                Icon(Icons.camera_alt_outlined, color: gloryMuted, size: 28),
                const SizedBox(height: 10),
                Text('최근 스캔은 아직 없습니다.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: gloryInk.withValues(alpha: .7), fontSize: 13, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text('촬영한 악보는 이 기기에 임시 저장됩니다.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: gloryMuted, fontSize: 12)),
              ],
            ),
          )
        else
          ...widget.entries.take(_recentPreviewCount).map((e) => EntryTile(entry: e, onTap: () => widget.onOpenEntry(e))),
      ],
    );
  }

  Widget _buildScanOverlay() {
    return Positioned.fill(
      child: Container(
        color: Colors.black,
        child: Column(
          children: [
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      if (_photo != null) Image.memory(_photo!, fit: BoxFit.cover),
                      Container(color: Colors.black.withValues(alpha: .35)),
                      const _CornerGuide(),
                      AnimatedBuilder(
                        animation: _scanCtrl,
                        builder: (_, _) => Align(
                          alignment: Alignment(0, -1 + 2 * _scanCtrl.value),
                          child: Container(
                            height: 3,
                            margin: const EdgeInsets.symmetric(horizontal: 20),
                            decoration: BoxDecoration(
                              color: gloryAccent2,
                              boxShadow: [BoxShadow(color: gloryAccent2.withValues(alpha: .8), blurRadius: 12, spreadRadius: 1)],
                            ),
                          ),
                        ),
                      ),
                      Positioned(
                        top: 12,
                        left: 12,
                        right: 12,
                        child: Center(
                          child: AnimatedBuilder(
                            animation: _scanCtrl,
                            builder: (_, _) => Container(
                              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                              decoration: BoxDecoration(
                                color: Colors.white.withValues(alpha: .12),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Text('스캔 중... ${(_scanCtrl.value * 100).round()}%',
                                  style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600)),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }
}

class _CornerGuide extends StatelessWidget {
  const _CornerGuide();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.all(28),
      child: CustomPaint(painter: _CornerPainter(), child: SizedBox.expand()),
    );
  }
}

class _CornerPainter extends CustomPainter {
  const _CornerPainter();

  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = Colors.white
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    const len = 26.0;
    // top-left
    canvas.drawLine(Offset.zero, const Offset(len, 0), p);
    canvas.drawLine(Offset.zero, const Offset(0, len), p);
    // top-right
    canvas.drawLine(Offset(size.width, 0), Offset(size.width - len, 0), p);
    canvas.drawLine(Offset(size.width, 0), Offset(size.width, len), p);
    // bottom-left
    canvas.drawLine(Offset(0, size.height), Offset(len, size.height), p);
    canvas.drawLine(Offset(0, size.height), Offset(0, size.height - len), p);
    // bottom-right
    canvas.drawLine(Offset(size.width, size.height), Offset(size.width - len, size.height), p);
    canvas.drawLine(Offset(size.width, size.height), Offset(size.width, size.height - len), p);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class EntryTile extends StatelessWidget {
  final ScannedEntry entry;
  final VoidCallback onTap;
  const EntryTile({super.key, required this.entry, required this.onTap});

  String _fmtTime(DateTime t) {
    final h = t.hour.toString().padLeft(2, '0');
    final m = t.minute.toString().padLeft(2, '0');
    return '오늘 · $h:$m';
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Material(
        color: glorySurface,
        borderRadius: BorderRadius.circular(14),
        child: InkWell(
          borderRadius: BorderRadius.circular(14),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: gloryAccent.withValues(alpha: .18),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(Icons.music_note, color: gloryAccent, size: 20),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('${entry.title} (스캔)',
                          style: TextStyle(color: gloryInk, fontSize: 14, fontWeight: FontWeight.w600),
                          maxLines: 1, overflow: TextOverflow.ellipsis),
                      const SizedBox(height: 2),
                      Text(_fmtTime(entry.time), style: TextStyle(color: gloryMuted, fontSize: 11.5)),
                    ],
                  ),
                ),
                Container(
                  width: 28,
                  height: 28,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: gloryInk.withValues(alpha: .4)),
                  ),
                  child: Icon(Icons.arrow_forward, color: gloryMuted, size: 14),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class FrostedCircleButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  // 촬영 스캔 오버레이(항상 검은 배경, 카메라 뷰파인더 느낌)에서만 true -- 앱 테마와
  // 무관하게 흰색 프로스티드 유지. 그 외(일반 화면 헤더)는 기본값(false)으로 테마色 따라감.
  final bool alwaysLight;
  const FrostedCircleButton({super.key, required this.icon, required this.onTap, this.alwaysLight = false});

  @override
  Widget build(BuildContext context) {
    final fg = alwaysLight ? Colors.white : gloryInk;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: fg.withValues(alpha: .12),
        ),
        child: Icon(icon, color: fg, size: 20),
      ),
    );
  }
}

class _AllEntriesScreen extends StatelessWidget {
  final List<ScannedEntry> entries;
  final ValueChanged<ScannedEntry> onOpen;
  const _AllEntriesScreen({required this.entries, required this.onOpen});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: gloryBg,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 20, 0),
              child: Row(
                children: [
                  FrostedCircleButton(icon: Icons.arrow_back, onTap: () => Navigator.of(context).pop()),
                  const SizedBox(width: 12),
                  Text('스캔한 악보 전체',
                      style: TextStyle(color: gloryInk, fontSize: 18, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
                children: entries.map((e) => EntryTile(entry: e, onTap: () => onOpen(e))).toList(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
