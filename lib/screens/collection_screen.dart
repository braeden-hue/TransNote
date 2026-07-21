import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../data/samples.dart';
import '../theme/glory_theme.dart';
import '../widgets/notation_widget.dart';

// QuickScan UI 킷을 참고한 이 화면 전용 다크 팔레트 (앱 나머지는 라이트 테마 유지).
const _dkBg = Color(0xFF121218);
const _dkCard = Color(0xFF1C1C26);
const _dkAccent = Color(0xFF5B5FEF);
const _dkMuted = Color(0xFF9A9AB0);

class _ScannedEntry {
  final String title;
  final DateTime time;
  final SampleScore preview;
  _ScannedEntry(this.title, this.time, this.preview);
}

class CollectionScreen extends StatefulWidget {
  const CollectionScreen({super.key});

  @override
  State<CollectionScreen> createState() => _CollectionScreenState();
}

class _CollectionScreenState extends State<CollectionScreen> with SingleTickerProviderStateMixin {
  final List<_ScannedEntry> _entries = [];
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

  Future<void> _capture(ImageSource source) async {
    setState(() => _error = null);
    try {
      final file = await ImagePicker().pickImage(source: source, imageQuality: 85);
      if (file == null) return;
      final bytes = await file.readAsBytes();
      setState(() {
        _photo = bytes;
        _capturing = true;
      });
      await _scanCtrl.forward(from: 0);
      if (!mounted) return;
      final sample = samples[_entries.length % samples.length];
      setState(() {
        _entries.insert(0, _ScannedEntry(sample.title, DateTime.now(), sample));
        _capturing = false;
        _photo = null;
      });
    } catch (e) {
      setState(() {
        _capturing = false;
        _error = '카메라를 사용할 수 없어요: $e';
      });
    }
  }

  void _openEntry(_ScannedEntry e) {
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => _EntryDetailScreen(entry: e)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _dkBg,
      body: Stack(
        children: [
          SafeArea(child: _buildBody(context)),
          if (!_capturing)
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: SafeArea(top: false, child: _buildActionRow()),
            ),
          if (_capturing) _buildScanOverlay(),
        ],
      ),
    );
  }

  Widget _buildActionRow() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20, top: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          _ImportBadge(
            icon: Icons.photo_library_outlined,
            color: const Color(0xFF3EA06B),
            label: '갤러리',
            onTap: () => _capture(ImageSource.gallery),
          ),
          const SizedBox(width: 40),
          GestureDetector(
            onTap: () => _capture(ImageSource.camera),
            child: Container(
              width: 68,
              height: 68,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _dkAccent,
                boxShadow: [BoxShadow(color: _dkAccent.withValues(alpha: .5), blurRadius: 20, spreadRadius: 2)],
              ),
              child: const Icon(Icons.camera_alt, color: Colors.white, size: 28),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBody(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 20, 0),
          child: Row(
            children: [
              _FrostedCircleButton(icon: Icons.arrow_back, onTap: () => Navigator.of(context).pop()),
              const SizedBox(width: 12),
              const Text('악보 모음집',
                  style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
            ],
          ),
        ),
        Expanded(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 160),
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: _dkCard,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('악보를 촬영해서\n바로 정리해보세요',
                              style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold, height: 1.3)),
                          const SizedBox(height: 6),
                          Text('촬영한 사진은 커스텀 악보로 자동 변환돼요.',
                              style: TextStyle(color: _dkMuted, fontSize: 12)),
                        ],
                      ),
                    ),
                    Icon(Icons.description_outlined, color: _dkAccent.withValues(alpha: .7), size: 44),
                  ],
                ),
              ),
              if (_error != null) ...[
                const SizedBox(height: 12),
                Text(_error!, style: const TextStyle(color: Color(0xFFEE7777), fontSize: 12)),
              ],
              const SizedBox(height: 24),
              const Text('최근 스캔',
                  style: TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold)),
              const SizedBox(height: 12),
              if (_entries.isEmpty)
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(color: _dkCard, borderRadius: BorderRadius.circular(16)),
                  child: Column(
                    children: [
                      Icon(Icons.camera_alt_outlined, color: _dkMuted, size: 28),
                      const SizedBox(height: 10),
                      Text('아직 스캔한 악보가 없어요.\n아래 카메라 버튼으로 촬영해보세요.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: _dkMuted, fontSize: 12.5, height: 1.4)),
                    ],
                  ),
                )
              else
                ..._entries.map((e) => _EntryTile(entry: e, onTap: () => _openEntry(e))),
            ],
          ),
        ),
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
                              color: _dkAccent,
                              boxShadow: [BoxShadow(color: _dkAccent.withValues(alpha: .8), blurRadius: 12, spreadRadius: 1)],
                            ),
                          ),
                        ),
                      ),
                      Positioned(
                        top: 12,
                        left: 12,
                        right: 12,
                        child: Row(
                          children: [
                            _FrostedCircleButton(
                              icon: Icons.close,
                              onTap: () {
                                _scanCtrl.stop();
                                setState(() {
                                  _capturing = false;
                                  _photo = null;
                                });
                              },
                            ),
                            Expanded(
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
                            const SizedBox(width: 40),
                          ],
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

class _EntryTile extends StatelessWidget {
  final _ScannedEntry entry;
  final VoidCallback onTap;
  const _EntryTile({required this.entry, required this.onTap});

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
        color: _dkCard,
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
                    color: _dkAccent.withValues(alpha: .18),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(Icons.music_note, color: _dkAccent, size: 20),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('${entry.title} (스캔)',
                          style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
                          maxLines: 1, overflow: TextOverflow.ellipsis),
                      const SizedBox(height: 2),
                      Text(_fmtTime(entry.time), style: TextStyle(color: _dkMuted, fontSize: 11.5)),
                    ],
                  ),
                ),
                Container(
                  width: 28,
                  height: 28,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: _dkMuted.withValues(alpha: .4)),
                  ),
                  child: Icon(Icons.arrow_forward, color: _dkMuted, size: 14),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _ImportBadge extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String label;
  final VoidCallback onTap;
  const _ImportBadge({required this.icon, required this.color, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: color.withValues(alpha: .18),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(height: 6),
          Text(label, style: TextStyle(color: _dkMuted, fontSize: 11)),
        ],
      ),
    );
  }
}

class _FrostedCircleButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  const _FrostedCircleButton({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: Colors.white.withValues(alpha: .12),
        ),
        child: Icon(icon, color: Colors.white, size: 20),
      ),
    );
  }
}

class _EntryDetailScreen extends StatelessWidget {
  final _ScannedEntry entry;
  const _EntryDetailScreen({required this.entry});

  @override
  Widget build(BuildContext context) {
    final s = entry.preview;
    return Scaffold(
      backgroundColor: gloryBg,
      appBar: AppBar(
        backgroundColor: gloryBg,
        elevation: 0,
        iconTheme: const IconThemeData(color: gloryInk),
        title: Text('${s.emoji} ${entry.title}', style: const TextStyle(color: gloryInk, fontWeight: FontWeight.w600)),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0xFF7BC67E).withAlpha(30),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: const Color(0xFF7BC67E), width: 1),
                ),
                child: const Text('인식 완료 (데모)',
                    style: TextStyle(color: Color(0xFF3E8F45), fontSize: 12, fontWeight: FontWeight.bold)),
              ),
              const SizedBox(height: 12),
              Text('모델 학습이 아직 진행 중이라, 지금은 촬영 사진 대신 샘플 악보로 결과를 보여드려요.',
                  style: TextStyle(color: gloryInk.withValues(alpha: .45), fontSize: 11.5, height: 1.4)),
              const SizedBox(height: 16),
              NotationWidget(notes: s.notes, timeSignature: s.timeSignature),
            ],
          ),
        ),
      ),
    );
  }
}
