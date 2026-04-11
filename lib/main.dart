import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'omr_service.dart';

void main() => runApp(const MusicScoreApp());

class MusicScoreApp extends StatelessWidget {
  const MusicScoreApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MusicScore OMR',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const OmrHomePage(),
    );
  }
}

class OmrHomePage extends StatefulWidget {
  const OmrHomePage({super.key});

  @override
  State<OmrHomePage> createState() => _OmrHomePageState();
}

class _OmrHomePageState extends State<OmrHomePage> {
  String _status = '초기화 전';
  String _result = '';
  bool _loading = false;
  final _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _initPipeline();
  }

  Future<void> _initPipeline() async {
    setState(() => _status = '초기화 중...');
    final ok = await OmrService.initialize();
    setState(() => _status = ok ? 'OMR 파이프라인 준비됨' : '초기화 실패');
  }

  Future<void> _pickAndProcess() async {
    final file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;

    setState(() {
      _loading = true;
      _result = '';
    });

    try {
      final bytes = await file.readAsBytes();
      final xml = await OmrService.processImageBytes(Uint8List.fromList(bytes));
      setState(() => _result = xml ?? '처리 실패: 악보를 인식하지 못했습니다.');
    } catch (e) {
      setState(() => _result = '오류: $e');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('MusicScore OMR')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text('상태: $_status',
                    style: Theme.of(context).textTheme.bodyLarge),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _loading ? null : _pickAndProcess,
              icon: _loading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.music_note),
              label: const Text('악보 이미지 선택 → MusicXML 변환'),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(8),
                ),
                padding: const EdgeInsets.all(8),
                child: SingleChildScrollView(
                  child: Text(
                    _result.isEmpty ? '결과가 여기 표시됩니다.' : _result,
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
