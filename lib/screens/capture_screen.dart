import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../services/omr_service.dart';
import '../widgets/notation_widget.dart';
import '../widgets/piano_widget.dart';

/// 악보 촬영/갤러리 선택 → server.py `/api/recognize`로 실제 OMR 변환까지 담당하는 화면.
/// 서버가 안 켜져 있거나 연결이 안 되면 에러를 정직하게 보여주고 사진은 그대로 유지한다
/// (거짓으로 성공한 척하지 않음).
class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

enum _Stage { empty, recognizing, result, error }

class _CaptureScreenState extends State<CaptureScreen> {
  final _picker = ImagePicker();
  final _omr = const OmrService();
  final _pianoController = PianoController();

  Uint8List? _imageBytes;
  _Stage _stage = _Stage.empty;
  String? _pickError;
  String? _omrError;
  Sample? _result;

  @override
  void dispose() {
    _pianoController.dispose();
    super.dispose();
  }

  Future<void> _pick(ImageSource source) async {
    setState(() => _pickError = null);
    try {
      final file = await _picker.pickImage(source: source, imageQuality: 90);
      if (file == null) return;
      final bytes = await file.readAsBytes();
      if (!mounted) return;
      setState(() {
        _imageBytes = bytes;
        _result = null;
        _omrError = null;
      });
      await _recognize();
    } catch (e) {
      if (!mounted) return;
      setState(() => _pickError = '사진을 가져오지 못했어요. 카메라/갤러리 권한을 확인해주세요.');
    }
  }

  Future<void> _recognize() async {
    final bytes = _imageBytes;
    if (bytes == null) return;
    setState(() {
      _stage = _Stage.recognizing;
      _omrError = null;
    });
    try {
      final sample = await _omr.recognize(bytes);
      if (!mounted) return;
      setState(() {
        _result = sample;
        _stage = _Stage.result;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _omrError = e is OmrException ? e.message : '인식 중 알 수 없는 오류가 발생했어요.';
        _stage = _Stage.error;
      });
    }
  }

  void _reset() {
    setState(() {
      _imageBytes = null;
      _result = null;
      _omrError = null;
      _stage = _Stage.empty;
    });
  }

  void _playNote(String pitch) {
    audioService.playNote(pitch);
    _pianoController.flashCorrect(pitch);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('악보 촬영')),
      body: SafeArea(
        child: _imageBytes == null ? _buildEmpty(context) : _buildAfterPick(context),
      ),
    );
  }

  Widget _buildEmpty(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.camera_alt_outlined, size: 72, color: Theme.of(context).colorScheme.primary),
          const SizedBox(height: 16),
          const Text(
            '악보를 촬영하거나 갤러리에서 사진을 선택하세요',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16),
          ),
          const SizedBox(height: 32),
          FilledButton.icon(
            onPressed: () => _pick(ImageSource.camera),
            icon: const Icon(Icons.camera_alt),
            label: const Text('촬영하기'),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: () => _pick(ImageSource.gallery),
            icon: const Icon(Icons.photo_library_outlined),
            label: const Text('갤러리에서 선택'),
          ),
          if (_pickError != null)
            Padding(
              padding: const EdgeInsets.only(top: 16),
              child: Text(
                _pickError!,
                textAlign: TextAlign.center,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildAfterPick(BuildContext context) {
    if (_stage == _Stage.result && _result != null) {
      return _buildResult(context, _result!);
    }
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.memory(_imageBytes!, fit: BoxFit.contain, width: double.infinity),
            ),
          ),
          const SizedBox(height: 16),
          if (_stage == _Stage.recognizing) _buildRecognizingBanner(context),
          if (_stage == _Stage.error) _buildErrorBanner(context),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _reset,
                  icon: const Icon(Icons.refresh),
                  label: const Text('다시 찍기'),
                ),
              ),
              if (_stage == _Stage.error) ...[
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _recognize,
                    icon: const Icon(Icons.replay),
                    label: const Text('다시 시도'),
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRecognizingBanner(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.secondaryContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          const SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(strokeWidth: 2.5),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text('악보 인식 중이에요…', style: Theme.of(context).textTheme.bodyMedium),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorBanner(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.errorContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.error_outline, size: 20, color: Theme.of(context).colorScheme.onErrorContainer),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              _omrError ?? '인식에 실패했어요.',
              style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResult(BuildContext context, Sample sample) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
          child: Row(
            children: [
              const Icon(Icons.check_circle, color: Color(0xFF3DBE64)),
              const SizedBox(width: 8),
              Expanded(
                child: Text('변환 완료 — ${sample.title}',
                    style: Theme.of(context).textTheme.titleMedium),
              ),
              TextButton.icon(
                onPressed: _reset,
                icon: const Icon(Icons.refresh),
                label: const Text('다시 찍기'),
              ),
            ],
          ),
        ),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: sample.staves != null
                ? GrandStaffWidget(
                    staves: sample.staves!,
                    onNoteTap: (clef, i, note) {
                      if (note.isRest) return;
                      _playNote(note.pitch);
                    },
                  )
                : NotationWidget(
                    notes: sample.notes,
                    onNoteTap: (i, note) {
                      if (note.isRest) return;
                      _playNote(note.pitch);
                    },
                  ),
          ),
        ),
        PianoDock(
          child: PianoWidget(controller: _pianoController, onKeyPress: _playNote),
        ),
      ],
    );
  }
}
