import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

import '../omr_service.dart';

/// round3train/tflite_export/*의 임시 체크포인트(pubspec.yaml assets로 번들됨)를
/// 앱 저장소로 복사하고 OmrService를 초기화한다.
///
/// 네이티브 FFI는 파일시스템 경로만 받을 수 있어(에셋 번들 안을 직접 못 읽음),
/// 최초 1회 asset -> 디스크 복사가 필요하다(CLAUDE.md "Known Gaps" 참고).
/// 복사하면서 lib/omr_service.dart:107-110이 기대하는 plain 파일명으로 리네임
/// 한다 -- round3train/tflite_export/ 안의 실제 파일명(*_INT8.tflite)은 건드리지
/// 않으므로, 나중에 더 나은 체크포인트로 교체할 땐 그 폴더의 파일만 같은
/// 이름으로 덮어쓰면 된다.
class OmrModelLoader {
  OmrModelLoader._();

  static bool _initialized = false;
  static bool _initSucceeded = false;

  static const Map<String, String> _assetToPlainName = {
    'round3train/tflite_export/segnet_INT8.tflite': 'segnet.tflite',
    'round3train/tflite_export/encoder_INT8.tflite': 'encoder.tflite',
    'round3train/tflite_export/decoder_INT8.tflite': 'decoder.tflite',
    'round3train/tflite_export/tokenizer.json': 'tokenizer.json',
  };

  /// 최초 1회만 실제 복사 + init을 수행하고, 이후 호출은 캐시된 결과를 즉시 반환.
  static Future<bool> ensureInitialized() async {
    if (_initialized) return _initSucceeded;
    _initialized = true;
    try {
      final supportDir = await getApplicationSupportDirectory();
      final modelDir = Directory('${supportDir.path}/omr_models');
      if (!await modelDir.exists()) {
        await modelDir.create(recursive: true);
      }

      for (final entry in _assetToPlainName.entries) {
        final data = await rootBundle.load(entry.key);
        final outFile = File('${modelDir.path}/${entry.value}');
        // 이미 같은 크기로 복사돼 있으면 건너뜀 -- 모델이 총 ~225MB라 매 실행마다
        // 다시 쓰면 느리다.
        if (await outFile.exists() && await outFile.length() == data.lengthInBytes) {
          continue;
        }
        await outFile.writeAsBytes(
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
        );
      }

      _initSucceeded = OmrService.instance.init(modelDir.path);
    } catch (_) {
      _initSucceeded = false;
    }
    return _initSucceeded;
  }
}
