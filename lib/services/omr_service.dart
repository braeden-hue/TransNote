import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import '../data/samples.dart';

/// server.py를 돌리는 PC의 LAN IP:포트로 바꿔주세요 (기본값은 이 프로젝트를 만든
/// PC 기준 — 다른 PC에서 서버를 켰다면 그 PC의 IP로 바꿔야 폰에서 연결됩니다).
/// 확인 방법: 서버 PC에서 `python server.py` 실행 후 콘솔에 뜨는 주소, 또는
/// Windows는 `ipconfig`(IPv4 주소), Mac/Linux는 `ifconfig`/`ip addr`.
const String kOmrServerUrl = 'http://192.168.219.103:8080';

/// 인식 실패(서버 응답 4xx/5xx, 네트워크 오류, 오선 인식 실패 등)를 사용자에게
/// 보여줄 메시지와 함께 표현한다.
class OmrException implements Exception {
  final String message;
  OmrException(this.message);

  @override
  String toString() => message;
}

/// server.py의 `POST /api/recognize`를 호출해 촬영한 악보 이미지를 커스텀 악보
/// (Sample)로 변환한다. 응답 JSON 스키마는 token_to_notes.py / server.py 참고 —
/// Sample.fromRecognizeJson()이 그대로 매핑한다.
class OmrService {
  final String baseUrl;
  final Duration timeout;

  const OmrService({this.baseUrl = kOmrServerUrl, this.timeout = const Duration(seconds: 30)});

  Future<Sample> recognize(Uint8List imageBytes, {String filename = 'score.jpg'}) async {
    final uri = Uri.parse('$baseUrl/api/recognize?model=custom');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes('file', imageBytes, filename: filename));

    final http.StreamedResponse streamed;
    try {
      streamed = await request.send().timeout(timeout);
    } catch (e) {
      throw OmrException('서버에 연결할 수 없어요 — server.py가 켜져 있는지, 같은 Wi-Fi인지 확인해주세요.');
    }

    final response = await http.Response.fromStream(streamed);
    if (response.statusCode != 200) {
      throw OmrException(_extractDetail(response.body) ?? '인식에 실패했어요 (HTTP ${response.statusCode})');
    }

    final json = jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    return Sample.fromRecognizeJson(json);
  }

  /// FastAPI의 HTTPException은 {"detail": "메시지"} 형태로 온다.
  String? _extractDetail(String body) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map && decoded['detail'] is String) return decoded['detail'] as String;
    } catch (_) {
      // 본문이 JSON이 아니면 무시하고 기본 메시지를 쓴다.
    }
    return null;
  }
}
