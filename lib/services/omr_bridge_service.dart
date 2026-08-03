import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:xml/xml.dart' as xml;

import '../data/samples.dart';

/// homr(AGPL-3.0) 실제 가중치로 돌아가는 로컬 브리지 서버(omr_bridge/server.py) 클라이언트.
/// 절대 배포 빌드에 남기면 안 됨 -- 개발자 PC에서 서버를 띄워 로컬 테스트할 때만 쓰는
/// 코드다(AGPL-3.0 + 상업 배포 라이선스 충돌, project.md/CLAUDE.md 참고).
/// 서버가 꺼져 있으면(기본 상태) 연결이 바로 실패하고 null을 반환해서
/// CollectionScreen이 자동으로 데모 대체 결과로 폴백한다.
class RecognizedScore {
  final List<ScoreNote> treble;
  final List<ScoreNote> bass;
  final List<int> timeSignature;
  RecognizedScore({required this.treble, required this.bass, required this.timeSignature});
}

class OmrBridgeService {
  // 개발자 PC에서 omr_bridge/server.py를 띄웠을 때의 주소. Chrome(Flutter Web)으로 같은
  // PC에서 테스트할 때는 localhost로 충분 -- 폰에서 테스트하려면 PC의 LAN IP로 바꿔야
  // 한다(같은 Wi-Fi에 붙어 있어야 함, PowerShell `Get-NetIPAddress`로 재확인).
  static const String baseUrl = 'http://127.0.0.1:8008';
  static const Duration _timeout = Duration(seconds: 90);

  static Future<RecognizedScore?> recognize(Uint8List photo) async {
    try {
      final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/recognize'))
        ..files.add(http.MultipartFile.fromBytes('file', photo, filename: 'score.jpg'));
      final streamed = await request.send().timeout(_timeout);
      final response = await http.Response.fromStream(streamed);
      if (response.statusCode != 200) return null;
      return _parseMusicXml(response.body);
    } catch (_) {
      // 서버 미실행/네트워크 오류 -- 정상적인 폴백 경로라 조용히 null만 반환.
      return null;
    }
  }

  static const _stepSemitone = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11};
  static const _sharpNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  // MusicXML은 flat(예: Bb)도 그대로 쓰지만 우리 앱 음이름 표기는 sharp만 지원(samples.dart
  // _noteNames 참고) -- step+alter를 반음 계산해서 항상 sharp 철자로 변환한다.
  static String _toSharpPitch(String step, int alter, int octave) {
    var semitone = (_stepSemitone[step] ?? 0) + alter;
    var oct = octave;
    if (semitone < 0) {
      semitone += 12;
      oct -= 1;
    } else if (semitone > 11) {
      semitone -= 12;
      oct += 1;
    }
    return '${_sharpNames[semitone]}$oct';
  }

  static xml.XmlElement? _firstChild(xml.XmlElement el, String name) {
    final found = el.findElements(name);
    return found.isEmpty ? null : found.first;
  }

  static RecognizedScore? _parseMusicXml(String xmlStr) {
    final doc = xml.XmlDocument.parse(xmlStr);
    final parts = doc.findAllElements('part');
    if (parts.isEmpty) return null;
    final part = parts.first;

    var divisions = 4;
    var timeBeats = 4, timeBeatType = 4;
    final treble = <ScoreNote>[];
    final bass = <ScoreNote>[];

    for (final measure in part.findElements('measure')) {
      var trebleBeat = 1;
      var bassBeat = 1;

      for (final el in measure.childElements) {
        if (el.name.local == 'attributes') {
          final divEl = _firstChild(el, 'divisions');
          if (divEl != null) divisions = int.tryParse(divEl.innerText) ?? divisions;
          final timeEl = _firstChild(el, 'time');
          if (timeEl != null) {
            timeBeats = int.tryParse(_firstChild(timeEl, 'beats')?.innerText ?? '') ?? timeBeats;
            timeBeatType = int.tryParse(_firstChild(timeEl, 'beat-type')?.innerText ?? '') ?? timeBeatType;
          }
        } else if (el.name.local == 'note') {
          final isChordTone = el.findElements('chord').isNotEmpty;
          final isRest = el.findElements('rest').isNotEmpty;
          final durationUnits = int.tryParse(_firstChild(el, 'duration')?.innerText ?? '') ?? divisions;
          final durationQuarters = divisions == 0 ? 1.0 : durationUnits / divisions;
          final staff = int.tryParse(_firstChild(el, 'staff')?.innerText ?? '1') ?? 1;

          String pitch = '';
          if (!isRest) {
            final pitchEl = _firstChild(el, 'pitch');
            if (pitchEl == null) continue; // 손상된 음표 -- 건너뜀
            final step = _firstChild(pitchEl, 'step')?.innerText ?? 'C';
            final alter = int.tryParse(_firstChild(pitchEl, 'alter')?.innerText ?? '0') ?? 0;
            final octave = int.tryParse(_firstChild(pitchEl, 'octave')?.innerText ?? '4') ?? 4;
            pitch = _toSharpPitch(step, alter, octave);
          }

          final target = staff == 2 ? bass : treble;
          if (isChordTone && !isRest && target.isNotEmpty) {
            // 같은 타이밍에 울리는 화음 -- 새 슬롯을 만들지 않고 직전 음표의
            // chordNotes에 얹는다(ScoreNote 모델 참고).
            final last = target.removeLast();
            target.add(ScoreNote(
              pitch: last.pitch,
              duration: last.duration,
              beat: last.beat,
              chordNotes: [...last.chordNotes, pitch],
              isRest: last.isRest,
            ));
            continue;
          }

          final beat = staff == 2 ? bassBeat : trebleBeat;
          target.add(ScoreNote(pitch: pitch, duration: durationQuarters, beat: beat, isRest: isRest));
          final advance = durationQuarters.round().clamp(1, 4);
          if (staff == 2) {
            bassBeat = (bassBeat - 1 + advance) % 4 + 1;
          } else {
            trebleBeat = (trebleBeat - 1 + advance) % 4 + 1;
          }
        }
      }
    }

    if (treble.isEmpty) return null;
    return RecognizedScore(treble: treble, bass: bass, timeSignature: [timeBeats, timeBeatType]);
  }
}
