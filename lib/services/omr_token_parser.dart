import '../data/samples.dart';
import '../omr_service.dart';
import 'omr_bridge_service.dart' show RecognizedScore;

/// round3train/mscz_to_tokens.py 토큰 문법에 따라 OmrService.process()가 반환한
/// 평평한 토큰 목록(SOS/EOS/PAD는 C++ TokenParser::decode_ids에서 이미 제거됨)을
/// ScoreNote 리스트로 바꾼다. OmrBridgeService.RecognizedScore와 같은 타입
/// (treble/bass/timeSignature)을 그대로 반환해서 collection_screen.dart의
/// 소비 코드를 최소한만 바꾸면 되게 한다.
///
/// 문법 요약 (round3train/mscz_to_tokens.py 기준):
///   clef-{G|F} key-{name} time-{n}/{d}
///   매 마디: [barline-start-repeat]?
///            (rest-{dur} | note-{pitch} dur-{dur} [chord-{pitch}]*)
///              [artic-*|fermata|ornament-*|tie]* 반복
///            [staff-bass [clef-F|clef-G]? <베이스 마디, 같은 문법>]?
///            barline_token (barline|barline-double|barline-final|barline-end-repeat)
///
/// clef-*/key-* 토큰 자체는 무시한다 -- treble/bass 배열 소속 자체가 clef를
/// 암시하는 기존 관례(OmrBridgeService._parseMusicXml의 staff==2 분기와 동일)를
/// 따른다. artic-*/fermata/ornament-*/tie/dynamic-*(값 없이 opportunistic하게만
/// 반영)도 1차 연결 검증 범위 밖이라 대부분 건너뛴다 -- note/dur/rest/chord/
/// barline/staff-bass만 정확히 처리하는 게 우선.
class OmrTokenParser {
  OmrTokenParser._();

  static RecognizedScore parse(List<OmrToken> tokens) {
    final treble = <ScoreNote>[];
    final bass = <ScoreNote>[];
    var timeSig = const [4, 4];

    var onBass = false;
    var trebleBeat = 1;
    var bassBeat = 1;
    String? startRepeatPending;

    // 조립 중인 note/chord -- dur-*, chord-* 토큰을 계속 흡수하다가 다음
    // 경계 토큰(note-/rest-/staff-bass/barline*)을 만나면 finalize한다.
    String? pendingPitch;
    double pendingDuration = 1.0;
    final pendingChord = <String>[];
    String? pendingDynamic;
    var hasPending = false;

    void push({
      required String pitch,
      required double duration,
      List<String> chord = const [],
      bool isRest = false,
      String? dynamicMark,
    }) {
      final target = onBass ? bass : treble;
      final beat = onBass ? bassBeat : trebleBeat;
      target.add(ScoreNote(
        pitch: pitch,
        duration: duration,
        beat: beat,
        chordNotes: chord,
        isRest: isRest,
        dynamicMark: dynamicMark,
        repeatMark: startRepeatPending,
      ));
      startRepeatPending = null;
      final advance = duration.round().clamp(1, 4);
      if (onBass) {
        bassBeat = (bassBeat - 1 + advance) % 4 + 1;
      } else {
        trebleBeat = (trebleBeat - 1 + advance) % 4 + 1;
      }
    }

    void finalizePending() {
      if (!hasPending) return;
      push(
        pitch: pendingPitch ?? '',
        duration: pendingDuration,
        chord: List.of(pendingChord),
        dynamicMark: pendingDynamic,
      );
      pendingPitch = null;
      pendingDuration = 1.0;
      pendingChord.clear();
      pendingDynamic = null;
      hasPending = false;
    }

    // barline-end-repeat -- 방금 추가된 노트(해당 스태프의 마지막 원소)에
    // repeatMark='end-repeat'를 소급 적용.
    void markEndRepeat() {
      final target = onBass ? bass : treble;
      if (target.isEmpty) return;
      final last = target.removeLast();
      target.add(ScoreNote(
        pitch: last.pitch,
        duration: last.duration,
        beat: last.beat,
        chordNotes: last.chordNotes,
        isRest: last.isRest,
        dynamicMark: last.dynamicMark,
        repeatMark: 'end-repeat',
        hairpin: last.hairpin,
        clef: last.clef,
      ));
    }

    double fractionToQuarters(String frac) {
      final parts = frac.split('/');
      if (parts.length != 2) return 1.0;
      final n = double.tryParse(parts[0]) ?? 1;
      final d = double.tryParse(parts[1]) ?? 4;
      if (d == 0) return 1.0;
      return n / d * 4.0; // 1.0 = 4분음표 (samples.dart ScoreNote.duration과 동일 단위)
    }

    for (final tok in tokens) {
      final t = tok.text;
      if (t.startsWith('time-')) {
        final parts = t.substring(5).split('/');
        if (parts.length == 2) {
          final n = int.tryParse(parts[0]);
          final d = int.tryParse(parts[1]);
          if (n != null && d != null) timeSig = [n, d];
        }
      } else if (t.startsWith('clef-') || t.startsWith('key-')) {
        // 무시 (위 클래스 주석 참고).
      } else if (t == 'staff-bass') {
        finalizePending();
        onBass = true;
      } else if (t.startsWith('note-')) {
        finalizePending();
        pendingPitch = t.substring(5);
        pendingDuration = 1.0;
        hasPending = true;
      } else if (t.startsWith('dur-')) {
        if (hasPending) pendingDuration = fractionToQuarters(t.substring(4));
      } else if (t.startsWith('chord-')) {
        if (hasPending) pendingChord.add(t.substring(6));
      } else if (t.startsWith('rest-')) {
        finalizePending();
        push(pitch: '', duration: fractionToQuarters(t.substring(5)), isRest: true);
      } else if (t.startsWith('dynamic-')) {
        if (hasPending) pendingDynamic = t.substring(8);
      } else if (t == 'barline-start-repeat') {
        finalizePending();
        startRepeatPending = 'start-repeat';
      } else if (t == 'barline' ||
          t == 'barline-double' ||
          t == 'barline-final' ||
          t == 'barline-end-repeat') {
        finalizePending();
        if (t == 'barline-end-repeat') markEndRepeat();
        // 마디 경계 -- 두 보표 모두 다음 마디는 박자 1부터, treble로 복귀.
        trebleBeat = 1;
        bassBeat = 1;
        onBass = false;
      }
      // artic-*/fermata/ornament-*/tie: 무시.
    }
    finalizePending();

    return RecognizedScore(treble: treble, bass: bass, timeSignature: timeSig);
  }
}
