import 'dart:math' as math;

const Map<int, int> beatColorValues = {
  1: 0xFFFF6B35,
  2: 0xFF7BC67E,
  3: 0xFF5BC0EB,
  4: 0xFFC97FD6,
};

const _noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const _blackLabel = {'C#': '1', 'D#': '2', 'F#': '3', 'G#': '4', 'A#': '5'};

class ScoreNote {
  final String pitch;       // 쉼표일 때는 ''
  final double duration;
  final int beat;
  final List<String> chordNotes;
  final bool isRest;
  final String? dynamicMark;  // 'ppp'~'fff' | 'fp' | 'sf' | 'sfz'
  final String? repeatMark;   // 'end-repeat' → '반복' / 'start-repeat' → '여기부터'
  final String? hairpin;      // 'cresc' → '점점 세게' / 'dim' → '점점 약하게'
  final String? clef;         // 'treble' | 'bass' — 마디 중간 클렙 전환(오버라이드). null이면 보표 clef 상속

  const ScoreNote({
    required this.pitch,
    required this.duration,
    required this.beat,
    this.chordNotes = const [],
    this.isRest = false,
    this.dynamicMark,
    this.repeatMark,
    this.hairpin,
    this.clef,
  });
}

/// note의 유효 클렙(effective clef) = note.clef 오버라이드가 있으면 그 값, 없으면 보표 clef 상속.
String effectiveClef(ScoreNote note, String staffClef) => note.clef ?? staffClef;

/// note 리스트 안에 서로 다른 유효 클렙이 2개 이상 섞여 있는지 판정.
/// true일 때만 "마디 중간 클렙 전환" 배경 틴트를 그린다 (섞여 있지 않으면 기존 렌더링과 동일해야 함).
bool hasMixedClef(List<ScoreNote> notes, String staffClef) {
  String? first;
  for (final n in notes) {
    final c = effectiveClef(n, staffClef);
    first ??= c;
    if (c != first) return true;
  }
  return false;
}

class SampleScore {
  final String id, title, emoji;
  final int tempo;
  final List<int> timeSignature;
  final List<ScoreNote> notes;
  final List<ScoreNote>? bassNotes; // 대보표(그랜드 스태프) 샘플만 채움 -- 총 duration이 notes와 같아야 함(같은 타임라인 공유)
  const SampleScore({
    required this.id,
    required this.title,
    required this.emoji,
    required this.tempo,
    required this.timeSignature,
    required this.notes,
    this.bassNotes,
  });
}

String formatNoteName(String pitch) {
  final name = pitch.substring(0, pitch.length - 1);
  return _blackLabel[name] ?? name;
}

/// 높은음자리표: 4옥(가온다 포함) = 제일 아래(2), 5옥 = 중간(1), 6옥+ = 위(0)
/// 낮은음자리표: 3옥(가온다-1~B3) = 제일 위(0), 2옥 = 중간(1), 1옥 이하 = 아래(2)
int pitchToZone(String pitch, {String clef = 'treble'}) {
  final oct = int.parse(pitch[pitch.length - 1]);
  if (clef == 'bass') {
    if (oct >= 3) return 0;
    if (oct == 2) return 1;
    return 2;
  }
  if (oct >= 6) return 0;
  if (oct == 5) return 1;
  return 2;
}

List<String> zoneLabels(String clef) {
  if (clef == 'bass') return ['3옥+', '2옥', '1옥↓'];
  return ['6옥+', '5옥', '4옥'];
}

const _solfegeLabels = {
  'C': '도', 'D': '레', 'E': '미', 'F': '파',
  'G': '솔', 'A': '라', 'B': '시',
  'C#': '도#', 'D#': '레#', 'F#': '파#', 'G#': '솔#', 'A#': '라#',
};

/// 'C#4' -> '도#' 같은 계이름 라벨(옥타브 제외).
String solfegeLabel(String pitch) {
  final name = pitch.substring(0, pitch.length - 1);
  return _solfegeLabels[name] ?? name;
}

/// 마디(박자 리셋 기준)별로 가로 폭을 균등 분배하고, 마디 안에서는 음표 개수만큼 다시
/// 균등 분배해 각 음표의 근사 x좌표(0~1 비율)를 계산한다. 실제 촬영 원본 이미지에는 음표별
/// 정확한 픽셀 좌표 정보가 없으므로(모델이 좌표를 출력하지 않음) 이 근사치로 탭 위치를 매칭한다.
List<double> approximateNotePositions(List<ScoreNote> notes) {
  if (notes.isEmpty) return [];
  final measures = <List<int>>[];
  var current = <int>[];
  int? lastBeat;
  for (var i = 0; i < notes.length; i++) {
    final b = notes[i].beat;
    if (lastBeat != null && b <= lastBeat) {
      measures.add(current);
      current = [];
    }
    current.add(i);
    lastBeat = b;
  }
  if (current.isNotEmpty) measures.add(current);

  final positions = List<double>.filled(notes.length, 0);
  final measureW = 1.0 / measures.length;
  for (var m = 0; m < measures.length; m++) {
    final idxs = measures[m];
    final noteW = measureW / idxs.length;
    for (var k = 0; k < idxs.length; k++) {
      positions[idxs[k]] = m * measureW + (k + 0.5) * noteW;
    }
  }
  return positions;
}

double noteToFrequency(String pitch) {
  final name = pitch.substring(0, pitch.length - 1);
  final oct = int.parse(pitch[pitch.length - 1]);
  final semi = _noteNames.indexOf(name);
  return 440 * math.pow(2, (semi + (oct - 4) * 12 - 9) / 12).toDouble();
}

const samples = [
  SampleScore(
    id: 'twinkle', title: '반짝반짝 작은별', emoji: '⭐',
    tempo: 110, timeSignature: [4, 4],
    notes: [
      ScoreNote(pitch: 'C4', duration: 1, beat: 1),
      ScoreNote(pitch: 'C4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 1, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 4),
      ScoreNote(pitch: 'A4', duration: 1, beat: 1),
      ScoreNote(pitch: 'A4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 2, beat: 3),
      ScoreNote(pitch: 'F4', duration: 1, beat: 1),
      ScoreNote(pitch: 'F4', duration: 1, beat: 2),
      ScoreNote(pitch: 'E4', duration: 1, beat: 3),
      ScoreNote(pitch: 'E4', duration: 1, beat: 4),
      ScoreNote(pitch: 'D4', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'C4', duration: 2, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'F4', duration: 1, beat: 3),
      ScoreNote(pitch: 'F4', duration: 1, beat: 4),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'D4', duration: 2, beat: 3),
    ],
    // 마디당 1개(duration 4) 화성 반주 -- 대보표 2줄 렌더링 데모용. 총 duration 24로 treble과 동일.
    bassNotes: [
      ScoreNote(pitch: 'C3', duration: 4, beat: 1),
      ScoreNote(pitch: 'F3', duration: 4, beat: 1),
      ScoreNote(pitch: 'C3', duration: 4, beat: 1),
      ScoreNote(pitch: 'G3', duration: 4, beat: 1),
      ScoreNote(pitch: 'C3', duration: 4, beat: 1),
      ScoreNote(pitch: 'G3', duration: 4, beat: 1),
    ],
  ),
  SampleScore(
    id: 'school', title: '학교종이 땡땡땡', emoji: '🔔',
    tempo: 100, timeSignature: [4, 4],
    notes: [
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'A4', duration: 1, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 4),
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 2, beat: 3),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'E4', duration: 2, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 2, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'A4', duration: 1, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 4),
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 2, beat: 3),
    ],
  ),
  SampleScore(
    id: 'butterfly', title: '나비야 나비야', emoji: '🦋',
    tempo: 95, timeSignature: [3, 4],
    notes: [
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 1, beat: 3),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 1, beat: 3),
      ScoreNote(pitch: 'F4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'D4', duration: 1, beat: 3),
      ScoreNote(pitch: 'C4', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'C4', duration: 2, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 1, beat: 3),
      ScoreNote(pitch: 'A4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'F4', duration: 1, beat: 3),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'C4', duration: 2, beat: 3),
    ],
  ),
  SampleScore(
    id: 'ode', title: '기쁨의 송가 (베토벤)', emoji: '🎼',
    tempo: 120, timeSignature: [4, 4],
    notes: [
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'F4', duration: 1, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 4),
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'F4', duration: 1, beat: 2),
      ScoreNote(pitch: 'E4', duration: 1, beat: 3),
      ScoreNote(pitch: 'D4', duration: 1, beat: 4),
      ScoreNote(pitch: 'C4', duration: 1, beat: 1),
      ScoreNote(pitch: 'C4', duration: 1, beat: 2),
      ScoreNote(pitch: 'D4', duration: 1, beat: 3),
      ScoreNote(pitch: 'E4', duration: 1, beat: 4),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'D4', duration: 2, beat: 3),
    ],
  ),
  SampleScore(
    id: 'doremi', title: '도레미송', emoji: '🎵',
    tempo: 108, timeSignature: [4, 4],
    notes: [
      ScoreNote(pitch: 'C4', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'E4', duration: 1, beat: 3),
      ScoreNote(pitch: 'C4', duration: 1, beat: 4),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'C4', duration: 1, beat: 2),
      ScoreNote(pitch: 'E4', duration: 2, beat: 3),
      ScoreNote(pitch: 'D4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'F4', duration: 1, beat: 3),
      ScoreNote(pitch: 'F4', duration: 1, beat: 4),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'F4', duration: 2, beat: 3),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'F4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 1, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 4),
      ScoreNote(pitch: 'F4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 2, beat: 3),
    ],
  ),
  // 왼손(반주)+오른손(멜로디)을 함께 쓰는 예시 -- 멜로디는 검증된 'butterfly'와 동일(재사용),
  // 왼손은 3/4박자에 맞춘 왈츠 반주(근음+5음+5음, "쿵-짝-짝") 패턴으로 새로 작성.
  // 총 duration은 멜로디(23)와 반드시 일치해야 함(AutoPlayer가 같은 타임라인을 공유).
  SampleScore(
    id: 'butterfly_hands', title: '나비야 나비야 (두 손 반주)', emoji: '🦋',
    tempo: 95, timeSignature: [3, 4],
    notes: [
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 1, beat: 3),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 1, beat: 3),
      ScoreNote(pitch: 'F4', duration: 1, beat: 1),
      ScoreNote(pitch: 'E4', duration: 1, beat: 2),
      ScoreNote(pitch: 'D4', duration: 1, beat: 3),
      ScoreNote(pitch: 'C4', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'C4', duration: 2, beat: 3),
      ScoreNote(pitch: 'G4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'G4', duration: 1, beat: 3),
      ScoreNote(pitch: 'A4', duration: 1, beat: 1),
      ScoreNote(pitch: 'G4', duration: 1, beat: 2),
      ScoreNote(pitch: 'F4', duration: 1, beat: 3),
      ScoreNote(pitch: 'E4', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'C4', duration: 2, beat: 3),
    ],
    bassNotes: [
      ScoreNote(pitch: 'C3', duration: 1, beat: 1),
      ScoreNote(pitch: 'G3', duration: 1, beat: 2),
      ScoreNote(pitch: 'G3', duration: 1, beat: 3),
      ScoreNote(pitch: 'C3', duration: 1, beat: 1),
      ScoreNote(pitch: 'G3', duration: 1, beat: 2),
      ScoreNote(pitch: 'G3', duration: 1, beat: 3),
      ScoreNote(pitch: 'F3', duration: 1, beat: 1),
      ScoreNote(pitch: 'C4', duration: 1, beat: 2),
      ScoreNote(pitch: 'C4', duration: 1, beat: 3),
      ScoreNote(pitch: 'C3', duration: 1, beat: 1),
      ScoreNote(pitch: 'G3', duration: 1, beat: 2),
      ScoreNote(pitch: 'G3', duration: 1, beat: 3),
      ScoreNote(pitch: 'G3', duration: 1, beat: 1),
      ScoreNote(pitch: 'D4', duration: 1, beat: 2),
      ScoreNote(pitch: 'D4', duration: 1, beat: 3),
      ScoreNote(pitch: 'F3', duration: 1, beat: 1),
      ScoreNote(pitch: 'C4', duration: 1, beat: 2),
      ScoreNote(pitch: 'C4', duration: 1, beat: 3),
      ScoreNote(pitch: 'C3', duration: 1, beat: 1),
      ScoreNote(pitch: 'G3', duration: 1, beat: 2),
      ScoreNote(pitch: 'G3', duration: 3, beat: 3),
    ],
  ),
];
