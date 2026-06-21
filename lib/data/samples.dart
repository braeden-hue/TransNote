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

  const ScoreNote({
    required this.pitch,
    required this.duration,
    required this.beat,
    this.chordNotes = const [],
    this.isRest = false,
    this.dynamicMark,
    this.repeatMark,
    this.hairpin,
  });
}

class SampleScore {
  final String id, title, emoji;
  final int tempo;
  final List<int> timeSignature;
  final List<ScoreNote> notes;
  const SampleScore({
    required this.id,
    required this.title,
    required this.emoji,
    required this.tempo,
    required this.timeSignature,
    required this.notes,
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
];
