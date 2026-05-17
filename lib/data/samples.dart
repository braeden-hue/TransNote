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
  final String pitch;
  final double duration;
  final int beat;
  const ScoreNote({required this.pitch, required this.duration, required this.beat});
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

int pitchToZone(String pitch) {
  final oct = int.parse(pitch[pitch.length - 1]);
  if (oct >= 5) return 0;
  if (oct == 4) return 1;
  return 2;
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
