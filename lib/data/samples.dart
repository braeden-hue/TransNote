// webpage/js/samples.js 포팅 — 규칙(음 이름 표기, 존 계산)은 원본과 100% 동일해야 함.

class NoteEvent {
  final String pitch; // 'C4', 'F#5' 등. 쉼표면 빈 문자열.
  final double duration; // quarterLength 배수 (1 = 4분음표)
  final double beat;
  final bool isRest;
  final List<String> chordNotes;
  final String? clef; // 마디 중간 클렙 전환 오버라이드

  const NoteEvent({
    required this.pitch,
    required this.duration,
    required this.beat,
    this.isRest = false,
    this.chordNotes = const [],
    this.clef,
  });

  bool get isChord => chordNotes.isNotEmpty;

  /// server.py `/api/recognize` 응답의 note 하나(token_to_notes.py의 ScoreNote와
  /// 필드명이 동일 — pitch/duration/beat/chordNotes/isRest). dynamicMark·repeatMark·
  /// tuplet은 아직 이 앱 모델에 없어서 무시한다(추후 확장 지점).
  factory NoteEvent.fromJson(Map<String, dynamic> json) {
    return NoteEvent(
      pitch: json['pitch'] as String? ?? '',
      duration: (json['duration'] as num?)?.toDouble() ?? 1,
      beat: (json['beat'] as num?)?.toDouble() ?? 1,
      isRest: json['isRest'] as bool? ?? false,
      chordNotes: (json['chordNotes'] as List?)?.map((e) => e as String).toList() ?? const [],
    );
  }
}

class Stave {
  final String clef; // 'treble' | 'bass'
  final List<NoteEvent> notes;

  const Stave({required this.clef, required this.notes});

  factory Stave.fromJson(Map<String, dynamic> json) {
    return Stave(
      clef: json['clef'] as String? ?? 'treble',
      notes: (json['notes'] as List? ?? const [])
          .map((n) => NoteEvent.fromJson(n as Map<String, dynamic>))
          .toList(),
    );
  }
}

class Sample {
  final String id;
  final String title;
  final String emoji;
  final int tempo;
  final List<int> timeSignature;
  final List<NoteEvent> notes; // 단일 보표 표기(오른손 기준)
  final List<Stave>? staves; // 대보표(양손) — null이면 단일 오선

  const Sample({
    required this.id,
    required this.title,
    required this.emoji,
    required this.tempo,
    required this.timeSignature,
    required this.notes,
    this.staves,
  });

  /// server.py `POST /api/recognize` 응답 JSON -> Sample.
  /// 응답 스키마: { id, title, tempo, timeSignature, notes, staves? }
  /// (staves가 있으면 notes는 staves[0]과 동일한 오른손 표기 — Sample.notes 규약과 맞음)
  factory Sample.fromRecognizeJson(Map<String, dynamic> json) {
    final stavesJson = json['staves'] as List?;
    return Sample(
      id: json['id'] as String? ?? 'recognized_${DateTime.now().millisecondsSinceEpoch}',
      title: json['title'] as String? ?? '촬영한 악보',
      emoji: '📷',
      tempo: (json['tempo'] as num?)?.toInt() ?? 100,
      timeSignature: (json['timeSignature'] as List?)?.map((e) => (e as num).toInt()).toList() ??
          const [4, 4],
      notes: (json['notes'] as List? ?? const [])
          .map((n) => NoteEvent.fromJson(n as Map<String, dynamic>))
          .toList(),
      staves: stavesJson?.map((s) => Stave.fromJson(s as Map<String, dynamic>)).toList(),
    );
  }
}

const List<Sample> samples = [
  Sample(
    id: 'sample_butterfly',
    title: '나비야 나비야',
    emoji: '🦋',
    tempo: 95,
    timeSignature: [4, 4],
    notes: [
      NoteEvent(pitch: 'G4', duration: 1, beat: 1),
      NoteEvent(pitch: 'E4', duration: 1, beat: 2),
      NoteEvent(pitch: 'E4', duration: 2, beat: 3),
      NoteEvent(pitch: 'F4', duration: 1, beat: 1),
      NoteEvent(pitch: 'D4', duration: 1, beat: 2),
      NoteEvent(pitch: 'D4', duration: 2, beat: 3),
      NoteEvent(pitch: 'C4', duration: 1, beat: 1),
      NoteEvent(pitch: 'D4', duration: 1, beat: 2),
      NoteEvent(pitch: 'E4', duration: 1, beat: 3),
      NoteEvent(pitch: 'F4', duration: 1, beat: 4),
      NoteEvent(pitch: 'G4', duration: 1, beat: 1),
      NoteEvent(pitch: 'G4', duration: 1, beat: 2),
      NoteEvent(pitch: 'G4', duration: 2, beat: 3),
      NoteEvent(pitch: 'G4', duration: 1, beat: 1),
      NoteEvent(pitch: 'E4', duration: 1, beat: 2),
      NoteEvent(pitch: 'E4', duration: 1, beat: 3),
      NoteEvent(pitch: 'E4', duration: 1, beat: 4),
      NoteEvent(pitch: 'F4', duration: 1, beat: 1),
      NoteEvent(pitch: 'D4', duration: 1, beat: 2),
      NoteEvent(pitch: 'D4', duration: 2, beat: 3),
      NoteEvent(pitch: 'C4', duration: 1, beat: 1),
      NoteEvent(pitch: 'E4', duration: 1, beat: 2),
      NoteEvent(pitch: 'G4', duration: 1, beat: 3),
      NoteEvent(pitch: 'G4', duration: 1, beat: 4),
      NoteEvent(pitch: 'E4', duration: 1, beat: 1),
      NoteEvent(pitch: 'E4', duration: 1, beat: 2),
      NoteEvent(pitch: 'E4', duration: 2, beat: 3),
    ],
    staves: [
      Stave(clef: 'treble', notes: [
        NoteEvent(pitch: 'G4', duration: 1, beat: 1),
        NoteEvent(pitch: 'E4', duration: 1, beat: 2),
        NoteEvent(pitch: 'E4', duration: 2, beat: 3),
        NoteEvent(pitch: 'F4', duration: 1, beat: 1),
        NoteEvent(pitch: 'D4', duration: 1, beat: 2),
        NoteEvent(pitch: 'D4', duration: 2, beat: 3),
        NoteEvent(pitch: 'C4', duration: 1, beat: 1),
        NoteEvent(pitch: 'D4', duration: 1, beat: 2),
        NoteEvent(pitch: 'E4', duration: 1, beat: 3),
        NoteEvent(pitch: 'F4', duration: 1, beat: 4),
        NoteEvent(pitch: 'G4', duration: 1, beat: 1),
        NoteEvent(pitch: 'G4', duration: 1, beat: 2),
        NoteEvent(pitch: 'G4', duration: 2, beat: 3),
        NoteEvent(pitch: 'G4', duration: 1, beat: 1),
        NoteEvent(pitch: 'E4', duration: 1, beat: 2),
        NoteEvent(pitch: 'E4', duration: 1, beat: 3),
        NoteEvent(pitch: 'E4', duration: 1, beat: 4),
        NoteEvent(pitch: 'F4', duration: 1, beat: 1),
        NoteEvent(pitch: 'D4', duration: 1, beat: 2),
        NoteEvent(pitch: 'D4', duration: 2, beat: 3),
        NoteEvent(pitch: 'C4', duration: 1, beat: 1),
        NoteEvent(pitch: 'E4', duration: 1, beat: 2),
        NoteEvent(pitch: 'G4', duration: 1, beat: 3),
        NoteEvent(pitch: 'G4', duration: 1, beat: 4),
        NoteEvent(pitch: 'E4', duration: 1, beat: 1),
        NoteEvent(pitch: 'E4', duration: 1, beat: 2),
        NoteEvent(pitch: 'E4', duration: 2, beat: 3),
      ]),
      Stave(clef: 'bass', notes: [
        NoteEvent(pitch: 'C3', duration: 2, beat: 1),
        NoteEvent(pitch: 'E3', duration: 2, beat: 3, chordNotes: ['G3']),
        NoteEvent(pitch: 'D3', duration: 2, beat: 1),
        NoteEvent(pitch: 'F3', duration: 2, beat: 3, chordNotes: ['G3']),
        NoteEvent(pitch: 'C3', duration: 1, beat: 1),
        NoteEvent(pitch: 'G3', duration: 1, beat: 2),
        NoteEvent(pitch: 'E3', duration: 1, beat: 3),
        NoteEvent(pitch: 'F3', duration: 1, beat: 4),
        NoteEvent(pitch: 'E3', duration: 1, beat: 1, chordNotes: ['G3']),
        NoteEvent(pitch: 'E3', duration: 1, beat: 2, chordNotes: ['G3']),
        NoteEvent(pitch: 'E3', duration: 1, beat: 3, chordNotes: ['G3']),
        NoteEvent(pitch: '', duration: 1, beat: 4, isRest: true),
        NoteEvent(pitch: 'C3', duration: 1, beat: 1),
        NoteEvent(pitch: 'G3', duration: 1, beat: 2),
        NoteEvent(pitch: 'E3', duration: 1, beat: 3),
        NoteEvent(pitch: 'G3', duration: 1, beat: 4),
        NoteEvent(pitch: 'D3', duration: 1, beat: 1),
        NoteEvent(pitch: 'G3', duration: 1, beat: 2),
        NoteEvent(pitch: 'F3', duration: 1, beat: 3),
        NoteEvent(pitch: 'G3', duration: 1, beat: 4),
        NoteEvent(pitch: 'C3', duration: 2, beat: 1),
        NoteEvent(pitch: 'E3', duration: 2, beat: 3, chordNotes: ['G3']),
        NoteEvent(pitch: 'E3', duration: 1, beat: 1, chordNotes: ['G3']),
        NoteEvent(pitch: 'E3', duration: 1, beat: 2, chordNotes: ['G3']),
        NoteEvent(pitch: 'E3', duration: 1, beat: 3, chordNotes: ['G3']),
        NoteEvent(pitch: '', duration: 1, beat: 4, isRest: true),
      ]),
    ],
  ),
  Sample(
    id: 'sample_doremi',
    title: '도레미송',
    emoji: '🎵',
    tempo: 104,
    timeSignature: [4, 4],
    notes: [
      NoteEvent(pitch: 'C4', duration: 1.5, beat: 1),
      NoteEvent(pitch: 'D4', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'E4', duration: 1.5, beat: 3),
      NoteEvent(pitch: 'C4', duration: 0.5, beat: 4.5),
      NoteEvent(pitch: 'E4', duration: 1, beat: 1),
      NoteEvent(pitch: 'C4', duration: 1, beat: 2),
      NoteEvent(pitch: 'E4', duration: 2, beat: 3),
      NoteEvent(pitch: 'D4', duration: 1.5, beat: 1),
      NoteEvent(pitch: 'E4', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'F4', duration: 0.5, beat: 3),
      NoteEvent(pitch: 'F4', duration: 0.5, beat: 3.5),
      NoteEvent(pitch: 'E4', duration: 0.5, beat: 4),
      NoteEvent(pitch: 'D4', duration: 0.5, beat: 4.5),
      NoteEvent(pitch: 'F4', duration: 4, beat: 1),
    ],
  ),
  Sample(
    id: 'sample_schoolstart',
    title: '수업 시작이다',
    emoji: '🔔',
    tempo: 108,
    timeSignature: [4, 4],
    notes: [
      NoteEvent(pitch: 'F4', duration: 1.5, beat: 1),
      NoteEvent(pitch: 'A4', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'C5', duration: 2, beat: 3),
      NoteEvent(pitch: 'D5', duration: 0.75, beat: 1),
      NoteEvent(pitch: 'C5', duration: 0.25, beat: 1.75),
      NoteEvent(pitch: 'A#4', duration: 0.5, beat: 2),
      NoteEvent(pitch: 'D5', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'C5', duration: 2, beat: 3),
      NoteEvent(pitch: 'A#4', duration: 1.5, beat: 1),
      NoteEvent(pitch: 'C5', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'A4', duration: 1.5, beat: 3),
      NoteEvent(pitch: 'F4', duration: 0.5, beat: 4.5),
      NoteEvent(pitch: 'G4', duration: 1, beat: 1),
      NoteEvent(pitch: 'A4', duration: 1, beat: 2),
      NoteEvent(pitch: 'F4', duration: 2, beat: 3),
    ],
    staves: [
      Stave(clef: 'treble', notes: [
        NoteEvent(pitch: 'F4', duration: 1.5, beat: 1),
        NoteEvent(pitch: 'A4', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'C5', duration: 2, beat: 3),
        NoteEvent(pitch: 'D5', duration: 0.75, beat: 1),
        NoteEvent(pitch: 'C5', duration: 0.25, beat: 1.75),
        NoteEvent(pitch: 'A#4', duration: 0.5, beat: 2),
        NoteEvent(pitch: 'D5', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'C5', duration: 2, beat: 3),
        NoteEvent(pitch: 'A#4', duration: 1.5, beat: 1),
        NoteEvent(pitch: 'C5', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'A4', duration: 1.5, beat: 3),
        NoteEvent(pitch: 'F4', duration: 0.5, beat: 4.5),
        NoteEvent(pitch: 'G4', duration: 1, beat: 1),
        NoteEvent(pitch: 'A4', duration: 1, beat: 2),
        NoteEvent(pitch: 'F4', duration: 2, beat: 3),
      ]),
      Stave(clef: 'bass', notes: [
        NoteEvent(pitch: 'F3', duration: 0.5, beat: 1),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 1.5),
        NoteEvent(pitch: 'A3', duration: 0.5, beat: 2),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'F3', duration: 0.5, beat: 3),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 3.5),
        NoteEvent(pitch: 'A3', duration: 0.5, beat: 4),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 4.5),
        NoteEvent(pitch: 'F3', duration: 0.5, beat: 1),
        NoteEvent(pitch: 'D4', duration: 0.5, beat: 1.5),
        NoteEvent(pitch: 'A#3', duration: 0.5, beat: 2),
        NoteEvent(pitch: 'D4', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'F3', duration: 0.5, beat: 3),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 3.5),
        NoteEvent(pitch: 'A3', duration: 0.5, beat: 4),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 4.5),
        NoteEvent(pitch: 'E3', duration: 0.5, beat: 1),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 1.5),
        NoteEvent(pitch: 'A#3', duration: 0.5, beat: 2),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'F3', duration: 0.5, beat: 3),
        NoteEvent(pitch: 'C4', duration: 0.5, beat: 3.5),
        NoteEvent(pitch: 'A#3', duration: 0.5, beat: 4),
        NoteEvent(pitch: 'A3', duration: 0.5, beat: 4.5),
        NoteEvent(pitch: 'G3', duration: 1, beat: 1),
        NoteEvent(pitch: 'E3', duration: 1, beat: 2),
        NoteEvent(pitch: 'C3', duration: 2, beat: 3, chordNotes: ['F3']),
      ]),
    ],
  ),
  Sample(
    id: 'sample_christmas',
    title: '크리스마스',
    emoji: '🎄',
    tempo: 80,
    timeSignature: [12, 8],
    notes: [
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 1),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 1.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 2),
      NoteEvent(pitch: 'A#5', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 3),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 3.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 4),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 4.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 5),
      NoteEvent(pitch: 'A#5', duration: 0.5, beat: 5.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 6),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 6.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 1),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 1.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 2),
      NoteEvent(pitch: 'G#5', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 3),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 3.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 4),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 4.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 5),
      NoteEvent(pitch: 'G#5', duration: 0.5, beat: 5.5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 6),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 6.5),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 1),
      NoteEvent(pitch: 'C#5', duration: 0.5, beat: 1.5),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 2),
      NoteEvent(pitch: 'G#5', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 3),
      NoteEvent(pitch: 'C#5', duration: 0.5, beat: 3.5),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 4),
      NoteEvent(pitch: 'C#5', duration: 0.5, beat: 4.5),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 5),
      NoteEvent(pitch: 'G#5', duration: 0.5, beat: 5.5),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 6),
      NoteEvent(pitch: 'C#5', duration: 0.5, beat: 6.5),
      NoteEvent(pitch: 'C5', duration: 0.5, beat: 1, chordNotes: ['D#5']),
      NoteEvent(pitch: 'C#5', duration: 0.5, beat: 1.5),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 2),
      NoteEvent(pitch: 'G#5', duration: 0.5, beat: 2.5),
      NoteEvent(pitch: 'D#5', duration: 0.5, beat: 3),
      NoteEvent(pitch: 'C#5', duration: 0.5, beat: 3.5),
      NoteEvent(pitch: 'C5', duration: 0.5, beat: 4),
      NoteEvent(pitch: 'A#4', duration: 0.5, beat: 4.5),
      NoteEvent(pitch: 'C5', duration: 0.5, beat: 5),
      NoteEvent(pitch: 'F5', duration: 0.5, beat: 5.5),
      NoteEvent(pitch: 'C5', duration: 0.5, beat: 6),
      NoteEvent(pitch: 'A#4', duration: 0.5, beat: 6.5),
    ],
    staves: [
      Stave(clef: 'treble', notes: [
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 1),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 1.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 2),
        NoteEvent(pitch: 'A#5', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 3),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 3.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 4),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 4.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 5),
        NoteEvent(pitch: 'A#5', duration: 0.5, beat: 5.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 6),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 6.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 1),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 1.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 2),
        NoteEvent(pitch: 'G#5', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 3),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 3.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 4),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 4.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 5),
        NoteEvent(pitch: 'G#5', duration: 0.5, beat: 5.5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 6),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 6.5),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 1),
        NoteEvent(pitch: 'C#5', duration: 0.5, beat: 1.5),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 2),
        NoteEvent(pitch: 'G#5', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 3),
        NoteEvent(pitch: 'C#5', duration: 0.5, beat: 3.5),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 4),
        NoteEvent(pitch: 'C#5', duration: 0.5, beat: 4.5),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 5),
        NoteEvent(pitch: 'G#5', duration: 0.5, beat: 5.5),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 6),
        NoteEvent(pitch: 'C#5', duration: 0.5, beat: 6.5),
        NoteEvent(pitch: 'C5', duration: 0.5, beat: 1, chordNotes: ['D#5']),
        NoteEvent(pitch: 'C#5', duration: 0.5, beat: 1.5),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 2),
        NoteEvent(pitch: 'G#5', duration: 0.5, beat: 2.5),
        NoteEvent(pitch: 'D#5', duration: 0.5, beat: 3),
        NoteEvent(pitch: 'C#5', duration: 0.5, beat: 3.5),
        NoteEvent(pitch: 'C5', duration: 0.5, beat: 4),
        NoteEvent(pitch: 'A#4', duration: 0.5, beat: 4.5),
        NoteEvent(pitch: 'C5', duration: 0.5, beat: 5),
        NoteEvent(pitch: 'F5', duration: 0.5, beat: 5.5),
        NoteEvent(pitch: 'C5', duration: 0.5, beat: 6),
        NoteEvent(pitch: 'A#4', duration: 0.5, beat: 6.5),
      ]),
      Stave(clef: 'bass', notes: [
        NoteEvent(pitch: 'F#3', duration: 6, beat: 1, chordNotes: ['G#3', 'C#4']),
        NoteEvent(pitch: 'F#3', duration: 6, beat: 1, chordNotes: ['G#3', 'C4']),
        NoteEvent(pitch: 'F3', duration: 6, beat: 1, chordNotes: ['G#3', 'C4']),
        NoteEvent(pitch: 'A#2', duration: 3, beat: 1, chordNotes: ['F3', 'G#3']),
        NoteEvent(pitch: 'G#2', duration: 3, beat: 4, chordNotes: ['C3', 'C#3', 'F3']),
      ]),
    ],
  ),
];

const List<String> noteNames = [
  'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
];

const Map<String, String> _blackLabel = {
  'C#': '1', 'D#': '2', 'F#': '3', 'G#': '4', 'A#': '5',
};

const Map<int, int> beatColorValues = {
  1: 0xFF0076CE,
  2: 0xFF5BB8F5,
  3: 0xFFA8D5F5,
  4: 0xFFD0EBFA,
};

const int beatColorFallback = 0xFF888888;

int beatColor(double beat) {
  final rounded = beat.round();
  if (rounded == beat) {
    return beatColorValues[rounded] ?? beatColorFallback;
  }
  return beatColorFallback;
}

/// 커스텀 악보 표기: 옥타브 숫자 없음 — 흰건반은 알파벳, 검은건반은 1~5 숫자.
String formatNoteName(String pitch) {
  final name = pitch.substring(0, pitch.length - 1);
  return _blackLabel[name] ?? name;
}

int _octaveOf(String pitch) => int.parse(pitch.substring(pitch.length - 1));

/// 높은음자리표: 4옥(가온다 포함)=아래(2), 5옥=중간(1), 6옥+=위(0)
/// 낮은음자리표: 4옥 이상=위(0), 3옥=중간(1), 2옥 이하=아래(2)
int pitchToZone(String pitch, String clef) {
  final oct = _octaveOf(pitch);
  if (clef == 'bass') {
    if (oct >= 4) return 0;
    if (oct == 3) return 1;
    return 2;
  }
  if (oct >= 6) return 0;
  if (oct == 5) return 1;
  return 2;
}

String effectiveClef(NoteEvent note, String staffClef) => note.clef ?? staffClef;

bool hasMixedClef(List<NoteEvent> notes, String staffClef) {
  String? first;
  for (final n in notes) {
    final c = effectiveClef(n, staffClef);
    first ??= c;
    if (c != first) return true;
  }
  return false;
}
