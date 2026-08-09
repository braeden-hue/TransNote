// 아래 3곡은 designKit/*.mscz(나비야, 도레미송, 수업 시작이다)를 MuseScore로 렌더링한 뒤
// 실제 학습된 OMR 모델(round3train/checkpoints/r15_cropfix_coordconv)로 인식시켜 얻은
// 진짜 변환 결과다(손으로 입력한 예시 아님) — server.py의 /api/recognize와 동일한 파이프라인.
// 추후 더 추가될 예정.
export const SAMPLES = [
  {
    id: 'sample_butterfly',
    title: '나비야 나비야',
    emoji: '🦋',
    tempo: 95,
    timeSignature: [4, 4],
    notes: [
      { pitch: 'G4', duration: 1, beat: 1 },
      { pitch: 'E4', duration: 1, beat: 2 },
      { pitch: 'E4', duration: 2, beat: 3 },
      { pitch: 'F4', duration: 1, beat: 1 },
      { pitch: 'D4', duration: 1, beat: 2 },
      { pitch: 'D4', duration: 2, beat: 3 },
      { pitch: 'C4', duration: 1, beat: 1 },
      { pitch: 'D4', duration: 1, beat: 2 },
      { pitch: 'E4', duration: 1, beat: 3 },
      { pitch: 'F4', duration: 1, beat: 4 },
      { pitch: 'G4', duration: 1, beat: 1 },
      { pitch: 'G4', duration: 1, beat: 2 },
      { pitch: 'G4', duration: 2, beat: 3 },
      { pitch: 'G4', duration: 1, beat: 1 },
      { pitch: 'E4', duration: 1, beat: 2 },
      { pitch: 'E4', duration: 1, beat: 3 },
      { pitch: 'E4', duration: 1, beat: 4 },
      { pitch: 'C4', duration: 1, beat: 1 },
      { pitch: 'E4', duration: 1, beat: 2 },
      { pitch: 'G4', duration: 2, beat: 3 },
      { pitch: 'E4', duration: 1, beat: 1 },
      { pitch: 'E4', duration: 1, beat: 2 },
      { pitch: 'E4', duration: 2, beat: 3 },
    ],
    staves: [
      {
        clef: 'treble',
        notes: [
          { pitch: 'G4', duration: 1, beat: 1 },
          { pitch: 'E4', duration: 1, beat: 2 },
          { pitch: 'E4', duration: 2, beat: 3 },
          { pitch: 'F4', duration: 1, beat: 1 },
          { pitch: 'D4', duration: 1, beat: 2 },
          { pitch: 'D4', duration: 2, beat: 3 },
          { pitch: 'C4', duration: 1, beat: 1 },
          { pitch: 'D4', duration: 1, beat: 2 },
          { pitch: 'E4', duration: 1, beat: 3 },
          { pitch: 'F4', duration: 1, beat: 4 },
          { pitch: 'G4', duration: 1, beat: 1 },
          { pitch: 'G4', duration: 1, beat: 2 },
          { pitch: 'G4', duration: 2, beat: 3 },
          { pitch: 'G4', duration: 1, beat: 1 },
          { pitch: 'E4', duration: 1, beat: 2 },
          { pitch: 'E4', duration: 1, beat: 3 },
          { pitch: 'E4', duration: 1, beat: 4 },
          { pitch: 'C4', duration: 1, beat: 1 },
          { pitch: 'E4', duration: 1, beat: 2 },
          { pitch: 'G4', duration: 2, beat: 3 },
          { pitch: 'E4', duration: 1, beat: 1 },
          { pitch: 'E4', duration: 1, beat: 2 },
          { pitch: 'E4', duration: 2, beat: 3 },
        ],
      },
      {
        clef: 'bass',
        notes: [
          { pitch: 'C3', duration: 2, beat: 1 },
          { pitch: 'E3', duration: 2, beat: 3, chordNotes: ['G3'] },
          { pitch: 'D3', duration: 2, beat: 1 },
          { pitch: 'F3', duration: 2, beat: 3, chordNotes: ['G3'] },
          { pitch: 'C3', duration: 1, beat: 1 },
          { pitch: 'G3', duration: 1, beat: 2 },
          { pitch: 'E3', duration: 1, beat: 3 },
          { pitch: 'F3', duration: 1, beat: 4 },
          { pitch: 'E3', duration: 1, beat: 1, chordNotes: ['G3'] },
          { pitch: 'E3', duration: 1, beat: 2, chordNotes: ['G3'] },
          { pitch: 'E3', duration: 1, beat: 3, chordNotes: ['G3'] },
          { pitch: '', duration: 1, beat: 4, isRest: true },
          { pitch: 'C3', duration: 1, beat: 1 },
          { pitch: 'G3', duration: 1, beat: 2 },
          { pitch: 'E3', duration: 1, beat: 3 },
          { pitch: 'G3', duration: 1, beat: 4 },
          { pitch: 'C3', duration: 2, beat: 1 },
          { pitch: 'E3', duration: 2, beat: 3, chordNotes: ['G3'] },
          { pitch: 'E3', duration: 1, beat: 1, chordNotes: ['G3'] },
          { pitch: 'E3', duration: 1, beat: 2, chordNotes: ['G3'] },
          { pitch: 'E3', duration: 1, beat: 3, chordNotes: ['G3'] },
          { pitch: '', duration: 1, beat: 4, isRest: true },
        ],
      },
    ],
  },
  {
    id: 'sample_doremi',
    title: '도레미송',
    emoji: '🎵',
    tempo: 104,
    timeSignature: [4, 4],
    // designKit/도레미송.mscz와 대조해서 다시 뽑은 정확한 데이터(마디 4개, beat는
    // 마디 안 진짜 누적 위치 — 예전 값은 마디 경계가 잘못 나뉘어 있었음).
    notes: [
      { pitch: 'C4', duration: 1.5, beat: 1 },
      { pitch: 'D4', duration: 0.5, beat: 2.5 },
      { pitch: 'E4', duration: 1.5, beat: 3 },
      { pitch: 'C4', duration: 0.5, beat: 4.5 },
      { pitch: 'E4', duration: 1, beat: 1 },
      { pitch: 'C4', duration: 1, beat: 2 },
      { pitch: 'E4', duration: 2, beat: 3 },
      { pitch: 'D4', duration: 1.5, beat: 1 },
      { pitch: 'E4', duration: 0.5, beat: 2.5 },
      { pitch: 'F4', duration: 0.5, beat: 3 },
      { pitch: 'F4', duration: 0.5, beat: 3.5 },
      { pitch: 'E4', duration: 0.5, beat: 4 },
      { pitch: 'D4', duration: 0.5, beat: 4.5 },
      { pitch: 'F4', duration: 4, beat: 1 },
    ],
    // 도레미송은 원래 단일 오선(왼손 반주 없음) — staves(대보표)를 넣지 않아야
    // sampleToNotation()이 notes만으로 단일 보표로 취급한다.
  },
  {
    id: 'sample_schoolstart',
    title: '수업 시작이다',
    emoji: '🔔',
    tempo: 108,
    timeSignature: [4, 4],
    // designKit/수업 시작이다.mscz와 대조해서 다시 뽑음 — 음 자체는 기존과 같았지만
    // 마디 경계(beat)가 잘못 잘려 있었음(특히 왼손 8분음표 반주가 4음마디로 잘못 쪼개짐).
    notes: [
      { pitch: 'F4', duration: 1.5, beat: 1 },
      { pitch: 'A4', duration: 0.5, beat: 2.5 },
      { pitch: 'C5', duration: 2, beat: 3 },
      { pitch: 'D5', duration: 0.75, beat: 1 },
      { pitch: 'C5', duration: 0.25, beat: 1.75 },
      { pitch: 'A#4', duration: 0.5, beat: 2 },
      { pitch: 'D5', duration: 0.5, beat: 2.5 },
      { pitch: 'C5', duration: 2, beat: 3 },
      { pitch: 'A#4', duration: 1.5, beat: 1 },
      { pitch: 'C5', duration: 0.5, beat: 2.5 },
      { pitch: 'A4', duration: 1.5, beat: 3 },
      { pitch: 'F4', duration: 0.5, beat: 4.5 },
      { pitch: 'G4', duration: 1, beat: 1 },
      { pitch: 'A4', duration: 1, beat: 2 },
      { pitch: 'F4', duration: 2, beat: 3 },
    ],
    staves: [
      {
        clef: 'treble',
        notes: [
          { pitch: 'F4', duration: 1.5, beat: 1 },
          { pitch: 'A4', duration: 0.5, beat: 2.5 },
          { pitch: 'C5', duration: 2, beat: 3 },
          { pitch: 'D5', duration: 0.75, beat: 1 },
          { pitch: 'C5', duration: 0.25, beat: 1.75 },
          { pitch: 'A#4', duration: 0.5, beat: 2 },
          { pitch: 'D5', duration: 0.5, beat: 2.5 },
          { pitch: 'C5', duration: 2, beat: 3 },
          { pitch: 'A#4', duration: 1.5, beat: 1 },
          { pitch: 'C5', duration: 0.5, beat: 2.5 },
          { pitch: 'A4', duration: 1.5, beat: 3 },
          { pitch: 'F4', duration: 0.5, beat: 4.5 },
          { pitch: 'G4', duration: 1, beat: 1 },
          { pitch: 'A4', duration: 1, beat: 2 },
          { pitch: 'F4', duration: 2, beat: 3 },
        ],
      },
      {
        clef: 'bass',
        notes: [
          { pitch: 'F3', duration: 0.5, beat: 1 },
          { pitch: 'C4', duration: 0.5, beat: 1.5 },
          { pitch: 'A3', duration: 0.5, beat: 2 },
          { pitch: 'C4', duration: 0.5, beat: 2.5 },
          { pitch: 'F3', duration: 0.5, beat: 3 },
          { pitch: 'C4', duration: 0.5, beat: 3.5 },
          { pitch: 'A3', duration: 0.5, beat: 4 },
          { pitch: 'C4', duration: 0.5, beat: 4.5 },
          { pitch: 'F3', duration: 0.5, beat: 1 },
          { pitch: 'D4', duration: 0.5, beat: 1.5 },
          { pitch: 'A#3', duration: 0.5, beat: 2 },
          { pitch: 'D4', duration: 0.5, beat: 2.5 },
          { pitch: 'F3', duration: 0.5, beat: 3 },
          { pitch: 'C4', duration: 0.5, beat: 3.5 },
          { pitch: 'A3', duration: 0.5, beat: 4 },
          { pitch: 'C4', duration: 0.5, beat: 4.5 },
          { pitch: 'E3', duration: 0.5, beat: 1 },
          { pitch: 'C4', duration: 0.5, beat: 1.5 },
          { pitch: 'A#3', duration: 0.5, beat: 2 },
          { pitch: 'C4', duration: 0.5, beat: 2.5 },
          { pitch: 'F3', duration: 0.5, beat: 3 },
          { pitch: 'C4', duration: 0.5, beat: 3.5 },
          { pitch: 'A#3', duration: 0.5, beat: 4 },
          { pitch: 'A3', duration: 0.5, beat: 4.5 },
          { pitch: 'G3', duration: 1, beat: 1 },
          { pitch: 'E3', duration: 1, beat: 2 },
          { pitch: 'C3', duration: 2, beat: 3, chordNotes: ['F3'] },
        ],
      },
    ],
  },
  {
    id: 'sample_christmas',
    title: '크리스마스',
    emoji: '🎄',
    tempo: 80,
    timeSignature: [12, 8],
    // designKit/christmas.mscz에서 그대로 뽑음(12/8, 4마디, 오른손 아르페지오 + 왼손 지속화음).
    notes: [
      { pitch: 'F5', duration: 0.5, beat: 1 },
      { pitch: 'D#5', duration: 0.5, beat: 1.5 },
      { pitch: 'F5', duration: 0.5, beat: 2 },
      { pitch: 'A#5', duration: 0.5, beat: 2.5 },
      { pitch: 'F5', duration: 0.5, beat: 3 },
      { pitch: 'D#5', duration: 0.5, beat: 3.5 },
      { pitch: 'F5', duration: 0.5, beat: 4 },
      { pitch: 'D#5', duration: 0.5, beat: 4.5 },
      { pitch: 'F5', duration: 0.5, beat: 5 },
      { pitch: 'A#5', duration: 0.5, beat: 5.5 },
      { pitch: 'F5', duration: 0.5, beat: 6 },
      { pitch: 'D#5', duration: 0.5, beat: 6.5 },
      { pitch: 'F5', duration: 0.5, beat: 1 },
      { pitch: 'D#5', duration: 0.5, beat: 1.5 },
      { pitch: 'F5', duration: 0.5, beat: 2 },
      { pitch: 'G#5', duration: 0.5, beat: 2.5 },
      { pitch: 'F5', duration: 0.5, beat: 3 },
      { pitch: 'D#5', duration: 0.5, beat: 3.5 },
      { pitch: 'F5', duration: 0.5, beat: 4 },
      { pitch: 'D#5', duration: 0.5, beat: 4.5 },
      { pitch: 'F5', duration: 0.5, beat: 5 },
      { pitch: 'G#5', duration: 0.5, beat: 5.5 },
      { pitch: 'F5', duration: 0.5, beat: 6 },
      { pitch: 'D#5', duration: 0.5, beat: 6.5 },
      { pitch: 'D#5', duration: 0.5, beat: 1 },
      { pitch: 'C#5', duration: 0.5, beat: 1.5 },
      { pitch: 'D#5', duration: 0.5, beat: 2 },
      { pitch: 'G#5', duration: 0.5, beat: 2.5 },
      { pitch: 'D#5', duration: 0.5, beat: 3 },
      { pitch: 'C#5', duration: 0.5, beat: 3.5 },
      { pitch: 'D#5', duration: 0.5, beat: 4 },
      { pitch: 'C#5', duration: 0.5, beat: 4.5 },
      { pitch: 'D#5', duration: 0.5, beat: 5 },
      { pitch: 'G#5', duration: 0.5, beat: 5.5 },
      { pitch: 'D#5', duration: 0.5, beat: 6 },
      { pitch: 'C#5', duration: 0.5, beat: 6.5 },
      { pitch: 'C5', duration: 0.5, beat: 1, chordNotes: ['D#5'] },
      { pitch: 'C#5', duration: 0.5, beat: 1.5 },
      { pitch: 'D#5', duration: 0.5, beat: 2 },
      { pitch: 'G#5', duration: 0.5, beat: 2.5 },
      { pitch: 'D#5', duration: 0.5, beat: 3 },
      { pitch: 'C#5', duration: 0.5, beat: 3.5 },
      { pitch: 'C5', duration: 0.5, beat: 4 },
      { pitch: 'A#4', duration: 0.5, beat: 4.5 },
      { pitch: 'C5', duration: 0.5, beat: 5 },
      { pitch: 'F5', duration: 0.5, beat: 5.5 },
      { pitch: 'C5', duration: 0.5, beat: 6 },
      { pitch: 'A#4', duration: 0.5, beat: 6.5 },
    ],
    staves: [
      {
        clef: 'treble',
        notes: [
          { pitch: 'F5', duration: 0.5, beat: 1 },
          { pitch: 'D#5', duration: 0.5, beat: 1.5 },
          { pitch: 'F5', duration: 0.5, beat: 2 },
          { pitch: 'A#5', duration: 0.5, beat: 2.5 },
          { pitch: 'F5', duration: 0.5, beat: 3 },
          { pitch: 'D#5', duration: 0.5, beat: 3.5 },
          { pitch: 'F5', duration: 0.5, beat: 4 },
          { pitch: 'D#5', duration: 0.5, beat: 4.5 },
          { pitch: 'F5', duration: 0.5, beat: 5 },
          { pitch: 'A#5', duration: 0.5, beat: 5.5 },
          { pitch: 'F5', duration: 0.5, beat: 6 },
          { pitch: 'D#5', duration: 0.5, beat: 6.5 },
          { pitch: 'F5', duration: 0.5, beat: 1 },
          { pitch: 'D#5', duration: 0.5, beat: 1.5 },
          { pitch: 'F5', duration: 0.5, beat: 2 },
          { pitch: 'G#5', duration: 0.5, beat: 2.5 },
          { pitch: 'F5', duration: 0.5, beat: 3 },
          { pitch: 'D#5', duration: 0.5, beat: 3.5 },
          { pitch: 'F5', duration: 0.5, beat: 4 },
          { pitch: 'D#5', duration: 0.5, beat: 4.5 },
          { pitch: 'F5', duration: 0.5, beat: 5 },
          { pitch: 'G#5', duration: 0.5, beat: 5.5 },
          { pitch: 'F5', duration: 0.5, beat: 6 },
          { pitch: 'D#5', duration: 0.5, beat: 6.5 },
          { pitch: 'D#5', duration: 0.5, beat: 1 },
          { pitch: 'C#5', duration: 0.5, beat: 1.5 },
          { pitch: 'D#5', duration: 0.5, beat: 2 },
          { pitch: 'G#5', duration: 0.5, beat: 2.5 },
          { pitch: 'D#5', duration: 0.5, beat: 3 },
          { pitch: 'C#5', duration: 0.5, beat: 3.5 },
          { pitch: 'D#5', duration: 0.5, beat: 4 },
          { pitch: 'C#5', duration: 0.5, beat: 4.5 },
          { pitch: 'D#5', duration: 0.5, beat: 5 },
          { pitch: 'G#5', duration: 0.5, beat: 5.5 },
          { pitch: 'D#5', duration: 0.5, beat: 6 },
          { pitch: 'C#5', duration: 0.5, beat: 6.5 },
          { pitch: 'C5', duration: 0.5, beat: 1, chordNotes: ['D#5'] },
          { pitch: 'C#5', duration: 0.5, beat: 1.5 },
          { pitch: 'D#5', duration: 0.5, beat: 2 },
          { pitch: 'G#5', duration: 0.5, beat: 2.5 },
          { pitch: 'D#5', duration: 0.5, beat: 3 },
          { pitch: 'C#5', duration: 0.5, beat: 3.5 },
          { pitch: 'C5', duration: 0.5, beat: 4 },
          { pitch: 'A#4', duration: 0.5, beat: 4.5 },
          { pitch: 'C5', duration: 0.5, beat: 5 },
          { pitch: 'F5', duration: 0.5, beat: 5.5 },
          { pitch: 'C5', duration: 0.5, beat: 6 },
          { pitch: 'A#4', duration: 0.5, beat: 6.5 },
        ],
      },
      {
        clef: 'bass',
        notes: [
          { pitch: 'F#3', duration: 6, beat: 1, chordNotes: ['G#3', 'C#4'] },
          { pitch: 'F#3', duration: 6, beat: 1, chordNotes: ['G#3', 'C4'] },
          { pitch: 'F3', duration: 6, beat: 1, chordNotes: ['G#3', 'C4'] },
          { pitch: 'A#2', duration: 3, beat: 1, chordNotes: ['F3', 'G#3'] },
          { pitch: 'G#2', duration: 3, beat: 4, chordNotes: ['C3', 'C#3', 'F3'] },
        ],
      },
    ],
  },
];

export const BEAT_COLORS = { 1: '#0076CE', 2: '#5BB8F5', 3: '#A8D5F5', 4: '#D0EBFA' };

export const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

// 커스텀 악보 표기: 옥타브 숫자 없음 — 존(행)이 옥타브를 나타냄
// 흰건반: C D E F G A B  /  검은건반: 1(C#) 2(D#) 3(F#) 4(G#) 5(A#)
const BLACK_LABEL = { 'C#':'1', 'D#':'2', 'F#':'3', 'G#':'4', 'A#':'5' };

export function formatNoteName(pitch) {
  const name = pitch.slice(0, -1);
  return BLACK_LABEL[name] ?? name;
}

export function formatChordName(pitches) {
  return pitches.map(formatNoteName).join('');
}

// 높은음자리표: 4옥(가온다 포함) = 아래(2), 5옥 = 중간(1), 6옥+ = 위(0)
// 낮은음자리표: 3옥(C3~B3) = 위(0), 2옥 = 중간(1), 1옥 이하 = 아래(2)
export function pitchToZone(pitch, clef = 'treble') {
  const oct = parseInt(pitch.slice(-1));
  if (clef === 'bass') {
    if (oct >= 3) return 0;
    if (oct === 2) return 1;
    return 2;
  }
  if (oct >= 6) return 0;
  if (oct === 5) return 1;
  return 2;
}

export function zoneLabels(clef = 'treble') {
  if (clef === 'bass') return ['높음 (3옥)', '중간 (2옥)', '낮음 (1옥↓)'];
  return ['높음 (6옥+)', '중간 (5옥)', '낮음 (4옥, 가온다)'];
}

// note의 유효 클렙(effective clef) = note.clef 오버라이드가 있으면 그 값, 없으면 보표 clef 상속.
// (마디 중간 클렙 전환 표기 — docs/music-notation-rule-designer.md 참고)
export function effectiveClef(note, staffClef) {
  return note.clef ?? staffClef;
}

// note 리스트 안에 서로 다른 유효 클렙이 2개 이상 섞여 있는지 판정.
// true일 때만 마디 중간 클렙 전환 배경 틴트를 그린다 (섞여 있지 않으면 기존 렌더링과 동일해야 함).
export function hasMixedClef(notes, staffClef) {
  let first;
  for (const n of notes) {
    const c = effectiveClef(n, staffClef);
    if (first === undefined) first = c;
    else if (c !== first) return true;
  }
  return false;
}

export function noteToFrequency(pitch) {
  const name = pitch.slice(0, -1);
  const oct  = parseInt(pitch.slice(-1));
  const semi = NOTE_NAMES.indexOf(name);
  return 440 * Math.pow(2, (semi + (oct - 4) * 12 - 9) / 12);
}
