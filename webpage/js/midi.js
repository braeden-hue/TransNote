// midi.js — Web MIDI API로 전자 피아노와 양방향 연동.
//   입력: 피아노에서 누른 건반을 note-on/off로 받아 연주하기 화면의 피아노와 동일하게 처리.
//   출력: 자동재생 시 커스텀 악보 음을 실제 피아노로 note-on/off 전송해 소리내게 함.
//
// Web MIDI는 secure context(https:// 또는 http://localhost)에서만 동작한다 — LAN IP로
// http:// 접속한 폰에서는 사용 불가(브라우저 제약, 이 파일에서 고칠 수 있는 부분 아님).

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

let midiAccess  = null;
let curInput    = null;
let curOutput   = null;

export function isSupported() {
  return !!navigator.requestMIDIAccess;
}

export async function requestAccess() {
  midiAccess = await navigator.requestMIDIAccess({ sysex: false });
  return midiAccess;
}

export function listInputs() {
  return midiAccess ? [...midiAccess.inputs.values()] : [];
}

export function listOutputs() {
  return midiAccess ? [...midiAccess.outputs.values()] : [];
}

export function midiNumToPitch(n) {
  return NOTE_NAMES[n % 12] + (Math.floor(n / 12) - 1);
}

function pitchToMidiNum(pitch) {
  if (!pitch) return null;
  const name = pitch.slice(0, -1);
  const oct  = parseInt(pitch.slice(-1));
  const idx  = NOTE_NAMES.indexOf(name);
  if (idx < 0 || Number.isNaN(oct)) return null;
  return (oct + 1) * 12 + idx;
}

// 방금 눌린/뗀 건반을 pitch 문자열로 넘겨준다 (예: 'C4').
export function setInput(id, { onNoteOn, onNoteOff } = {}) {
  if (curInput) curInput.onmidimessage = null;
  curInput = listInputs().find(i => i.id === id) || null;
  if (!curInput) return;

  curInput.onmidimessage = e => {
    const [status, data1, data2] = e.data;
    const cmd = status & 0xf0;
    if (cmd === 0x90 && data2 > 0) {
      onNoteOn?.(midiNumToPitch(data1), data2);
    } else if (cmd === 0x80 || (cmd === 0x90 && data2 === 0)) {
      onNoteOff?.(midiNumToPitch(data1));
    }
  };
}

export function setOutput(id) {
  curOutput = listOutputs().find(o => o.id === id) || null;
}

export function hasOutput() {
  return !!curOutput;
}

export function sendNoteOn(pitch, velocity = 100) {
  const n = pitchToMidiNum(pitch);
  if (n == null || !curOutput) return;
  curOutput.send([0x90, n, velocity]);
}

export function sendNoteOff(pitch) {
  const n = pitchToMidiNum(pitch);
  if (n == null || !curOutput) return;
  curOutput.send([0x80, n, 0]);
}
