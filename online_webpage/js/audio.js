import { noteToFrequency } from './samples.js';

class AudioEngine {
  constructor() {
    this.ctx = null;
    this.master = null;
  }

  _init() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.6;
    this.master.connect(this.ctx.destination);
  }

  unlock() {
    this._init();
    if (this.ctx.state === 'suspended') this.ctx.resume();
  }

  playNote(pitch, durationSec = 0.5) {
    if (!pitch) return; // 쉼표(pitch 없음) — 소리 없이 넘어감
    this._init();
    if (this.ctx.state === 'suspended') this.ctx.resume();

    const freq = noteToFrequency(pitch);
    const now = this.ctx.currentTime;
    const env = this.ctx.createGain();
    env.connect(this.master);

    // 저음역 보강 — 태블릿 스피커는 기본음(fundamental) 자체가 낮으면 물리적으로
    // 잘 재생을 못 해서 "저음이 안 들린다"고 느껴진다. 실제 피아노 저음이 작은
    // 스피커로도 들리는 건 배음(harmonics)이 풍부해서 기본음 없이도 배음만으로
    // 음높이가 인지되기 때문(missing fundamental) — 이 원리를 그대로 흉내낸다.
    // C4 위로는 원래 소리 그대로, A1 이하는 배음/게인을 최대로 보강.
    const REF_HI = 261.63; // C4
    const REF_LO = 55;     // A1
    const bassAmt = Math.min(1, Math.max(0, (REF_HI - freq) / (REF_HI - REF_LO)));
    const loudnessBoost = 1 + bassAmt * 0.25; // 최대 25% 게인 보강

    // ADSR envelope for piano-like sound
    const peak = 0.9 * loudnessBoost;
    const sustain = 0.6 * loudnessBoost;
    env.gain.setValueAtTime(0, now);
    env.gain.linearRampToValueAtTime(peak, now + 0.008);
    env.gain.exponentialRampToValueAtTime(sustain, now + 0.08);
    env.gain.setValueAtTime(sustain, now + Math.max(durationSec - 0.05, 0.05));
    env.gain.exponentialRampToValueAtTime(0.001, now + durationSec + 0.35);

    // Fundamental + harmonics — 저음일수록(bassAmt↑) 기본음 비중을 줄이고
    // 배음 비중/개수를 늘려 작은 스피커에서도 음높이가 들리게 한다.
    const partials = [
      { type: 'triangle', ratio: 1, gain: 0.65 - bassAmt * 0.15 },
      { type: 'sine',     ratio: 2, gain: 0.22 + bassAmt * 0.10 },
      { type: 'sine',     ratio: 3, gain: 0.08 + bassAmt * 0.08 },
      { type: 'sine',     ratio: 4, gain: bassAmt * 0.08 },
      { type: 'sine',     ratio: 5, gain: bassAmt * 0.04 },
    ];
    partials.forEach(({ type, ratio, gain }) => {
      if (gain <= 0) return;
      const osc = this.ctx.createOscillator();
      const g   = this.ctx.createGain();
      osc.type = type;
      osc.frequency.value = freq * ratio;
      g.gain.value = gain;
      osc.connect(g);
      g.connect(env);
      osc.start(now);
      osc.stop(now + durationSec + 0.4);
    });
  }

  // Returns a cancel function
  playSequence(notes, tempo, onStep, onEnd) {
    this._init();
    const qSec = 60 / tempo;
    let idx = 0;
    let cancelled = false;
    let timeoutId = null;

    const step = () => {
      if (cancelled || idx >= notes.length) {
        if (!cancelled) onEnd?.();
        return;
      }
      const note = notes[idx];
      this.playNote(note.pitch, note.duration * qSec * 0.9);
      onStep?.(idx);
      idx++;
      timeoutId = setTimeout(step, note.duration * qSec * 1000);
    };

    step();

    return () => {
      cancelled = true;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }
}

export const audio = new AudioEngine();
