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
    this._init();
    if (this.ctx.state === 'suspended') this.ctx.resume();

    const freq = noteToFrequency(pitch);
    const now = this.ctx.currentTime;
    const env = this.ctx.createGain();
    env.connect(this.master);

    // ADSR envelope for piano-like sound
    env.gain.setValueAtTime(0, now);
    env.gain.linearRampToValueAtTime(0.9, now + 0.008);
    env.gain.exponentialRampToValueAtTime(0.6, now + 0.08);
    env.gain.setValueAtTime(0.6, now + Math.max(durationSec - 0.05, 0.05));
    env.gain.exponentialRampToValueAtTime(0.001, now + durationSec + 0.35);

    // Fundamental + 2nd harmonic for richer tone
    const partials = [
      { type: 'triangle', ratio: 1, gain: 0.65 },
      { type: 'sine',     ratio: 2, gain: 0.22 },
      { type: 'sine',     ratio: 3, gain: 0.08 },
    ];
    partials.forEach(({ type, ratio, gain }) => {
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
