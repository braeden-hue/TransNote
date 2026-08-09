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

  // iOS Safari는 기기의 무음(벨소리) 스위치가 켜져 있으면 Web Audio API로 만든 소리를
  // "ambient" 카테고리로 취급해 조용히 재생을 막는다(에러 없음, 그냥 안 들림) — resume()만
  // 해서는 해결이 안 되고, "미디어 재생"으로 인식시켜야 스위치를 무시하고 소리가 난다.
  // ① 최신 Safari의 AudioSession API로 재생 카테고리를 직접 지정(있으면)
  // ② 구버전/다른 브라우저 대응으로 무음 오디오 태그를 함께 재생(표준 우회 트릭)
  // 둘 다 반드시 사용자 제스처(클릭/터치) 핸들러 안에서 동기적으로 호출돼야 함.
  unlock() {
    this._init();
    if (this.ctx.state === 'suspended') this.ctx.resume();
    if (this._unlocked) return;
    this._unlocked = true;

    if (navigator.audioSession) {
      try { navigator.audioSession.type = 'playback'; } catch { /* 미지원 브라우저는 무시 */ }
    }
    try {
      const silent = new Audio(
        'data:audio/wav;base64,UklGRiUAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQEAAACA'
      );
      silent.play().catch(() => {}); // 자동재생 정책으로 거부돼도 무시(그냥 시도가 중요)
    } catch { /* Audio 생성 자체가 실패해도 무시 */ }
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
      const dur = note.duration * qSec * 0.9;
      this.playNote(note.pitch, dur);
      // 화음(chordNotes) — 딸린 음들도 같이 울려야 "화음"인데, 지금까지는 주 음(pitch)만
      // 나가고 chordNotes는 소리 없이 무시되고 있었다(눈으로는 화음, 귀로는 단음).
      note.chordNotes?.forEach(p => this.playNote(p, dur));
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
