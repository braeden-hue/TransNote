import { noteToFrequency } from './samples.js';

// 실제 스타인웨이 그랜드피아노 샘플(합성 아님) — CDN에서 지연 로드, 별도 빌드/에셋
// 번들링 불필요. 로딩 전이나 네트워크 실패 시엔 아래 _playNoteSynth() 합성음으로
// 자동 폴백하므로 무음이 되는 경우는 없다.
let SplendidGrandPiano = null;
const pianoModulePromise = import('https://unpkg.com/smplr/dist/index.mjs')
  .then(mod => { SplendidGrandPiano = mod.SplendidGrandPiano; })
  .catch(() => { /* CDN 접근 불가 등 — 합성음 폴백으로 계속 동작 */ });

class AudioEngine {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.piano = null;
    this.pianoLoaded = false;
  }

  _init() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.6;
    this.master.connect(this.ctx.destination);

    // 페이지 로드 직후(사용자 제스처 전이라도) 바로 샘플 다운로드를 시작해서, 실제로
    // 건반을 누를 때쯤엔 이미 로드가 끝나 있을 확률을 높인다(재생 자체는 아니라서
    // 오디오 정책과 무관하게 미리 받아둘 수 있음).
    pianoModulePromise.then(() => {
      if (!SplendidGrandPiano || this.piano) return;
      this.piano = SplendidGrandPiano(this.ctx, { destination: this.master });
      this.piano.ready.then(() => { this.pianoLoaded = true; }).catch(() => {});
    });
  }

  // preload(): 사용자 제스처(unlock) 이전에도 호출 가능 — 앱 최초 로드 시 한 번
  // 호출해두면 샘플 다운로드를 최대한 일찍 시작할 수 있다.
  preload() { this._init(); }

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

  // 실제 그랜드피아노 샘플이 로드돼 있으면 그걸 쓰고(진짜 녹음, smplr
  // SplendidGrandPiano), 아직 로딩 전이거나 CDN 접근 실패면 _playNoteSynth()
  // 합성음으로 자동 폴백한다 — 항상 소리는 난다.
  playNote(pitch, durationSec = 0.5) {
    if (!pitch) return; // 쉼표(pitch 없음) — 소리 없이 넘어감
    this._init();
    if (this.ctx.state === 'suspended') this.ctx.resume();

    if (this.pianoLoaded) {
      // 저음역 보강(태블릿/폰 스피커의 물리적 저음 재생 한계 대응)은 velocity로 흉내낸다.
      const freq = noteToFrequency(pitch);
      const bassAmt = Math.min(1, Math.max(0, (261.63 - freq) / (261.63 - 55)));
      const velocity = Math.min(127, Math.round(100 + bassAmt * 25));
      this.piano.start({ note: pitch, duration: durationSec, velocity });
      return;
    }
    this._playNoteSynth(pitch, durationSec);
  }

  // 그랜드피아노 실샘플이 없을 때만 쓰는 합성 폴백(위 playNote 참고) — 순수 배음
  // 합성. 오르간/신디사이저 소리가 나던 이전 버전 대비 두 가지를 추가했다:
  // ① inharmonicity(배음 스트레치) — 실제 피아노 현은 강성 때문에 배음이 정확한
  //    정수배가 아니라 위로 갈수록 살짝 날카로워진다. 이걸 넣어야 "너무 깨끗한"
  //    신디사이저 배음이 아니라 실제 현악기 특유의 미세한 셔틀림이 느껴진다.
  // ② sustain 없는 지속 감쇠 — 피아노는 오르간과 달리 건반을 누르고 있어도 평평하게
  //    안 울리고 계속(느리게) 잦아든다. 높은 배음일수록 더 빨리 죽어서, 시간이
  //    지날수록 밝던 타격음이 점점 부드러운 저음 위주로 어두워진다.
  _playNoteSynth(pitch, durationSec = 0.5) {
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

    // 배음 스트레치 계수 — 저음현일수록(bassAmt↑) 물리적으로 더 두꺼워 stretch가 두드러짐
    const B = 0.00055 + bassAmt * 0.0009;
    const stretchedFreq = ratio => freq * ratio * Math.sqrt(1 + B * ratio * ratio);

    // 엔벨로프 타임라인 — 절대시간이 아니라 이전 지점 기준 상대 오프셋으로 쌓아서,
    // 아주 짧은 음(빠른 16분음표 등)이 와도 시간이 역전되는 일이 없게 한다.
    const peak       = 0.95 * loudnessBoost;
    const attackT     = now + 0.006;                                        // 타격(빠른 어택)
    const decay1T     = attackT + Math.max(durationSec * 0.28, 0.03);       // 초반 급감쇠
    const releaseT    = Math.max(decay1T + 0.01, now + durationSec);        // 건반을 뗀 시점(지속 중 계속 감쇠, sustain 없음)
    const tailT       = releaseT + 0.55;                                    // 릴리즈 꼬리

    env.gain.setValueAtTime(0, now);
    env.gain.linearRampToValueAtTime(peak, attackT);
    env.gain.exponentialRampToValueAtTime(peak * 0.35, decay1T);
    env.gain.exponentialRampToValueAtTime(peak * 0.12, releaseT);
    env.gain.exponentialRampToValueAtTime(0.0008, tailT);

    // 배음별 상대 감쇠 속도(1에 가까울수록 오래 남음) — 높은 배음일수록 빨리 죽는다.
    const partials = [
      { ratio: 1, gain: 0.60 - bassAmt * 0.12, decay: 1.00 },
      { ratio: 2, gain: 0.26 + bassAmt * 0.08, decay: 0.85 },
      { ratio: 3, gain: 0.14 + bassAmt * 0.07, decay: 0.68 },
      { ratio: 4, gain: 0.07 + bassAmt * 0.06, decay: 0.55 },
      { ratio: 5, gain: 0.04 + bassAmt * 0.05, decay: 0.45 },
      { ratio: 6, gain: bassAmt * 0.035,       decay: 0.38 },
    ];
    partials.forEach(({ ratio, gain, decay }) => {
      if (gain <= 0) return;
      const osc = this.ctx.createOscillator();
      const g   = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = stretchedFreq(ratio);
      g.gain.setValueAtTime(gain, now);
      g.gain.exponentialRampToValueAtTime(Math.max(gain * 0.001, 0.0001), releaseT + 0.55 * decay);
      osc.connect(g);
      g.connect(env);
      osc.start(now);
      osc.stop(tailT + 0.1);
    });

    // 해머 타격 노이즈 — 아주 짧은(15ms) 필터링된 화이트노이즈로 "쳐지는" 어택감을 더한다.
    const noiseDur = 0.015;
    const bufferSize = Math.max(1, Math.floor(this.ctx.sampleRate * noiseDur));
    const noiseBuf = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const chan = noiseBuf.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) chan[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
    const noise = this.ctx.createBufferSource();
    noise.buffer = noiseBuf;
    const noiseFilter = this.ctx.createBiquadFilter();
    noiseFilter.type = 'bandpass';
    noiseFilter.frequency.value = Math.min(freq * 3, 4000);
    noiseFilter.Q.value = 0.7;
    const noiseGain = this.ctx.createGain();
    noiseGain.gain.value = 0.18 * loudnessBoost;
    noise.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(env);
    noise.start(now);
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
      this.stopAll(); // 이후 예정된 음뿐 아니라 지금 울리고 있는 음도 즉시 끊는다(중첩 재생 방지)
    };
  }

  // 지금 울리고 있는 모든 음을 즉시 정지 — "연주 듣기"를 연타하거나 화면을 벗어날 때
  // 이전 재생이 새 재생/침묵 위에 겹쳐 들리지 않도록 한다. 실샘플 피아노는 즉시
  // 끊기지만, 합성 폴백 중인 음은 이미 스케줄된 개별 오실레이터라 자연 감쇠까지는
  // 그대로 둔다(드물게 폴백 상태에서 겹칠 수 있으나, 실사용 대부분은 샘플 로드 완료 후).
  stopAll() {
    this.piano?.stop();
  }
}

export const audio = new AudioEngine();
