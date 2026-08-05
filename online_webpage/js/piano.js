import { audio } from './audio.js';

// ── 건반 규격 ─────────────────────────────────────────────────────────────────
const WK_W = 36, WK_H = 148, BK_W = 22, BK_H = 92;

const BLACK_DEFS = [
  { name:'C#', pos:0.64 }, { name:'D#', pos:1.64 },
  { name:'F#', pos:3.65 }, { name:'G#', pos:4.64 }, { name:'A#', pos:5.64 },
];

const KB_MAP = {
  a:'C4', w:'C#4', s:'D4', e:'D#4', d:'E4',
  f:'F4', t:'F#4', g:'G4', y:'G#4', h:'A4', u:'A#4', j:'B4',
  k:'C5', o:'C#5', l:'D5', p:'D#5',
};

// ── x offset helpers ──────────────────────────────────────────────────────────
function leftOfOct(oct) {
  if (oct === 0) return 0;
  return 2 * WK_W + (oct - 1) * 7 * WK_W;
}

function whiteKeyX(note) {
  const name = note.slice(0, -1);
  const oct  = parseInt(note.slice(-1));
  const whites = ['C','D','E','F','G','A','B'];
  if (oct === 0) {
    return ['A','B'].indexOf(name) * WK_W;
  }
  return leftOfOct(oct) + whites.indexOf(name) * WK_W;
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
function makeKey(isBlack) {
  const k = document.createElement('div');
  k.className = isBlack ? 'piano-key bk' : 'piano-key wk';
  return k;
}

// ── 메인 빌더 ─────────────────────────────────────────────────────────────────
export function buildPiano(pianoEl, pianoWrapper, {
  showLabels = true,
  onPress,
  onRelease,
  navPrevEl    = null,
  navNextEl    = null,
  navLabelEl   = null,
  onOctaveChange = null,
} = {}) {
  pianoEl.innerHTML = '';
  pianoEl.style.cssText =
    'position:relative; display:inline-flex; align-items:flex-end; touch-action:none; user-select:none;';

  const keyEls = {};
  let viewOct  = 3;
  const VISIBLE_OCTS = 2;

  appendPartialOctave(pianoEl, keyEls, 0, ['A','B'], [{ name:'A#', pos:0.64 }], showLabels);
  for (let o = 1; o <= 7; o++) appendFullOctave(pianoEl, keyEls, o, showLabels);
  appendPartialOctave(pianoEl, keyEls, 8, ['C'], [], showLabels);

  // ── 옥타브 네비게이션 ─────────────────────────────────────────────────────
  const navPrev  = navPrevEl  ?? document.getElementById('btn-octave-down');
  const navNext  = navNextEl  ?? document.getElementById('btn-octave-up');
  const navLabel = navLabelEl ?? document.getElementById('octave-label');

  // ── 누름 상태 관리 ────────────────────────────────────────────────────────
  const pressedSet   = new Set();
  const wrongSet     = new Set(); // 틀리게 누른 건반 (지속 깜빡임)
  const expectedState = { note: null };

  function applyVisual(note) {
    const k = keyEls[note]; if (!k) return;
    const isB = k.classList.contains('bk');
    const isE = note === expectedState.note;
    const isP = pressedSet.has(note);
    const isW = wrongSet.has(note);

    // 틀린 건반: CSS 깜빡임 클래스 (스타일 override는 CSS가 담당)
    k.classList.toggle('key-wrong-blink', isW);

    if (isW) {
      k.style.transform = '';
      return; // CSS animation이 배경/그림자 담당
    }

    if (isP) {
      k.style.background = isB ? '#004d8a' : '#b8ddf8';
      k.style.transform  = isB ? 'translateY(4px) scaleY(0.97)'
                                : 'perspective(300px) rotateX(6deg)';
      k.style.boxShadow  = isB ? 'none'
                                : 'inset 0 6px 0 rgba(0,0,0,0.12), inset 0 -2px 0 rgba(255,255,255,0.4)';
    } else if (isE) {
      k.style.background = isB ? '#003a6e' : '#d0ebfa';
      k.style.boxShadow  = isB ? '0 0 10px 3px #0076CE'
                                : '0 0 10px 3px #0076CE, inset 0 -4px 0 rgba(0,118,206,0.3)';
      k.style.transform  = '';
    } else {
      k.style.background = isB ? '#1c1c1c' : '#f4efe6';
      k.style.transform  = '';
      k.style.boxShadow  = isB ? '' : 'inset 0 -4px 0 rgba(0,0,0,0.12)';
    }
  }

  // ── 옥타브 네비게이션 ─────────────────────────────────────────────────────
  function scrollToOct(oct) {
    const prevOct = viewOct;
    viewOct = Math.max(1, Math.min(6, oct));
    const octDelta = viewOct - prevOct;

    // Re-map any currently pressed keys to the equivalent note in the new octave
    if (octDelta !== 0 && pressedSet.size > 0) {
      const oldPressed = [...pressedSet];
      oldPressed.forEach(note => {
        const noteName = note.slice(0, -1);
        const noteOct  = parseInt(note.slice(-1));
        const newNote  = noteName + (noteOct + octDelta);
        pressedSet.delete(note);
        applyVisual(note);
        if (keyEls[newNote]) {
          pressedSet.add(newNote);
          applyVisual(newNote);
        }
      });
    }

    pianoWrapper.scrollLeft = leftOfOct(viewOct);
    if (navLabel) navLabel.textContent = `C${viewOct} ~ B${viewOct + VISIBLE_OCTS - 1} 영역`;
    navPrev && (navPrev.disabled = viewOct <= 1);
    navNext && (navNext.disabled = viewOct >= 6);
    onOctaveChange?.(viewOct, viewOct + VISIBLE_OCTS - 1);
  }

  navPrev?.addEventListener('click', () => scrollToOct(viewOct - 1));
  navNext?.addEventListener('click', () => scrollToOct(viewOct + 1));
  scrollToOct(viewOct);

  function press(note, { silent = false } = {}) {
    if (!keyEls[note] || pressedSet.has(note)) return;
    pressedSet.add(note);
    applyVisual(note);
    if (!silent) {
      audio.unlock();
      audio.playNote(note, 0.5);
    }
    onPress?.(note);
  }

  function release(note) {
    if (!pressedSet.has(note)) return;
    pressedSet.delete(note);
    // 뗄 때 자동으로 wrong 상태도 해제
    if (wrongSet.has(note)) {
      wrongSet.delete(note);
    }
    applyVisual(note);
    onRelease?.(note);
  }

  // ── 포인터 이벤트 ─────────────────────────────────────────────────────────
  const pointerNoteMap = {};

  pianoEl.addEventListener('pointerdown', e => {
    const k = e.target.closest('.piano-key'); if (!k) return;
    e.preventDefault();
    pianoEl.setPointerCapture(e.pointerId);
    pointerNoteMap[e.pointerId] = k.dataset.note;
    press(k.dataset.note);
  });

  pianoEl.addEventListener('pointermove', e => {
    if (!pointerNoteMap[e.pointerId]) return;
    const k = document.elementFromPoint(e.clientX, e.clientY)?.closest('.piano-key');
    const oldNote = pointerNoteMap[e.pointerId];
    const newNote = k?.dataset.note;
    if (newNote && newNote !== oldNote) {
      release(oldNote);
      pointerNoteMap[e.pointerId] = newNote;
      press(newNote);
    }
  });

  pianoEl.addEventListener('pointerup', e => {
    const note = pointerNoteMap[e.pointerId];
    if (note) { release(note); delete pointerNoteMap[e.pointerId]; }
  });

  pianoEl.addEventListener('pointercancel', e => {
    const note = pointerNoteMap[e.pointerId];
    if (note) { release(note); delete pointerNoteMap[e.pointerId]; }
  });

  // ── 키보드 이벤트 ─────────────────────────────────────────────────────────
  const kbHeld = new Set();
  function kbDown(e) {
    if (e.repeat || e.ctrlKey || e.metaKey) return;
    const note = KB_MAP[e.key.toLowerCase()]; if (!note || kbHeld.has(note)) return;
    kbHeld.add(note); press(note);
  }
  function kbUp(e) {
    const note = KB_MAP[e.key.toLowerCase()]; if (!note) return;
    kbHeld.delete(note); release(note);
  }
  document.addEventListener('keydown', kbDown);
  document.addEventListener('keyup',   kbUp);

  // ── 공개 API ──────────────────────────────────────────────────────────────
  return {
    // 외부 입력(MIDI 등)이 클릭/키보드와 동일한 경로(시각 피드백 + onPress/onRelease 콜백)를
    // 타도록 노출. silent:true면 자체 신시사이저 소리를 내지 않는다(실물 피아노가 이미 냄).
    press,
    release,
    setExpected(note) {
      const prev = expectedState.note;
      expectedState.note = note;
      if (prev && prev !== note) applyVisual(prev);
      applyVisual(note);
      const oct = parseInt(note?.slice(-1));
      if (!isNaN(oct) && (oct < viewOct || oct > viewOct + VISIBLE_OCTS - 1)) {
        scrollToOct(Math.max(1, oct - 1));
      }
    },
    clearExpected() {
      const prev = expectedState.note;
      expectedState.note = null;
      if (prev) applyVisual(prev);
    },
    flashCorrect(note) {
      const k = keyEls[note]; if (!k) return;
      const isB = k.classList.contains('bk');
      k.style.background = isB ? '#004d8a' : '#b8ddf8';
      k.style.boxShadow  = '0 0 12px 4px #0076CE';
      setTimeout(() => applyVisual(note), 350);
    },
    flashWrong(note) {
      const k = keyEls[note]; if (!k) return;
      const isB = k.classList.contains('bk');
      k.style.background = isB ? '#6a0000' : '#ffb0b0';
      k.style.boxShadow  = '0 0 10px 3px #ff4444';
      setTimeout(() => applyVisual(note), 350);
    },
    markWrong(note) {
      if (!keyEls[note]) return;
      wrongSet.add(note);
      applyVisual(note);
    },
    clearWrong(note) {
      wrongSet.delete(note);
      applyVisual(note);
    },
    clearHighlights() {
      expectedState.note = null;
      Object.keys(keyEls).forEach(n => applyVisual(n));
    },
    updateLabels(show) {
      pianoEl.querySelectorAll('.key-label').forEach(l => {
        l.style.display = show ? '' : 'none';
      });
    },

    // ── 참조 화살표: 높은/낮은음자리 중앙 건반 표시 ──────────────────────────
    setArrows(specs) {
      pianoEl.querySelectorAll('.piano-ref-arrow').forEach(a => a.remove());
      specs.forEach(({ note, color = '#FF4444', label = '' }) => {
        const k = keyEls[note];
        if (!k || k.classList.contains('bk')) return;  // only white keys

        k.style.position = 'relative';

        const arrow = document.createElement('div');
        arrow.className = 'piano-ref-arrow';
        arrow.style.cssText = `
          position:absolute; top:3px; left:0; right:0;
          display:flex; flex-direction:column; align-items:center; gap:1px;
          pointer-events:none; z-index:5;
        `;
        arrow.innerHTML = `
          <span style="
            font-size:7px; color:${color}; font-weight:900;
            font-family:system-ui; line-height:1; text-align:center;
            text-shadow:0 0 4px #000, 0 0 2px #000;
            white-space:nowrap;
          ">${label}</span>
          <svg width="10" height="7" viewBox="0 0 10 7" style="display:block;overflow:visible;">
            <polygon points="5,7 0,0 10,0" fill="${color}"
              style="filter:drop-shadow(0 0 2px ${color});"/>
          </svg>
        `;
        k.appendChild(arrow);
      });
    },

    // 옥타브 구역을 반투명 색 띠로 건반 위에 표시 (규칙1: 음높이=세로위치 학습용).
    // bands: [{ fromNote, toNote, color }] — fromNote/toNote는 흰 건반 이름(예: 'C4','B4').
    setZoneBands(bands) {
      pianoEl.querySelectorAll('.piano-zone-band').forEach(b => b.remove());
      bands.forEach(({ fromNote, toNote, color }) => {
        const x1 = whiteKeyX(fromNote);
        const x2 = whiteKeyX(toNote) + WK_W;
        const band = document.createElement('div');
        band.className = 'piano-zone-band';
        band.style.cssText = `
          position:absolute; top:0; left:${x1}px; width:${x2 - x1}px; height:${WK_H}px;
          background:${color}; pointer-events:none; z-index:4;
        `;
        pianoEl.appendChild(band);
      });
    },

    // 건반 위에 컬러 점 표시 (가온다 등 기준음 마킹)
    setDots(specs) {
      pianoEl.querySelectorAll('.piano-dot').forEach(d => d.remove());
      specs.forEach(({ note, color = '#FF4444' }) => {
        const k = keyEls[note];
        if (!k) return;
        k.style.position = 'relative';
        const dot = document.createElement('div');
        dot.className = 'piano-dot';
        dot.style.cssText = `
          position:absolute; bottom:20px; left:50%; transform:translateX(-50%);
          width:7px; height:7px; border-radius:50%;
          background:${color}; pointer-events:none; z-index:6;
          box-shadow:0 0 4px 1px ${color};
        `;
        k.appendChild(dot);
      });
    },

    destroy() {
      document.removeEventListener('keydown', kbDown);
      document.removeEventListener('keyup',   kbUp);
    },
  };
}

// ── 옥타브 그룹 헬퍼 ─────────────────────────────────────────────────────────
function appendFullOctave(pianoEl, keyEls, oct, showLabels) {
  const wrap = document.createElement('div');
  wrap.style.cssText =
    `position:relative; display:inline-flex; flex-shrink:0; width:${7 * WK_W}px; height:${WK_H}px;`;

  ['C','D','E','F','G','A','B'].forEach(name => {
    const note = name + oct;
    const k = makeWhiteKey(note, showLabels ? name : '');
    keyEls[note] = k;
    wrap.appendChild(k);
  });
  BLACK_DEFS.forEach(({ name, pos }) => {
    const note = name + oct;
    const k = makeBlackKey(note, pos);
    keyEls[note] = k;
    wrap.appendChild(k);
  });
  pianoEl.appendChild(wrap);
}

function appendPartialOctave(pianoEl, keyEls, oct, whites, blacks, showLabels) {
  const wrap = document.createElement('div');
  wrap.style.cssText =
    `position:relative; display:inline-flex; flex-shrink:0; width:${whites.length * WK_W}px; height:${WK_H}px;`;

  whites.forEach(name => {
    const note = name + oct;
    const k = makeWhiteKey(note, showLabels ? name + oct : '');
    keyEls[note] = k;
    wrap.appendChild(k);
  });
  blacks.forEach(({ name, pos }) => {
    const note = name + oct;
    const k = makeBlackKey(note, pos);
    keyEls[note] = k;
    wrap.appendChild(k);
  });
  pianoEl.appendChild(wrap);
}

function makeWhiteKey(note, label) {
  const k = document.createElement('div');
  k.className    = 'piano-key wk';
  k.dataset.note = note;
  k.style.cssText = `
    width:${WK_W}px; height:${WK_H}px; flex-shrink:0;
    background:#f4efe6;
    border-left:1px solid #c8c0b0; border-right:1px solid #c8c0b0; border-bottom:1px solid #a89878;
    border-radius:0 0 5px 5px;
    display:flex; align-items:flex-end; justify-content:center; padding-bottom:4px;
    cursor:pointer; transform-origin:top center;
    box-shadow:inset 0 -4px 0 rgba(0,0,0,0.12);
    transition:background .06s;
  `;
  if (label) {
    const s = document.createElement('span');
    s.className   = 'key-label';
    s.textContent = label;
    s.style.cssText = 'font-size:8px; color:#b0a090; pointer-events:none; font-family:system-ui;';
    k.appendChild(s);
  }
  return k;
}

// ── 라벨 옥타브 (튜토리얼 규칙0 전용) ────────────────────────────────────────
// 실제 88건반 레이아웃과 별개로, 딱 한 옥타브만 크게 그려서 흰건반엔 음이름+계이름,
// 검은건반엔 1~5 숫자+계이름을 건반 위에 직접 겹쳐 표시한다 — "피아노 한 옥타브 그림".
const OCT_BLACK = [
  { name: 'C#', pos: 0.64, label: '1' }, { name: 'D#', pos: 1.64, label: '2' },
  { name: 'F#', pos: 3.65, label: '3' }, { name: 'G#', pos: 4.64, label: '4' },
  { name: 'A#', pos: 5.64, label: '5' },
];
const SOLFEGE_MAP = {
  C: '도', D: '레', E: '미', F: '파', G: '솔', A: '라', B: '시',
  'C#': '도#', 'D#': '레#', 'F#': '파#', 'G#': '솔#', 'A#': '라#',
};

export function renderLabeledOctave(container, { oct = 4, onPress } = {}) {
  const LW = 100, LH = 260, BW = 58, BH = 158;
  container.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.style.cssText = 'position:relative; display:inline-flex; user-select:none; touch-action:none;';

  function flash(k, activeColor, restColor) {
    k.style.background = activeColor;
    setTimeout(() => { k.style.background = restColor; }, 220);
  }

  ['C', 'D', 'E', 'F', 'G', 'A', 'B'].forEach(name => {
    const note = name + oct;
    const k = document.createElement('div');
    k.dataset.note = note;
    k.style.cssText = `
      width:${LW}px; height:${LH}px; flex-shrink:0;
      background:#f4efe6; border:1px solid #c8c0b0; border-radius:0 0 12px 12px;
      display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
      padding-bottom:18px; gap:5px; cursor:pointer; transition:background .1s;
      box-shadow:inset 0 -6px 0 rgba(0,0,0,0.10);
    `;
    k.innerHTML = `<span style="font-size:32px;font-weight:800;color:#0076CE;font-family:system-ui;">${name}</span>
                    <span style="font-size:16px;color:#4A6080;font-family:system-ui;">${SOLFEGE_MAP[name]}</span>`;
    k.addEventListener('click', () => {
      audio.unlock(); audio.playNote(note, 0.5);
      flash(k, '#b8ddf8', '#f4efe6');
      onPress?.(note);
    });
    wrap.appendChild(k);
  });

  OCT_BLACK.forEach(({ name, pos, label }) => {
    const note = name + oct;
    const k = document.createElement('div');
    k.dataset.note = note;
    k.style.cssText = `
      position:absolute; top:0; left:${pos * LW - BW / 2}px;
      width:${BW}px; height:${BH}px; z-index:2;
      background:#1c1c1c; border-radius:0 0 10px 10px;
      display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
      padding-bottom:14px; gap:3px; cursor:pointer; transition:background .1s;
    `;
    k.innerHTML = `<span style="font-size:24px;font-weight:800;color:#fff;font-family:system-ui;">${label}</span>
                    <span style="font-size:12px;color:#cbd5e1;font-family:system-ui;">${SOLFEGE_MAP[name]}</span>`;
    k.addEventListener('click', () => {
      audio.unlock(); audio.playNote(note, 0.5);
      flash(k, '#004d8a', '#1c1c1c');
      onPress?.(note);
    });
    wrap.appendChild(k);
  });

  container.appendChild(wrap);
}

function makeBlackKey(note, pos) {
  const k = document.createElement('div');
  k.className    = 'piano-key bk';
  k.dataset.note = note;
  k.style.cssText = `
    position:absolute; top:0; left:${pos * WK_W - BK_W / 2}px;
    width:${BK_W}px; height:${BK_H}px; z-index:2;
    background:#1c1c1c;
    border-left:1px solid #111; border-right:1px solid #111; border-bottom:2px solid #000;
    border-radius:0 0 4px 4px;
    cursor:pointer; transform-origin:top center;
    transition:background .06s;
  `;
  return k;
}
