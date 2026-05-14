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
  const navPrev  = document.getElementById('btn-octave-down');
  const navNext  = document.getElementById('btn-octave-up');
  const navLabel = document.getElementById('octave-label');

  function scrollToOct(oct) {
    viewOct = Math.max(1, Math.min(6, oct));
    pianoWrapper.scrollLeft = leftOfOct(viewOct);
    if (navLabel) navLabel.textContent = `C${viewOct} ~ B${viewOct + VISIBLE_OCTS - 1} 영역`;
    navPrev && (navPrev.disabled = viewOct <= 1);
    navNext && (navNext.disabled = viewOct >= 6);
  }

  navPrev?.addEventListener('click', () => scrollToOct(viewOct - 1));
  navNext?.addEventListener('click', () => scrollToOct(viewOct + 1));
  scrollToOct(viewOct);

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
      k.style.background = isB ? '#c0521a' : '#ffd5bb';
      k.style.transform  = isB ? 'translateY(4px) scaleY(0.97)'
                                : 'perspective(300px) rotateX(6deg)';
      k.style.boxShadow  = isB ? 'none'
                                : 'inset 0 6px 0 rgba(0,0,0,0.18), inset 0 -2px 0 rgba(255,255,255,0.3)';
    } else if (isE) {
      k.style.background = isB ? '#4a3000' : '#fffacc';
      k.style.boxShadow  = isB ? '0 0 10px 3px #ffd700'
                                : '0 0 10px 3px #ffd700, inset 0 -4px 0 rgba(255,215,0,0.4)';
      k.style.transform  = '';
    } else {
      k.style.background = isB ? '#1c1c1c' : '#f4efe6';
      k.style.transform  = '';
      k.style.boxShadow  = isB ? '' : 'inset 0 -4px 0 rgba(0,0,0,0.12)';
    }
  }

  function press(note) {
    if (!keyEls[note] || pressedSet.has(note)) return;
    pressedSet.add(note);
    applyVisual(note);
    audio.unlock();
    audio.playNote(note, 0.5);
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
      k.style.background = isB ? '#3a7a3a' : '#c8f5cc';
      k.style.boxShadow  = '0 0 12px 4px #7BC67E';
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
