import { audio } from './audio.js';

// ── 건반 규격 ─────────────────────────────────────────────────────────────────
// 흰건반 폭을 36 -> 48 로 넓혔다. 36px 는 Apple 44pt / Material 48dp 터치 최소치에
// 못 미쳐 오타가 나기 쉬웠다. 나머지 좌표는 전부 이 두 값에서 파생되므로 건반 전체가
// 비례해서 커진다. 높이(148/92)는 이미 충분해서 그대로 두었다 — 가로 모드에서
// 화면 높이를 더 먹으면 악보 영역이 좁아진다.
//
// 검은건반 30px 는 여전히 44px 미만이지만, 이건 피아노 기하 자체의 제약이다(검은건반은
// 흰건반 사이에 끼므로 넓히면 흰건반을 침범한다). 전체 배율을 키우는 것이 실질적 완화책.
const WK_W = 48, WK_H = 148, BK_W = 30, BK_H = 92;

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
  // pannable: true면 마우스 왼쪽 클릭+드래그(데스크톱)/좌우 스와이프(태블릿·폰)로 건반
  // 뷰를 직접 옆으로 옮길 수 있게 한다(체험하기 연주 화면 전용 — 기존 화면들은 옥타브
  // 이동 버튼만 쓰므로 기본값 false로 기존 동작을 그대로 유지).
  pannable = false,
  // centerOnNotes: [note, note] — 지정하면 초기 스크롤 위치를 옥타브 단위 스냅 없이 이
  // 두 음의 정중앙으로 맞춘다(미지정 시 기존처럼 scrollToOct(viewOct)).
  centerOnNotes = null,
} = {}) {
  pianoEl.innerHTML = '';
  pianoEl.style.cssText =
    `position:relative; display:inline-flex; align-items:flex-end; user-select:none;` +
    (pannable ? ' touch-action:pan-x;' : ' touch-action:none;');
  if (pannable) {
    // .piano-wrapper(overflow:hidden)/.mini-piano(overflow-x:auto) 클래스 순서에 따라
    // CSS만으로는 어느 쪽이 이길지 불안정해서, 인라인 스타일로 확실히 스크롤 가능하게 한다.
    pianoWrapper.style.overflowX = 'auto';
    pianoWrapper.style.overflowY = 'hidden';
    pianoWrapper.style.scrollbarWidth = 'none'; // Firefox
    pianoWrapper.classList.add('piano-wrapper-pannable'); // ::-webkit-scrollbar 숨김용(style.css)

    // 건반 위 전용 드래그 손잡이 — 키 위에서 드래그해도 패닝은 되지만(아래 pointerdown
    // 임계값 로직), 그 전까지 짧게라도 press()가 한 번 불려서 음이 살짝 눌리는 게
    // 보였다("스크롤할 때 건반이 눌린다" 리포트). 건반이 전혀 없는 별도 손잡이 바를
    // 만들어 여기서 드래그하면 애초에 press() 호출 자체가 없어 그 문제가 안 생긴다.
    const handle = document.createElement('div');
    handle.className = 'piano-drag-handle';
    handle.innerHTML = '<span></span>';
    pianoWrapper.insertBefore(handle, pianoEl);

    let handleDragging = false, handleStartX = 0, handleStartScroll = 0;
    handle.addEventListener('pointerdown', e => {
      handleDragging = true;
      handleStartX = e.clientX;
      handleStartScroll = pianoWrapper.scrollLeft;
      handle.setPointerCapture(e.pointerId);
    });
    handle.addEventListener('pointermove', e => {
      if (!handleDragging) return;
      pianoWrapper.scrollLeft = handleStartScroll - (e.clientX - handleStartX);
    });
    handle.addEventListener('pointerup',     () => { handleDragging = false; });
    handle.addEventListener('pointercancel', () => { handleDragging = false; });
  }

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
  const paintedNotes  = {}; // note -> color, 건반 자체를 통째로 칠할 때(규칙3 화음 등)

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
      const paint = paintedNotes[note];
      k.style.background = paint ?? (isB ? '#1c1c1c' : '#f4efe6');
      k.style.transform  = '';
      k.style.boxShadow  = isB ? '' : 'inset 0 -4px 0 rgba(0,0,0,0.12)';
    }
  }

  // ── 옥타브 네비게이션 ─────────────────────────────────────────────────────
  // (예전엔 옥타브 스크롤 시 누르고 있던 건반을 새 옥타브의 같은 음이름 건반으로
  // "재매핑"했었는데, setExpected()가 다음 음을 보여주려고 자동으로 스크롤을 걸 때도
  // 이 로직이 같이 발동해서 — 예: 규칙2에서 솔(G4)을 누르는 순간 다음 기대음(도, C5)이
  // 화면 밖이라 자동 스크롤되면서 방금 누른 G4가 엉뚱하게 G5를 누른 것처럼 보이는
  // 버그가 있었다. 재매핑은 실제로 쓸모보다 혼란이 커서 통째로 제거.)
  function scrollToOct(oct) {
    viewOct = Math.max(1, Math.min(6, oct));
    pianoWrapper.scrollLeft = leftOfOct(viewOct);
    if (navLabel) navLabel.textContent = `C${viewOct} ~ B${viewOct + VISIBLE_OCTS - 1} 영역`;
    navPrev && (navPrev.disabled = viewOct <= 1);
    navNext && (navNext.disabled = viewOct >= 6);
    onOctaveChange?.(viewOct, viewOct + VISIBLE_OCTS - 1);
  }

  navPrev?.addEventListener('click', () => scrollToOct(viewOct - 1));
  navNext?.addEventListener('click', () => scrollToOct(viewOct + 1));

  // 지정된 두 음(예: 가온다 C4·한 옥타브 아래 도 C3)의 정중앙이 뷰 중앙에 오도록 —
  // scrollToOct처럼 옥타브 경계에 딱 맞추지 않고 픽셀 단위로 자유롭게 맞춘다.
  function centerOn(notes) {
    const xs = notes.map(whiteKeyX).filter(x => typeof x === 'number' && !isNaN(x));
    if (!xs.length) { scrollToOct(viewOct); return; }
    const mid = (Math.min(...xs) + Math.max(...xs) + WK_W) / 2;
    pianoWrapper.scrollLeft = Math.max(0, mid - pianoWrapper.clientWidth / 2);
    const octs = notes.map(n => parseInt(n.slice(-1))).filter(o => !isNaN(o));
    if (octs.length && navLabel) navLabel.textContent = `C${Math.min(...octs)} ~ B${Math.max(...octs)} 영역`;
  }

  if (centerOnNotes) centerOn(centerOnNotes); else scrollToOct(viewOct);

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
  // 데스크톱 마우스 드래그로 화면 이동(pannable 전용) — 터치는 위 touch-action:pan-x +
  // 브라우저 네이티브 스와이프 스크롤로 처리되지만(overflow-x:auto), 마우스는 클릭+드래그로
  // 스크롤하는 동작이 브라우저 기본 기능이 아니라서 여기서 직접 흉내낸다. 이동 임계값을
  // 넘기기 전까진 기존처럼 그냥 클릭(음 재생)으로 남고, 넘기는 순간 눌려있던 음은
  // 취소하고 그 뒤로는 전부 화면 이동으로만 처리한다(글리산도 슬라이드와 동시에 안 겹치게).
  const PAN_THRESHOLD = 6;
  let panState = null; // { pointerId, startX, startScroll, dragging, note }

  pianoEl.addEventListener('pointerdown', e => {
    const k = e.target.closest('.piano-key'); if (!k) return;
    e.preventDefault();
    pianoEl.setPointerCapture(e.pointerId);
    if (pannable && e.pointerType === 'mouse') {
      panState = { pointerId: e.pointerId, startX: e.clientX, startScroll: pianoWrapper.scrollLeft, dragging: false, note: k.dataset.note };
    }
    pointerNoteMap[e.pointerId] = k.dataset.note;
    press(k.dataset.note);
  });

  pianoEl.addEventListener('pointermove', e => {
    if (panState && panState.pointerId === e.pointerId) {
      const dx = e.clientX - panState.startX;
      if (!panState.dragging && Math.abs(dx) > PAN_THRESHOLD) {
        panState.dragging = true;
        release(panState.note); // 드래그로 확정되는 순간 처음 눌렸던 음은 취소
        delete pointerNoteMap[e.pointerId];
      }
      if (panState.dragging) {
        pianoWrapper.scrollLeft = panState.startScroll - dx;
        return; // 패닝 중엔 건반 슬라이드(글리산도) 판정을 하지 않음
      }
    }
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
    if (panState && panState.pointerId === e.pointerId) panState = null;
    const note = pointerNoteMap[e.pointerId];
    if (note) { release(note); delete pointerNoteMap[e.pointerId]; }
  });

  pianoEl.addEventListener('pointercancel', e => {
    if (panState && panState.pointerId === e.pointerId) panState = null;
    const note = pointerNoteMap[e.pointerId];
    if (note) { release(note); delete pointerNoteMap[e.pointerId]; }
  });

  // ── 키보드 이벤트 ─────────────────────────────────────────────────────────
  const kbHeld = new Set();
  function kbDown(e) {
    if (e.repeat || e.ctrlKey || e.metaKey) return;
    // 닉네임/제목 입력 등 텍스트 필드에 포커스가 있을 땐 글자 입력(a, s, d, f...)이
    // KB_MAP과 겹쳐 의도치 않게 피아노 소리가 나던 버그 — 입력 중엔 건반 매핑을 끈다.
    const active = document.activeElement;
    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) return;
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

    // 존 경계에 세로 구분선 표시(예: 왼손/오른손이 만나는 겹치는 지점을 회색 선으로
    // 표시) — note의 왼쪽 끝(그 앞 건반과의 경계)에 선을 긋는다.
    setZoneDivider(note, color = '#999') {
      pianoEl.querySelectorAll('.piano-zone-divider').forEach(d => d.remove());
      const x = whiteKeyX(note);
      const line = document.createElement('div');
      line.className = 'piano-zone-divider';
      line.style.cssText = `
        position:absolute; top:0; left:${x}px; width:2px; height:${WK_H}px;
        background:${color}; pointer-events:none; z-index:5;
      `;
      pianoEl.appendChild(line);
    },

    // 건반 위에 컬러 점 표시 (가온다 등 기준음 마킹). label(이모지 등)을 주면 점 위에
    // 작게 같이 띄운다(예: 왼손/오른손 기준점 구분용).
    setDots(specs) {
      pianoEl.querySelectorAll('.piano-dot').forEach(d => d.remove());
      specs.forEach(({ note, color = '#FF4444', label = '' }) => {
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
        if (label) {
          const lbl = document.createElement('div');
          lbl.className = 'piano-dot-label';
          lbl.textContent = label;
          lbl.style.cssText = `
            position:absolute; bottom:12px; left:50%; transform:translateX(-50%);
            font-size:14px; line-height:1; pointer-events:none; z-index:6;
            filter:drop-shadow(0 0 2px #fff) drop-shadow(0 0 2px #fff);
          `;
          k.appendChild(lbl);
        }
        k.appendChild(dot);
      });
    },

    // 건반 자체를 색으로 칠함 (동그라미 대신 건반 전체를 칠하고 싶을 때 — 규칙3 화음 등).
    // specs: [{ note, color }]
    paintKeys(specs) {
      Object.keys(paintedNotes).forEach(n => delete paintedNotes[n]);
      specs.forEach(({ note, color }) => { paintedNotes[note] = color; });
      Object.keys(keyEls).forEach(n => applyVisual(n));
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

// ═══════════════════════════════════════════════════════════════════════════════
// ── 사진(그랜드피아노) 기반 인터랙티브 건반 ──────────────────────────────────────
// 규칙0 이후 튜토리얼에서 사용 — 합성 건반 대신 실사(튜토.png) 위에 건반 히트존을
// 정확한 픽셀 좌표로 겹쳐서, 사진 그대로의 모양을 유지한 채 클릭/터치로 소리가 나게 한다.
// 좌표는 designKit/튜토.png(1408x768, 흰건반 A0~C8 = 52개)를 직접 측정해서 얻은 값 —
// 사진이 바뀌면 이 상수들도 다시 재야 한다.
const PHOTO_KEYS_LEFT   = 46.0;  // A0(첫 흰건반) 중심 x — 원본(=건반 크롭) 픽셀 기준
const PHOTO_WK_W        = 25.28; // 흰건반 폭(px, 원본 기준)
const PHOTO_IMG_W       = 1408;  // 원본/건반 크롭 이미지의 가로 폭(px)
const PHOTO_BK_W_RATIO  = 0.56;  // 검은건반 폭 = 흰건반 폭의 이 비율

function photoWhiteKeyList() {
  const list = [['A', 0], ['B', 0]];
  for (let oct = 1; oct <= 7; oct++) for (const n of ['C','D','E','F','G','A','B']) list.push([n, oct]);
  list.push(['C', 8]);
  return list;
}

// note('C4' 등) -> 원본 이미지 기준 중심 x(px) + 검은건반 여부. 못 찾으면 null.
function photoKeyCenterPx(note) {
  const name = note.slice(0, -1);
  const oct  = parseInt(note.slice(-1));
  const whites = photoWhiteKeyList();
  const idx = whites.findIndex(([n, o]) => n === name && o === oct);
  if (idx >= 0) return { centerPx: PHOTO_KEYS_LEFT + idx * PHOTO_WK_W, isBlack: false };
  const def = BLACK_DEFS.find(d => d.name === name);
  if (!def) return null;
  const cIdx = whites.findIndex(([n, o]) => n === 'C' && o === oct);
  if (cIdx < 0) return null;
  const groupCenter = PHOTO_KEYS_LEFT + cIdx * PHOTO_WK_W; // 그 옥타브 C키 중심
  return { centerPx: groupCenter + def.pos * PHOTO_WK_W, isBlack: true };
}

function photoAllKeyNames() {
  const out = [];
  photoWhiteKeyList().forEach(([n, o]) => out.push(`${n}${o}`));
  for (let oct = 1; oct <= 7; oct++) BLACK_DEFS.forEach(d => out.push(`${d.name}${oct}`));
  return out;
}

// container에 사진 배경 + 건반 히트존을 깐다.
//   imgSrc          — 배경 사진(건반 전체 크롭 또는 옥타브 클로즈업 크롭)
//   cropLeftPx/cropWidthPx — 그 imgSrc가 원본 이미지에서 잘라낸 x 구간(px, 원본 기준).
//                            기본값(0, 1408)은 "건반 전체가 그대로 담긴" 크롭(tut-piano-keys.jpg)용.
//                            옥타브 클로즈업처럼 일부만 잘라낸 이미지를 쓸 땐 그 구간을 넘겨야
//                            건반 위치가 사진과 정확히 겹친다.
//   notes            — null이면(기본) 크롭 범위 안에 들어오는 건반을 전부 자동으로 씀.
//                       특정 건반만 필요하면(옥타브 클로즈업 등) 배열로 지정.
// buildPiano()와 최대한 같은 반환 API(press/release/setExpected/flashCorrect/flashWrong/
// setDots/setZoneBands/setArrows/destroy)를 제공해 TUT_PAGES 쪽 코드를 그대로 재사용한다.
export function buildPhotoPiano(container, imgSrc, {
  onPress, onRelease,
  cropLeftPx = 0, cropWidthPx = PHOTO_IMG_W,
  notes = null,
  whiteHeightPct = 84, blackHeightPct = 58,
} = {}) {
  container.innerHTML = '';
  container.style.cssText = `
    position:relative; width:100%; height:100%; overflow:hidden;
    touch-action:none; user-select:none; background:#000;
  `;
  const imgEl = document.createElement('img');
  imgEl.src = imgSrc;
  imgEl.alt = '';
  imgEl.draggable = false;
  imgEl.style.cssText =
    'position:absolute; inset:0; width:100%; height:100%; object-fit:cover; pointer-events:none; display:block;';
  container.appendChild(imgEl);

  const keyEls = {};
  const allNotes = notes ?? photoAllKeyNames();
  allNotes.forEach(note => {
    const info = photoKeyCenterPx(note);
    if (!info) return;
    const wPx = info.isBlack ? PHOTO_WK_W * PHOTO_BK_W_RATIO : PHOTO_WK_W;
    const leftPx = info.centerPx - wPx / 2 - cropLeftPx;
    if (leftPx + wPx < 0 || leftPx > cropWidthPx) return; // 지금 크롭 범위 밖이면 건너뜀
    const k = document.createElement('div');
    k.className    = 'photo-piano-key' + (info.isBlack ? ' bk' : ' wk');
    k.dataset.note = note;
    k.style.cssText = `
      position:absolute; top:0; left:${leftPx / cropWidthPx * 100}%; width:${wPx / cropWidthPx * 100}%;
      height:${info.isBlack ? blackHeightPct : whiteHeightPct}%;
      z-index:${info.isBlack ? 2 : 1}; cursor:pointer;
      background:rgba(0,0,0,0); box-shadow:none; transition:background .1s, box-shadow .1s;
    `;
    keyEls[note] = k;
    container.appendChild(k);
  });

  // ── 상태 + 시각 피드백 — 색 자체(건반)는 사진 그대로 두고, 반투명 오버레이로만 표시 ──
  const pressedSet = new Set();
  const expectedState = { note: null };

  function applyVisual(note) {
    const k = keyEls[note]; if (!k) return;
    const isE = note === expectedState.note;
    const isP = pressedSet.has(note);
    if (isP) {
      k.style.background = 'rgba(0,118,206,0.55)';
      k.style.boxShadow   = 'inset 0 0 0 2px rgba(255,255,255,0.6)';
    } else if (isE) {
      k.style.background = 'rgba(0,118,206,0.3)';
      k.style.boxShadow   = '0 0 12px 3px rgba(0,118,206,0.85)';
    } else {
      k.style.background = 'rgba(0,0,0,0)';
      k.style.boxShadow   = 'none';
    }
  }

  function press(note, { silent = false } = {}) {
    if (!keyEls[note] || pressedSet.has(note)) return;
    pressedSet.add(note);
    applyVisual(note);
    if (!silent) { audio.unlock(); audio.playNote(note, 0.5); }
    onPress?.(note);
  }
  function release(note) {
    if (!pressedSet.has(note)) return;
    pressedSet.delete(note);
    applyVisual(note);
    onRelease?.(note);
  }

  const pointerNoteMap = {};
  container.addEventListener('pointerdown', e => {
    const k = e.target.closest('.photo-piano-key'); if (!k) return;
    e.preventDefault();
    container.setPointerCapture(e.pointerId);
    pointerNoteMap[e.pointerId] = k.dataset.note;
    press(k.dataset.note);
  });
  container.addEventListener('pointerup', e => {
    const note = pointerNoteMap[e.pointerId];
    if (note) { release(note); delete pointerNoteMap[e.pointerId]; }
  });
  container.addEventListener('pointercancel', e => {
    const note = pointerNoteMap[e.pointerId];
    if (note) { release(note); delete pointerNoteMap[e.pointerId]; }
  });

  // ── 키보드(QWERTY) 입력도 동일하게 지원 — 텍스트 입력 중엔 꺼짐(피아노.js의 kbDown과 동일 원칙) ──
  const kbHeld = new Set();
  function kbDown(e) {
    if (e.repeat || e.ctrlKey || e.metaKey) return;
    const active = document.activeElement;
    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) return;
    const note = KB_MAP[e.key.toLowerCase()]; if (!note || !keyEls[note] || kbHeld.has(note)) return;
    kbHeld.add(note); press(note);
  }
  function kbUp(e) {
    const note = KB_MAP[e.key.toLowerCase()]; if (!note) return;
    kbHeld.delete(note); release(note);
  }
  document.addEventListener('keydown', kbDown);
  document.addEventListener('keyup', kbUp);

  return {
    press, release,
    setExpected(note) {
      const prev = expectedState.note;
      expectedState.note = note;
      if (prev && prev !== note) applyVisual(prev);
      applyVisual(note);
    },
    clearExpected() {
      const prev = expectedState.note;
      expectedState.note = null;
      if (prev) applyVisual(prev);
    },
    flashCorrect(note) {
      const k = keyEls[note]; if (!k) return;
      k.style.background = 'rgba(47,170,110,0.6)';
      k.style.boxShadow   = '0 0 14px 4px rgba(47,170,110,0.9)';
      setTimeout(() => applyVisual(note), 350);
    },
    flashWrong(note) {
      const k = keyEls[note]; if (!k) return;
      k.style.background = 'rgba(230,69,69,0.6)';
      k.style.boxShadow   = '0 0 12px 3px rgba(230,69,69,0.9)';
      setTimeout(() => applyVisual(note), 350);
    },
    // 규칙0 클로즈업 전용 — 흰건반엔 음이름+계이름(도레미…), 검은건반엔 1~5+계이름을
    // 사진 건반 위에 직접 겹쳐 보여준다(renderLabeledOctave()의 라벨 방식과 동일 규칙).
    labelOctave(oct) {
      container.querySelectorAll('.photo-piano-label').forEach(l => l.remove());
      const whites = ['C','D','E','F','G','A','B'];
      const blackLabels = { 'C#':'1', 'D#':'2', 'F#':'3', 'G#':'4', 'A#':'5' };
      whites.forEach(name => {
        const k = keyEls[name + oct]; if (!k) return;
        const lbl = document.createElement('div');
        lbl.className = 'photo-piano-label';
        lbl.style.cssText = `
          position:absolute; bottom:6%; left:50%; transform:translateX(-50%);
          display:flex; flex-direction:column; align-items:center; gap:2px;
          pointer-events:none; z-index:6; text-align:center;`;
        lbl.innerHTML = `
          <span style="font-size:22px;font-weight:800;color:#0076CE;font-family:system-ui;
            text-shadow:0 0 3px #fff,0 0 6px #fff;">${name}</span>
          <span style="font-size:12px;font-weight:700;color:#0060AA;font-family:system-ui;
            text-shadow:0 0 3px #fff,0 0 6px #fff;">${SOLFEGE_MAP[name]}</span>`;
        k.appendChild(lbl);
      });
      Object.entries(blackLabels).forEach(([name, num]) => {
        const k = keyEls[name + oct]; if (!k) return;
        const lbl = document.createElement('div');
        lbl.className = 'photo-piano-label';
        lbl.style.cssText = `
          position:absolute; bottom:8%; left:50%; transform:translateX(-50%);
          display:flex; flex-direction:column; align-items:center; gap:1px;
          pointer-events:none; z-index:6; text-align:center;`;
        lbl.innerHTML = `
          <span style="font-size:15px;font-weight:800;color:#fff;font-family:system-ui;
            text-shadow:0 0 3px #000;">${num}</span>
          <span style="font-size:9px;font-weight:700;color:#eee;font-family:system-ui;
            text-shadow:0 0 3px #000;">${SOLFEGE_MAP[name]}</span>`;
        k.appendChild(lbl);
      });
    },
    // 옥타브별 색상을 실제 건반 위치 위에 반투명하게 직접 칠한다(규칙1).
    setZoneBands(bands) {
      container.querySelectorAll('.photo-piano-zone').forEach(b => b.remove());
      bands.forEach(({ fromNote, toNote, color }) => {
        const kFrom = keyEls[fromNote], kTo = keyEls[toNote];
        if (!kFrom || !kTo) return;
        const left  = parseFloat(kFrom.style.left);
        const right = parseFloat(kTo.style.left) + parseFloat(kTo.style.width);
        const band = document.createElement('div');
        band.className = 'photo-piano-zone';
        band.style.cssText = `
          position:absolute; left:${left}%; width:${right - left}%; top:0; height:100%;
          background:${color}; pointer-events:none; z-index:0;
        `;
        container.appendChild(band);
      });
    },
    setDots(specs) {
      container.querySelectorAll('.photo-piano-dot, .photo-piano-dot-label').forEach(d => d.remove());
      specs.forEach(({ note, color = '#FF4444', label = '' }) => {
        const k = keyEls[note]; if (!k) return;
        const dot = document.createElement('div');
        dot.className = 'photo-piano-dot';
        dot.style.cssText = `
          position:absolute; bottom:10%; left:50%; transform:translateX(-50%);
          width:8px; height:8px; border-radius:50%; background:${color};
          box-shadow:0 0 5px 1px ${color}; pointer-events:none; z-index:6;
        `;
        k.appendChild(dot);
        if (label) {
          const lbl = document.createElement('div');
          lbl.className = 'photo-piano-dot-label';
          lbl.textContent = label;
          lbl.style.cssText = `
            position:absolute; bottom:22%; left:50%; transform:translateX(-50%);
            font-size:12px; line-height:1; pointer-events:none; z-index:6;
            filter:drop-shadow(0 0 2px #fff) drop-shadow(0 0 2px #fff);
          `;
          k.appendChild(lbl);
        }
      });
    },
    setArrows(specs) {
      container.querySelectorAll('.photo-piano-arrow').forEach(a => a.remove());
      specs.forEach(({ note, color = '#FF4444', label = '' }) => {
        const k = keyEls[note];
        if (!k || k.classList.contains('bk')) return;
        const arrow = document.createElement('div');
        arrow.className = 'photo-piano-arrow';
        arrow.style.cssText = `
          position:absolute; top:2%; left:0; right:0;
          display:flex; flex-direction:column; align-items:center; gap:1px;
          pointer-events:none; z-index:5;
        `;
        arrow.innerHTML = `
          <span style="font-size:7px; color:${color}; font-weight:900; font-family:system-ui;
            line-height:1; text-shadow:0 0 4px #000, 0 0 2px #000; white-space:nowrap;">${label}</span>
          <svg width="10" height="7" viewBox="0 0 10 7" style="display:block;overflow:visible;">
            <polygon points="5,7 0,0 10,0" fill="${color}" style="filter:drop-shadow(0 0 2px ${color});"/>
          </svg>`;
        k.appendChild(arrow);
      });
    },
    destroy() {
      document.removeEventListener('keydown', kbDown);
      document.removeEventListener('keyup', kbUp);
    },
  };
}
