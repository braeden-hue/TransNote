import { SAMPLES, BEAT_COLORS }          from './samples.js';
import { audio }                           from './audio.js';
import { renderNotation, renderMiniNotation, renderGrandStaff } from './notation.js';
import { buildPiano }                      from './piano.js';
import { loadAll, saveNotation, deleteNotation, generateId } from './storage.js';

// ── 피아노 기준 화살표 — 가온다(C4, Middle C) ─────────────────────────────────
const REF_ARROWS = [
  { note: 'C4', color: '#0076CE', label: '가온다' },
];

// ── 전역 상태 ──────────────────────────────────────────────────────────────────
const state = {
  screen:            'tutorial',
  convertResult:     null,
  playNotation:      null,
  playNoteIdx:       -1,
  playedIdx:         -1,
  playCancel:        null,
  countdownTimer:    null,
  tutorialPianoCtrl: null,
  playPianoCtrl:     null,
  playNotationCtrl:  null,  // renderPlay 에서 반환된 ctrl
};

// ── 악보 화살표 nav 헬퍼 ──────────────────────────────────────────────────────
const notationNavUpdate = { tutorial: () => {}, convert: () => {}, play: () => {} };

function makeNotationNav(containerId, prevId, nextId) {
  const c = document.getElementById(containerId);
  const p = document.getElementById(prevId);
  const n = document.getElementById(nextId);
  if (!c || !p || !n) return () => {};

  const step = () => Math.max(c.clientWidth * 0.8, 160);

  function update() {
    p.disabled = c.scrollLeft <= 1;
    n.disabled = c.scrollLeft + c.clientWidth >= c.scrollWidth - 1;
  }
  p.addEventListener('click', () => {
    c.scrollLeft -= step();
    setTimeout(update, 350);
  });
  n.addEventListener('click', () => {
    c.scrollLeft += step();
    setTimeout(update, 350);
  });
  c.addEventListener('scroll', update);

  // Mouse wheel navigates the notation panel instead of the page
  const navWrap = c.closest('.notation-nav-wrap') ?? c.parentElement;
  navWrap.addEventListener('wheel', e => {
    e.preventDefault();
    e.stopPropagation();
    if (e.deltaY > 0) {
      c.scrollLeft += step();
    } else {
      c.scrollLeft -= step();
    }
    setTimeout(update, 350);
  }, { passive: false });

  return update;
}

// ── 풀페이지 스크롤 ─────────────────────────────────────────────────────────────
const SCREEN_ORDER = ['tutorial', 'convert', 'play', 'library'];
let wheelLocked = false;

function navigate(name, { instant = false } = {}) {
  if (wheelLocked && !instant) return;

  const prevName = state.screen;
  const prevIdx  = SCREEN_ORDER.indexOf(prevName);
  const nextIdx  = SCREEN_ORDER.indexOf(name);
  if (nextIdx < 0) return;
  const goingDown = nextIdx > prevIdx;

  // 모든 화면 초기화
  document.querySelectorAll('.screen').forEach(s => {
    s.classList.remove('active', 'above');
  });

  // 이전 화면: 위쪽으로 사라지거나(아래 스크롤), 아래로 복귀(위 스크롤)
  const prevEl = document.getElementById('screen-' + prevName);
  if (prevEl && prevName !== name) {
    if (goingDown) prevEl.classList.add('above');
    // 위로 스크롤 시 이전 화면(above였던 것)은 이미 translateY(100%)로 복귀됨
  }

  // 다음 화면 활성화
  const nextEl = document.getElementById('screen-' + name);
  if (nextEl) {
    // 아래서 올라오는 경우 below 상태(translateY 100%)에서 시작 — 기본값이므로 그냥 active
    nextEl.classList.add('active');
  }

  // nav-tab 업데이트
  document.querySelectorAll('.nav-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.screen === name));

  // page-dots 업데이트
  document.querySelectorAll('.page-dot').forEach(d =>
    d.classList.toggle('active', d.dataset.screen === name));

  state.screen = name;
  if (name === 'play')    initPlayScreen();
  if (name === 'library') initLibraryScreen();
}

// ── 마우스 휠 인터셉트 ─────────────────────────────────────────────────────────
// 발표/데모용: 휠 한 번 → 섹션 전환 (섹션 내 스크롤은 스크롤바·드래그로)
window.addEventListener('wheel', e => {
  e.preventDefault(); // 항상 기본 스크롤 막음
  if (wheelLocked) return;

  const goingDown = e.deltaY > 0;
  const cur  = SCREEN_ORDER.indexOf(state.screen);
  const next = goingDown ? cur + 1 : cur - 1;
  if (next >= 0 && next < SCREEN_ORDER.length) {
    wheelLocked = true;
    navigate(SCREEN_ORDER[next]);
    setTimeout(() => { wheelLocked = false; }, 800);
  }
}, { passive: false });

// ── Toast ──────────────────────────────────────────────────────────────────────
function toast(msg, ms = 2600) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('visible');
  setTimeout(() => el.classList.remove('visible'), ms);
}

// ── 카운트다운 바 ───────────────────────────────────────────────────────────────
function showCountdown(durationMs, onDone) {
  clearCountdown();
  const bar  = document.getElementById('countdown-bar');
  const wrap = document.getElementById('countdown-wrap');
  if (!bar || !wrap) { onDone?.(); return; }
  wrap.style.display = 'block';
  bar.style.transition = 'none';
  bar.style.width = '100%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      bar.style.transition = `width ${durationMs}ms linear`;
      bar.style.width = '0%';
    });
  });
  state.countdownTimer = setTimeout(() => {
    wrap.style.display = 'none';
    onDone?.();
  }, durationMs);
}

function clearCountdown() {
  if (state.countdownTimer) { clearTimeout(state.countdownTimer); state.countdownTimer = null; }
  const wrap = document.getElementById('countdown-wrap');
  const bar  = document.getElementById('countdown-bar');
  if (wrap) wrap.style.display = 'none';
  if (bar)  { bar.style.transition = 'none'; bar.style.width = '100%'; }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tutorial
// ═══════════════════════════════════════════════════════════════════════════════
function initTutorial() {
  // 규칙 1: 세로 위치 = 음 높이
  renderMiniNotation(document.getElementById('notation-demo-1'), [
    { pitch:'C5', duration:1, beat:1 },
    { pitch:'G4', duration:1, beat:2 },
    { pitch:'C4', duration:1, beat:3 },
  ], { unitW:72, cellH:36 });

  // 규칙 2: 셀 너비 = 음 길이
  renderMiniNotation(document.getElementById('notation-demo-2'), [
    { pitch:'C4', duration:0.5, beat:3 },
    { pitch:'C4', duration:1,   beat:1 },
    { pitch:'C4', duration:2,   beat:2 },
  ], { unitW:72, cellH:36 });

  // 규칙 3: 박자 색 클릭
  document.querySelectorAll('.beat-cell').forEach(cell => {
    cell.addEventListener('click', () => { audio.unlock(); audio.playNote('C4', 0.4); });
  });

  // ── 기본음 키보드 체험 섹션 ────────────────────────────────────────────────
  const KB_NOTES = [
    { key: 'A', pitch: 'C4', label: 'C', solfege: '도' },
    { key: 'S', pitch: 'D4', label: 'D', solfege: '레' },
    { key: 'D', pitch: 'E4', label: 'E', solfege: '미' },
    { key: 'F', pitch: 'F4', label: 'F', solfege: '파' },
    { key: 'G', pitch: 'G4', label: 'G', solfege: '솔' },
    { key: 'H', pitch: 'A4', label: 'A', solfege: '라' },
    { key: 'J', pitch: 'B4', label: 'B', solfege: '시' },
  ];
  const NOTE_COLORS = { C:'#0076CE', D:'#5BB8F5', E:'#A8D5F5', F:'#3A9EE0', G:'#0076CE', A:'#5BB8F5', B:'#A8D5F5' };
  const kbCellMap = {};

  const tryoutEl = document.getElementById('kb-tryout');
  if (tryoutEl) {
    const grid = document.createElement('div');
    grid.className = 'kb-note-grid';
    KB_NOTES.forEach(({ key, pitch, label, solfege }) => {
      const cell = document.createElement('div');
      cell.className = 'kb-note-cell';
      cell.style.setProperty('--nc', NOTE_COLORS[label]);
      cell.innerHTML = `
        <span class="kb-note-symbol">${label}</span>
        <span class="kb-note-solfege">${solfege}</span>
        <kbd class="kb-key-badge">${key}</kbd>`;
      cell.addEventListener('click', () => {
        audio.unlock(); audio.playNote(pitch, 0.4);
        cell.classList.add('active');
        setTimeout(() => cell.classList.remove('active'), 280);
      });
      kbCellMap[pitch] = cell;
      grid.appendChild(cell);
    });
    tryoutEl.appendChild(grid);

    const kbTryMap = { a:'C4', s:'D4', d:'E4', f:'F4', g:'G4', h:'A4', j:'B4' };
    document.addEventListener('keydown', e => {
      if (e.repeat) return;
      const p = kbTryMap[e.key.toLowerCase()];
      if (p && kbCellMap[p]) kbCellMap[p].classList.add('active');
    });
    document.addEventListener('keyup', e => {
      const p = kbTryMap[e.key.toLowerCase()];
      if (p && kbCellMap[p]) kbCellMap[p].classList.remove('active');
    });
  }

  // ── 인터랙티브 미리보기 (순차 진행) ────────────────────────────────────────
  const previewNotes = SAMPLES[0].notes.slice(0, 7);
  let tutExpIdx     = 0;   // 다음에 눌러야 할 음 인덱스
  let tutAdvTimer   = null; // 자동 진행 타이머

  function rerenderTutorial(hlIdx, expIdx) {
    renderNotation(document.getElementById('tutorial-notation'), previewNotes, {
      highlightIdx: hlIdx,
      expectedIdx:  expIdx,
      onNoteClick(idx, note) {
        clearTimeout(tutAdvTimer);
        audio.unlock(); audio.playNote(note.pitch, 0.4);
        tutExpIdx = idx;
        rerenderTutorial(idx, -1);
        state.tutorialPianoCtrl?.clearExpected();
        tutAdvTimer = setTimeout(() => {
          tutAdvTimer = null;
          const next = (idx + 1) % previewNotes.length;
          tutExpIdx = next;
          rerenderTutorial(-1, next);
          state.tutorialPianoCtrl?.setExpected(previewNotes[next].pitch);
        }, 700);
      },
    });
    notationNavUpdate.tutorial();
  }
  rerenderTutorial(-1, 0);

  const pianoEl      = document.getElementById('tutorial-piano');
  const pianoWrapper = document.getElementById('tutorial-piano-wrapper');
  state.tutorialPianoCtrl = buildPiano(pianoEl, pianoWrapper, {
    showLabels: true,
    navPrevEl:  document.getElementById('tutorial-oct-down'),
    navNextEl:  document.getElementById('tutorial-oct-up'),
    navLabelEl: document.getElementById('tutorial-oct-label'),
    onPress(note) {
      if (tutAdvTimer !== null) return;   // 진행 대기 중엔 무시
      const expNote = previewNotes[tutExpIdx];
      if (!expNote) return;

      if (note === expNote.pitch) {
        // ✅ 정답 — highlight 후 다음 음으로 자동 진행
        state.tutorialPianoCtrl.flashCorrect(note);
        const prevIdx = tutExpIdx;
        rerenderTutorial(prevIdx, -1);
        state.tutorialPianoCtrl?.clearExpected();

        tutAdvTimer = setTimeout(() => {
          tutAdvTimer = null;
          const next = (prevIdx + 1) % previewNotes.length;
          tutExpIdx = next;
          rerenderTutorial(-1, next);
          state.tutorialPianoCtrl?.setExpected(previewNotes[next].pitch);
        }, 450);
      } else {
        // ❌ 오답 — 빨간 플래시만
        state.tutorialPianoCtrl.flashWrong(note);
      }
    },
  });

  // 기준 화살표 + 가온다 빨간 점 + 첫 expected 표시
  state.tutorialPianoCtrl.setArrows(REF_ARROWS);
  state.tutorialPianoCtrl.setDots([{ note: 'C4', color: '#FF4444' }]);
  setTimeout(() => state.tutorialPianoCtrl?.setExpected(previewNotes[0].pitch), 120);

  document.getElementById('btn-start').addEventListener('click', () => navigate('convert'));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Convert
// ═══════════════════════════════════════════════════════════════════════════════
function initConvert() {
  const grid = document.getElementById('sample-grid');
  SAMPLES.forEach(s => {
    const card = document.createElement('button');
    card.className = 'sample-card';
    card.innerHTML = `<span class="sample-emoji">${s.emoji}</span><span>${s.title}</span>`;
    card.addEventListener('click', () => startConversion(s));
    grid.appendChild(card);
  });

  const dropZone  = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  document.getElementById('btn-file-pick').addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => { if (e.target.files[0]) handleUpload(e.target.files[0]); });
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
  });
  document.getElementById('btn-save-notation').addEventListener('click', saveCurrentResult);
}

function startConversion(sample) {
  showLoading();
  const data = sample.staves
    ? { id: generateId(), title: sample.title, tempo: sample.tempo,
        timeSignature: sample.timeSignature, staves: sample.staves,
        notes: sample.staves[0].notes, createdAt: Date.now() }
    : { id: generateId(), title: sample.title, tempo: sample.tempo,
        timeSignature: sample.timeSignature, notes: sample.notes, createdAt: Date.now() };
  setTimeout(() => showResult(data), 2200);
}

function handleUpload(file) {
  if (!file.type.startsWith('image/')) { toast('이미지 파일을 선택해주세요'); return; }
  if (file.size > 10 * 1024 * 1024)   { toast('파일 크기가 10MB를 초과합니다'); return; }
  showLoading();
  const demo = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
  const data = demo.staves
    ? { id: generateId(), title: file.name.replace(/\.[^.]+$/, ''),
        tempo: demo.tempo, timeSignature: demo.timeSignature,
        staves: demo.staves, notes: demo.staves[0].notes, createdAt: Date.now() }
    : { id: generateId(), title: file.name.replace(/\.[^.]+$/, ''),
        tempo: demo.tempo, timeSignature: demo.timeSignature,
        notes: demo.notes, createdAt: Date.now() };
  setTimeout(() => showResult(data), 2600);
}

function showLoading() {
  document.getElementById('output-placeholder').classList.add('hidden');
  document.getElementById('output-result').classList.add('hidden');
  document.getElementById('output-loading').classList.remove('hidden');
  const steps = document.querySelectorAll('.loading-step');
  const bar   = document.getElementById('progress-bar');
  steps.forEach(s => s.classList.remove('active','done'));
  bar.style.width = '0%';
  [0, 750, 1550].forEach((delay, idx) => {
    setTimeout(() => {
      steps.forEach((s, si) => {
        s.classList.toggle('active', si === idx);
        if (si < idx) s.classList.add('done');
      });
      bar.style.width = ((idx + 1) / steps.length * 100) + '%';
    }, delay);
  });
}

function showResult(data) {
  state.convertResult = data;
  document.getElementById('output-loading').classList.add('hidden');
  document.getElementById('output-result').classList.remove('hidden');
  document.getElementById('result-title').textContent = data.title;
  document.getElementById('save-title-input').value   = data.title;

  const noteCount = data.staves
    ? data.staves.reduce((s, st) => s + st.notes.length, 0)
    : data.notes.length;
  const staffLabel = data.staves ? ` · ${data.staves.length}단 악보` : '';
  document.getElementById('result-note-count').textContent = `음표 ${noteCount}개${staffLabel}`;
  document.getElementById('result-tempo').textContent  = `BPM ${data.tempo}`;

  const container = document.getElementById('convert-notation');

  if (data.staves) {
    renderGrandStaff(container, data.staves);
    setTimeout(notationNavUpdate.convert, 50);
  } else {
    function rerenderConvert(idx) {
      renderNotation(container, data.notes, {
        highlightIdx: idx,
        onNoteClick(i, note) {
          audio.unlock(); audio.playNote(note.pitch, 0.4);
          rerenderConvert(i);
        },
      });
      notationNavUpdate.convert();
    }
    rerenderConvert(-1);
  }
}

function saveCurrentResult() {
  if (!state.convertResult) return;
  const t = document.getElementById('save-title-input').value.trim();
  if (t) state.convertResult.title = t;
  // Grand staff: flatten to treble notes for playback, keep staves for display
  const toSave = { ...state.convertResult };
  if (!toSave.notes && toSave.staves) {
    toSave.notes = toSave.staves[0]?.notes ?? [];
  }
  saveNotation(toSave);
  toast(`💾 "${toSave.title}" 저장 완료!`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Play
// ═══════════════════════════════════════════════════════════════════════════════
function initPlayScreen() {
  const list  = document.getElementById('notation-select-list');
  const saved = loadAll();

  list.innerHTML = '';

  const sectionLabel = text => {
    const d = document.createElement('div');
    d.className = 'select-section-label';
    d.textContent = text;
    return d;
  };

  const makeItem = (title, meta, onClick) => {
    const btn = document.createElement('button');
    btn.className = 'notation-select-item';
    btn.innerHTML = `<span class="item-title">${title}</span>
      <span class="item-meta">${meta}</span>`;
    btn.addEventListener('click', onClick);
    return btn;
  };

  // 기본 샘플
  list.appendChild(sectionLabel('기본 샘플'));
  SAMPLES.forEach(s => {
    const notation = s.staves
      ? { id: s.id, title: s.title, tempo: s.tempo, timeSignature: s.timeSignature,
          notes: s.staves[0].notes, staves: s.staves, createdAt: 0 }
      : { id: s.id, title: s.title, tempo: s.tempo, timeSignature: s.timeSignature,
          notes: s.notes, createdAt: 0 };
    list.appendChild(makeItem(`${s.emoji} ${s.title}`, `${s.notes.length}음 · ${s.tempo}BPM`,
      () => loadPlayNotation(notation)));
  });

  // 내 저장 악보
  if (saved.length > 0) {
    list.appendChild(sectionLabel('내 악보'));
    saved.forEach(n => {
      list.appendChild(makeItem(n.title, `${n.notes.length}음 · ${n.tempo}BPM`,
        () => loadPlayNotation(n)));
    });
  }

  // 첫 진입 시 첫 샘플 자동 로드
  if (!state.playNotation) {
    const s = SAMPLES[0];
    const notation = { id: s.id, title: s.title, tempo: s.tempo,
                       timeSignature: s.timeSignature, notes: s.notes, createdAt: 0 };
    setTimeout(() => loadPlayNotation(notation), 80);
  }

  document.getElementById('btn-auto-play').onclick = toggleAutoPlay;
  document.getElementById('btn-next-note').onclick  = stepForward;
  document.getElementById('btn-prev-note').onclick  = stepBack;
  document.getElementById('btn-reset').onclick      = resetPlay;

  const tempoSlider  = document.getElementById('tempo-slider');
  const tempoDisplay = document.getElementById('tempo-display');
  tempoSlider.addEventListener('input', () => {
    tempoDisplay.textContent = tempoSlider.value + ' BPM';
  });

  document.getElementById('toggle-note-names').addEventListener('change', e => {
    state.playPianoCtrl?.updateLabels(e.target.checked);
  });

  const pianoEl      = document.getElementById('piano');
  const pianoWrapper = document.getElementById('piano-wrapper');
  pianoEl.innerHTML  = '';
  state.playPianoCtrl?.destroy();

  state.playPianoCtrl = buildPiano(pianoEl, pianoWrapper, {
    showLabels: true,
    onPress(pressedNote) {
      if (!state.playNotation) return;
      if (state.playCancel) return;

      const expIdx  = state.playNoteIdx;
      if (expIdx < 0 || expIdx >= state.playNotation.notes.length) return;

      const expNote = state.playNotation.notes[expIdx];

      if (pressedNote === expNote.pitch) {
        state.playPianoCtrl.flashCorrect(pressedNote);
        clearCountdown();

        const bpm    = parseInt(document.getElementById('tempo-slider').value);
        const beatMs = (60 / bpm) * expNote.duration * 1000;

        state.playedIdx = expIdx;
        renderPlay(expIdx, -1);

        showCountdown(beatMs, () => {
          const nextIdx = expIdx + 1;
          if (nextIdx < state.playNotation.notes.length) {
            state.playNoteIdx = nextIdx;
            state.playedIdx   = -1;
            renderPlay(-1, nextIdx);
            state.playPianoCtrl.setExpected(state.playNotation.notes[nextIdx].pitch);
          } else {
            state.playNoteIdx = -1;
            state.playedIdx   = -1;
            state.playPianoCtrl.clearExpected();
            renderPlay(-1, -1);
            toast('🎉 연주 완료! 처음부터 다시 하려면 ⏹ 버튼을 누르세요');
          }
        });
      } else {
        state.playPianoCtrl.markWrong(pressedNote);
        // 악보 컨테이너도 빨간 깜빡임
        const nc = document.getElementById('play-notation');
        if (nc) {
          nc.classList.add('notation-wrong-flash');
          setTimeout(() => nc.classList.remove('notation-wrong-flash'), 380);
        }
      }
    },
    onRelease(note) {
      state.playPianoCtrl?.clearWrong(note);
    },
    // 옥타브 변경 시 해당 옥타브의 음표로 악보 스크롤 동기화
    onOctaveChange(minOct, maxOct) {
      const notes = state.playNotation?.notes;
      if (!notes || !state.playNotationCtrl) return;
      // 현재 expected 음표가 새 옥타브 범위에 있으면 그것으로, 아니면 범위 내 첫 음표로
      const expOct = parseInt(notes[state.playNoteIdx]?.pitch?.slice(-1));
      if (!isNaN(expOct) && expOct >= minOct && expOct <= maxOct) {
        state.playNotationCtrl.scrollToNote(state.playNoteIdx);
      } else {
        const nearIdx = notes.findIndex(n => {
          const o = parseInt(n.pitch.slice(-1));
          return o >= minOct && o <= maxOct;
        });
        if (nearIdx >= 0) state.playNotationCtrl.scrollToNote(nearIdx);
      }
    },
  });

  // 기준 화살표 + 가온다 빨간 점 표시
  state.playPianoCtrl.setArrows(REF_ARROWS);
  state.playPianoCtrl.setDots([{ note: 'C4', color: '#FF4444' }]);
}

function loadPlayNotation(notation) {
  stopAutoPlay();
  clearCountdown();
  state.playNotation = notation;
  state.playNoteIdx  = 0;
  state.playedIdx    = -1;

  document.getElementById('play-notation-title').textContent = notation.title;
  document.querySelectorAll('.notation-select-item').forEach(el => {
    el.classList.toggle('active',
      el.querySelector('.item-title').textContent === notation.title);
  });

  renderPlay(-1, 0);
  state.playPianoCtrl?.setExpected(notation.notes[0].pitch);
}

function renderPlay(playedIdx, expectedIdx) {
  if (!state.playNotation) return;
  const container = document.getElementById('play-notation');

  // Grand staff: show both staves, but only treble has interaction
  if (state.playNotation.staves) {
    const trebleNotes = state.playNotation.staves[0].notes;
    const bassNotes   = state.playNotation.staves[1]?.notes ?? [];

    container.innerHTML = '';
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex; flex-direction:column; gap:10px;';

    const staffMeta = [
      { label: '🎵 높은음자리 (Treble)', color: '#0076CE', notes: trebleNotes, interactive: true },
      { label: '🎵 낮은음자리 (Bass)',   color: '#5BB8F5', notes: bassNotes,   interactive: false },
    ];

    let trebleCtrl = null;
    staffMeta.forEach(({ label, color, notes, interactive }) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex; flex-direction:column; gap:3px;';
      const lbl = document.createElement('div');
      lbl.textContent = label;
      lbl.style.cssText = `font-size:11px; font-weight:700; color:${color}; padding-left:68px; font-family:system-ui;`;
      row.appendChild(lbl);
      const inner = document.createElement('div');
      inner.className = 'notation-container scrollable tall';
      row.appendChild(inner);
      wrapper.appendChild(row);

      const ctrl = renderNotation(inner, notes, {
        highlightIdx: interactive ? playedIdx  : -1,
        expectedIdx:  interactive ? expectedIdx : -1,
        onNoteClick: interactive ? (i, note) => {
          if (state.playCancel) return;
          stopAutoPlay(); clearCountdown();
          state.playNoteIdx = i;
          state.playedIdx   = -1;
          audio.unlock(); audio.playNote(note.pitch, 0.45);
          state.playPianoCtrl?.clearExpected();
          state.playPianoCtrl?.setExpected(note.pitch);
          renderPlay(-1, i);
        } : undefined,
      });
      if (interactive) trebleCtrl = ctrl;
    });
    container.appendChild(wrapper);
    state.playNotationCtrl = trebleCtrl;
    if (playedIdx   >= 0 && trebleCtrl) trebleCtrl.scrollToNote(playedIdx);
    if (expectedIdx >= 0 && trebleCtrl) {
      trebleCtrl.scrollToMeasureOf
        ? trebleCtrl.scrollToMeasureOf(expectedIdx)
        : trebleCtrl.scrollToNote(expectedIdx);
    }
    return;
  }

  // Single staff
  const ctrl = renderNotation(
    container,
    state.playNotation.notes,
    {
      highlightIdx: playedIdx,
      expectedIdx,
      onNoteClick(i, note) {
        if (state.playCancel) return;
        stopAutoPlay(); clearCountdown();
        state.playNoteIdx = i;
        state.playedIdx   = -1;
        audio.unlock(); audio.playNote(note.pitch, 0.45);
        state.playPianoCtrl?.clearExpected();
        state.playPianoCtrl?.setExpected(note.pitch);
        renderPlay(-1, i);
      },
    },
  );
  state.playNotationCtrl = ctrl;
  if (playedIdx   >= 0 && ctrl) ctrl.scrollToNote(playedIdx);
  if (expectedIdx >= 0 && ctrl) {
    ctrl.scrollToMeasureOf
      ? ctrl.scrollToMeasureOf(expectedIdx)
      : ctrl.scrollToNote(expectedIdx);
  }
  notationNavUpdate.play();
}

function toggleAutoPlay() {
  const btn = document.getElementById('btn-auto-play');
  if (state.playCancel) {
    stopAutoPlay();
  } else {
    if (!state.playNotation) return;
    btn.textContent = '⏸';
    clearCountdown();
    state.playPianoCtrl?.clearExpected();

    const bpm   = parseInt(document.getElementById('tempo-slider').value);
    const start = Math.max(0, state.playNoteIdx < 0 ? 0 : state.playNoteIdx);
    const notes = state.playNotation.notes.slice(start);

    state.playCancel = audio.playSequence(
      notes, bpm,
      stepIdx => {
        const gIdx = start + stepIdx;
        state.playNoteIdx = gIdx;
        state.playedIdx   = gIdx;
        renderPlay(gIdx, -1);
        const note = state.playNotation.notes[gIdx];
        state.playPianoCtrl?.clearExpected();
        const k = document.querySelector(`[data-note="${note.pitch}"]`);
        if (k) {
          const isB = k.classList.contains('bk');
          k.style.background = '#0076CE';
          setTimeout(() => { k.style.background = isB ? '#1c1c1c' : '#f4efe6'; },
            note.duration * (60 / bpm) * 900);
        }
      },
      () => {
        state.playCancel = null;
        btn.textContent  = '▶';
        toast('🎵 재생 완료');
      },
    );
  }
}

function stopAutoPlay() {
  if (state.playCancel) { state.playCancel(); state.playCancel = null; }
  document.getElementById('btn-auto-play').textContent = '▶';
}

function stepForward() {
  if (!state.playNotation) return;
  stopAutoPlay(); clearCountdown();
  const idx = Math.min((state.playNoteIdx < 0 ? 0 : state.playNoteIdx + 1),
                        state.playNotation.notes.length - 1);
  state.playNoteIdx = idx;
  state.playedIdx   = -1;
  const note = state.playNotation.notes[idx];
  audio.unlock(); audio.playNote(note.pitch, 0.4);
  state.playPianoCtrl?.clearExpected();
  state.playPianoCtrl?.setExpected(note.pitch);
  renderPlay(-1, idx);
}

function stepBack() {
  if (!state.playNotation) return;
  stopAutoPlay(); clearCountdown();
  const idx = Math.max((state.playNoteIdx <= 0 ? 0 : state.playNoteIdx - 1), 0);
  state.playNoteIdx = idx;
  state.playedIdx   = -1;
  const note = state.playNotation.notes[idx];
  audio.unlock(); audio.playNote(note.pitch, 0.4);
  state.playPianoCtrl?.clearExpected();
  state.playPianoCtrl?.setExpected(note.pitch);
  renderPlay(-1, idx);
}

function resetPlay() {
  stopAutoPlay(); clearCountdown();
  state.playNoteIdx = 0;
  state.playedIdx   = -1;
  state.playPianoCtrl?.clearHighlights();
  if (state.playNotation) {
    renderPlay(-1, 0);
    state.playPianoCtrl?.setExpected(state.playNotation.notes[0].pitch);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Library
// ═══════════════════════════════════════════════════════════════════════════════
function initLibraryScreen() {
  const grid = document.getElementById('library-grid');
  const all  = loadAll();

  if (all.length === 0) {
    grid.innerHTML = `<div class="empty-state">저장된 악보가 없습니다.<br>
      <button class="link-btn" data-nav="convert">악보 변환</button>으로 첫 악보를 만들어보세요!</div>`;
    return;
  }

  grid.innerHTML = '';
  all.forEach(n => {
    const card = document.createElement('div');
    card.className = 'library-card';

    const barHtml = n.notes.slice(0, 14).map(nt =>
      `<span style="background:${BEAT_COLORS[nt.beat]};width:${nt.duration * 10}px"></span>`
    ).join('');

    card.innerHTML = `
      <div class="library-preview"><div class="beat-bar">${barHtml}</div></div>
      <div class="library-info">
        <strong>${n.title}</strong>
        <span>${n.notes.length}음 · ${n.tempo}BPM</span>
        <span>${new Date(n.createdAt).toLocaleDateString('ko-KR')}</span>
      </div>
      <div class="library-actions">
        <button class="btn-sm btn-play" data-id="${n.id}">🎹 연주하기</button>
        <button class="btn-sm btn-del"  data-id="${n.id}">🗑</button>
      </div>`;
    grid.appendChild(card);
  });

  grid.addEventListener('click', e => {
    const id = e.target.dataset.id; if (!id) return;
    if (e.target.classList.contains('btn-play')) {
      const n = loadAll().find(x => x.id === id);
      if (n) { navigate('play'); setTimeout(() => loadPlayNotation(n), 80); }
    }
    if (e.target.classList.contains('btn-del')) {
      if (confirm('이 악보를 삭제할까요?')) {
        deleteNotation(id); toast('삭제되었습니다'); initLibraryScreen();
      }
    }
  });
}

// ── 전역 이벤트 위임 ───────────────────────────────────────────────────────────
document.addEventListener('click', e => {
  const nav = e.target.dataset.nav || e.target.closest('[data-nav]')?.dataset.nav;
  if (nav) { wheelLocked = false; navigate(nav); return; }

  // nav-tab / page-dot 클릭은 wheelLocked 무시하고 즉시 전환
  const isTab = e.target.classList.contains('nav-tab') || e.target.classList.contains('page-dot');
  const scr   = e.target.dataset.screen;
  if (scr) {
    if (isTab) { wheelLocked = false; }
    navigate(scr);
  }
});

// ── 부팅 ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // 첫 화면(tutorial)은 transition 없이 즉시 표시
  const firstScreen = document.getElementById('screen-tutorial');
  if (firstScreen) {
    firstScreen.classList.add('no-transition');
    firstScreen.classList.add('active');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        firstScreen.classList.remove('no-transition');
      });
    });
  }

  // 악보 nav 버튼 초기화 (DOM 구성 후)
  notationNavUpdate.tutorial = makeNotationNav('tutorial-notation', 'tutorial-notation-prev', 'tutorial-notation-next');
  notationNavUpdate.convert  = makeNotationNav('convert-notation',  'convert-notation-prev',  'convert-notation-next');
  notationNavUpdate.play     = makeNotationNav('play-notation',     'play-notation-prev',     'play-notation-next');

  initTutorial();
  initConvert();
  // navigate는 이미 active를 설정했으므로 state만 맞춤
  state.screen = 'tutorial';
  document.querySelectorAll('.nav-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.screen === 'tutorial'));
  document.querySelectorAll('.page-dot').forEach(d =>
    d.classList.toggle('active', d.dataset.screen === 'tutorial'));
});
