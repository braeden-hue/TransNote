import { SAMPLES, BEAT_COLORS }          from './samples.js';
import { audio }                           from './audio.js';
import { renderNotation, renderGrandStaff } from './notation.js';
import { buildPiano, renderLabeledOctave } from './piano.js';
import { loadAll, saveNotation, deleteNotation, generateId } from './storage.js';
import { signInWithGoogle, signOutUser, onAuthChange,
         saveScoreCloud, loadScoresCloud, deleteScoreCloud } from './firebase.js';
import * as midi from './midi.js';

// ── 피아노 기준 화살표 — 가온다(C4, Middle C) ─────────────────────────────────
const REF_ARROWS = [
  { note: 'C4', color: '#0076CE', label: '가온다' },
];

// ── 전역 상태 ──────────────────────────────────────────────────────────────────
const state = {
  screen:            'tutorial',
  user:              null,
  convertResult:     null,
  playNotation:      null,
  playNoteIdx:       -1,
  playedIdx:         -1,
  playCancel:        null,
  countdownTimer:    null,
  tutorialPianoCtrl: null,
  playPianoCtrl:     null,
  playNotationCtrl:  null,  // renderPlay 에서 반환된 ctrl
  cameraStream:      null,
  activeCameraStop:  null, // 현재 열려 있는 카메라 캡처 UI를 정리하는 함수(둘 중 열린 쪽이 등록)
  kioskMode:         false, // false | 'tutorial'(태블릿1) | 'convert'(태블릿2) — 전시 부스 킨스크 고정 화면(URL ?kiosk=)
  flowLock:          null,  // null(자유 이동) | 허용 화면 이름 배열 — 랜딩에서 튜토리얼/체험하기 진입 시 그 화면(들)로만 제한
  flowFromLanding:   false, // flowLock이 랜딩 클릭으로 걸린 것인지(첫 화면 "이전"이 랜딩으로 나가는 종료 버튼이 됨)
  expScoreData:      null,  // 체험하기 화면에서 보여주는 중인 악보(원본 데이터)
  expMidiConfirmed:  false, // 연주하기 진입 시 전자 피아노 연결 테스트 결과
  expHandMode:       'right', // 'right'(오른손만, 100점) | 'both'(양손, 150점)
  expPerform:        null,  // 연주 진행 중 상태 { nickname, notes, idx, correct, wrong, pianoCtrl }
  accompanimentMode: false, // 반주 모드 — ▶ 재생이 왼손(베이스)만 자체 템포로 연주
};

// 카메라 캡처 UI가 2군데(기존 변환 화면 / 체험하기)라 어느 쪽이 열려 있든 여기서 끌 수
// 있게, 연 쪽이 자기 정리 함수를 등록해두고 이 함수는 그걸 호출만 한다.
function stopCamera() {
  state.activeCameraStop?.();
}

// ── 악보 화살표 nav 헬퍼 ──────────────────────────────────────────────────────
const notationNavUpdate = { convert: () => {}, play: () => {} };

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

// 화면 전환 컨테이너(.app-main/.screens-wrapper)는 항상 scrollTop=0이어야 하는데,
// 키보드 포커스 이동 등으로 브라우저가 자동 스크롤을 걸어버리는 경우가 있어(특히
// 콘텐츠가 긴 화면에서) 화면이 바뀔 때마다 강제로 되돌린다.
function resetShellScroll() {
  document.querySelector('.app-main')?.scrollTo(0, 0);
  document.querySelector('.screens-wrapper')?.scrollTo(0, 0);
}

function navigate(name, { instant = false } = {}) {
  if (state.kioskMode && name !== state.kioskMode) return; // 킨스크: 지정된 화면 밖으로 못 나가게 고정
  if (state.flowLock && !state.flowLock.includes(name)) return; // 랜딩발 가이드 흐름: 허용된 화면 밖으로 못 나가게 고정
  if (wheelLocked && !instant) return;
  resetShellScroll();

  const prevName = state.screen;
  const prevIdx  = SCREEN_ORDER.indexOf(prevName);
  const nextIdx  = SCREEN_ORDER.indexOf(name);
  if (nextIdx < 0) return;
  const goingDown = nextIdx > prevIdx;

  if (prevName === 'convert' && name !== 'convert') stopCamera();

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

  // 하단 탭바 업데이트
  document.querySelectorAll('.nav-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.screen === name));

  state.screen = name;
  if (name === 'play')    initPlayScreen();
  if (name === 'library') initLibraryScreen();
}

// ── 마우스 휠 인터셉트 ─────────────────────────────────────────────────────────
// 발표/데모용: 휠 한 번 → 섹션 전환 (섹션 내 스크롤은 스크롤바·드래그로).
// 단, 현재 화면 내부가 아직 스크롤 가능하면(맨 위/맨 아래가 아니면) 일반 스크롤을
// 그대로 허용한다 — 안 그러면 연주하기처럼 세로로 긴 화면에서 하단(피아노 건반 등)을
// 볼 방법이 없어진다(휠이 항상 화면 전환으로 소비돼버림).
window.addEventListener('wheel', e => {
  if (wheelLocked) { e.preventDefault(); return; }

  const goingDown = e.deltaY > 0;
  const activeScreen = document.querySelector('.screen.active');
  if (activeScreen) {
    const { scrollTop, scrollHeight, clientHeight } = activeScreen;
    const atBottom = scrollTop + clientHeight >= scrollHeight - 1;
    const atTop = scrollTop <= 0;
    if ((goingDown && !atBottom) || (!goingDown && !atTop)) return; // 내부 스크롤에 맡김
  }

  e.preventDefault();
  const cur  = SCREEN_ORDER.indexOf(state.screen);
  const next = goingDown ? cur + 1 : cur - 1;
  if (next >= 0 && next < SCREEN_ORDER.length) {
    wheelLocked = true;
    navigate(SCREEN_ORDER[next]);
    setTimeout(() => { wheelLocked = false; }, 800);
  }
}, { passive: false });

// ═══════════════════════════════════════════════════════════════════════════════
// 랜딩 화면 + 가이드 흐름(튜토리얼 전용 / 체험하기 전용)
// mainpage.png 위 3개 핫스팟에서 진입 — 진입한 흐름 밖으로는 못 나가게(하단 탭바 재사용
// CSS로 숨김) 잠그고, 각 흐름의 "이전" 맨 앞 단계에서만 랜딩으로 돌아가는 종료 통로를 둔다.
// ═══════════════════════════════════════════════════════════════════════════════
function showLanding() { document.getElementById('screen-landing')?.classList.remove('hidden'); }
function hideLanding() { document.getElementById('screen-landing')?.classList.add('hidden'); }

// screens: 이 흐름 안에서 허용할 화면 이름 배열 (예: ['tutorial'] 또는 ['convert','play'])
function enterFlow(screens) {
  state.flowLock = screens;
  state.flowFromLanding = true;
  // 하단 탭바/로그인 필 숨김 + (convert 포함 시) 부스용 개발자 UI 숨김을 킨스크 CSS 그대로 재사용
  document.body.classList.add('kiosk-mode');
  if (screens.includes('convert')) document.body.dataset.kiosk = 'convert';
}

function exitFlowToLanding() {
  state.flowLock = null;
  state.flowFromLanding = false;
  document.body.classList.remove('kiosk-mode');
  delete document.body.dataset.kiosk;
  setMagnifierVisible(false);
  showLanding();
}

// 핫스팟 %가 사진 원본 좌표 기준이라, .landing-frame이 사진과 정확히 같은 비율일 때만
// 점 위치가 맞는다. 뷰포트 비율이 사진과 다르면(세로 스마트폰 등) 레터박스가 생기도록
// 매 리사이즈마다 프레임 실측 px 크기를 직접 계산한다 — 순수 CSS로는 두 축을 동시에
// "꽉 차지만 넘치지 않게"(contain) 못 맞춰서 JS로 처리.
const LANDING_IMG_AR = 1374 / 768;
function fitLandingFrame() {
  const screenEl = document.getElementById('screen-landing');
  const frame = document.querySelector('.landing-frame');
  if (!screenEl || !frame) return;
  const vw = screenEl.clientWidth, vh = screenEl.clientHeight;
  if (!vw || !vh) return;
  if (vw / vh > LANDING_IMG_AR) {
    frame.style.height = vh + 'px';
    frame.style.width  = (vh * LANDING_IMG_AR) + 'px';
  } else {
    frame.style.width  = vw + 'px';
    frame.style.height = (vw / LANDING_IMG_AR) + 'px';
  }
}
window.addEventListener('resize', fitLandingFrame);

function initLanding() {
  fitLandingFrame();
  document.querySelectorAll('.landing-hotspot').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.landing;
      if (target === 'about') { showAboutModal(); return; }
      hideLanding();
      if (target === 'tutorial') {
        enterFlow(['tutorial']);
        renderTutPage(0);
        navigate('tutorial', { instant: true });
      } else if (target === 'experience') {
        showExpSelect();
      }
    });
  });
}

// ── 프로젝트에 대해 모달 ──────────────────────────────────────────────────────
function showAboutModal() { document.getElementById('about-modal')?.classList.remove('hidden'); }
function hideAboutModal() { document.getElementById('about-modal')?.classList.add('hidden'); }

function initAboutModal() {
  document.getElementById('about-modal-close')?.addEventListener('click', hideAboutModal);
  document.getElementById('about-modal-backdrop')?.addEventListener('click', hideAboutModal);
  document.getElementById('about-modal-ok')?.addEventListener('click', hideAboutModal);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') hideAboutModal();
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// 체험하기 — 랜딩에서만 진입하는 완전히 독립된 흐름(기존 변환/연주 화면과 무관).
// 1) 샘플 3곡 + 촬영 버튼만 있는 심플한 선택 화면
// 2) 선택한 곡의 커스텀 악보 첫 마디 + 연주 듣기(전체 곡)/연주하기 + 순위표
// 3) 연주하기 진입 시: 전자피아노 연결 테스트 + 닉네임 입력 대기 화면
// 4) 연주 진행(전자피아노 연결 시 화면 건반 숨김) + 점수 결과
// ═══════════════════════════════════════════════════════════════════════════════
function hideExpScreens() {
  ['screen-exp-select', 'screen-exp-score', 'screen-exp-handmode', 'screen-exp-wait', 'screen-exp-perform']
    .forEach(id => document.getElementById(id)?.classList.add('hidden'));
}

function showExpSelect() {
  hideExpScreens();
  document.getElementById('screen-exp-select')?.classList.remove('hidden');
}

// notes에서 beat===1이 두 번째로 나오기 전까지만 잘라 "첫 마디"만 반환 (마디가
// 하나뿐이면 전체 그대로).
function firstMeasure(notes) {
  if (!notes?.length) return [];
  let seen = 0;
  for (let i = 0; i < notes.length; i++) {
    if (notes[i].beat === 1) {
      seen++;
      if (seen === 2) return notes.slice(0, i);
    }
  }
  return notes;
}

// ── 곡별 순위표(상위 3명) — 서버가 없으니 이 브라우저(기기)에 localStorage로 저장 ──
const EXP_LEADERBOARD_KEY = 'expLeaderboard_v1';

function loadLeaderboard(songKey) {
  try {
    const all = JSON.parse(localStorage.getItem(EXP_LEADERBOARD_KEY) || '{}');
    return all[songKey] || [];
  } catch { return []; }
}

function saveLeaderboardEntry(songKey, nickname, score) {
  let all = {};
  try { all = JSON.parse(localStorage.getItem(EXP_LEADERBOARD_KEY) || '{}'); } catch { /* 무시 */ }
  const list = all[songKey] || [];
  list.push({ nickname, score });
  list.sort((a, b) => b.score - a.score);
  all[songKey] = list.slice(0, 3);
  try { localStorage.setItem(EXP_LEADERBOARD_KEY, JSON.stringify(all)); } catch { /* 저장 공간 부족 등 무시 */ }
  return all[songKey];
}

function renderLeaderboard(songKey) {
  const el = document.getElementById('exp-leaderboard-list');
  if (!el) return;
  const list = loadLeaderboard(songKey);
  el.innerHTML = list.length
    ? list.map((e, i) => `
        <li>
          <span class="exp-lb-rank">${i + 1}</span>
          <span class="exp-lb-name">${e.nickname}</span>
          <span class="exp-lb-score">${e.score}점</span>
        </li>`).join('')
    : '<li class="exp-leaderboard-empty">아직 기록이 없어요</li>';
}

function showExpScore(data) {
  state.expScoreData = data;
  hideExpScreens();
  document.getElementById('screen-exp-score')?.classList.remove('hidden');

  document.getElementById('exp-score-title').textContent = data.title || '';
  const container = document.getElementById('exp-score-notation');
  container.innerHTML = '';
  if (data.staves?.length >= 2) {
    const trimmed = data.staves.map(s => ({ ...s, notes: firstMeasure(s.notes) }));
    renderGrandStaff(container, trimmed);
  } else {
    renderNotation(container, firstMeasure(data.notes), {});
  }
  autoFitExpScore();

  // 촬영으로 만든 악보는 정해진 3곡과 달리 순위 기록 대상이 아님(제목이 매번 달라
  // 순위표 자체가 의미 없기도 함) — 패널을 숨긴다.
  const lb = document.querySelector('.exp-leaderboard');
  lb?.classList.toggle('hidden', !!data._noScore);
  if (!data._noScore) renderLeaderboard(data.title);
}

// 튜토리얼의 autoFitTutBoxes()와 같은 원리 — 악보 박스가 화면보다 크면(가로로 눕힌
// 짧은 화면 등) 축소해서 상하 스크롤 없이 맞춘다. 악보 화면/연주 화면 둘 다 씀.
function autoFitExpScore(boxId = 'exp-score-notation') {
  const box = document.getElementById(boxId);
  if (!box) return;
  Array.from(box.children).forEach(c => { c.style.zoom = ''; });
  const overflow = box.scrollHeight - box.clientHeight;
  if (overflow > 4 && box.clientHeight > 0) {
    const zoom = Math.max(0.4, (box.clientHeight / box.scrollHeight) * 0.95);
    Array.from(box.children).forEach(c => { c.style.zoom = zoom; });
  }
}
window.addEventListener('resize', () => {
  if (state.expScoreData) autoFitExpScore('exp-score-notation');
  if (state.expPerform)   autoFitExpScore('exp-perform-notation');
});

function playExpScore() {
  const data = state.expScoreData;
  if (!data) return;
  audio.unlock();
  const bpm = data.tempo || 90;
  const btn = document.getElementById('exp-play-btn');
  btn?.classList.add('playing');
  const done = () => btn?.classList.remove('playing');

  // 화면엔 첫 마디만 보여주지만, 재생은 곡 전체(모든 마디)를 들려준다.
  if (data.staves?.length >= 2) {
    // 대보표 악보 한정 — 왼손·오른손 전체를 같은 템포로 동시에 재생
    const treble = data.staves[0].notes;
    const bass   = data.staves[1].notes;
    let pending = 0;
    const onEnd = () => { pending--; if (pending <= 0) done(); };
    if (treble.length) { pending++; audio.playSequence(treble, bpm, () => {}, onEnd); }
    if (bass.length)   { pending++; audio.playSequence(bass,   bpm, () => {}, onEnd); }
    if (!pending) done();
  } else {
    const notes = data.notes;
    if (!notes?.length) { done(); return; }
    audio.playSequence(notes, bpm, () => {}, done);
  }
}

// 체험하기용 촬영 결과 처리 — round3train 체크포인트(r15)로 서버가 실제 인식한다.
// "변환 중"에도 새 로딩 화면을 따로 만들지 않고 지금 화면(선택 화면) 그대로 두고 토스트만
// 띄운다. 촬영해서 만든 악보는 정해진 3곡과 달리 순위표 대상이 아님(_noScore).
async function handleExpCameraCapture(file) {
  if (!file) return;
  toast('🔍 악보 인식 중... (체크포인트로 추론 중)');
  try {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/recognize?model=andromr', { method: 'POST', body: form });
    if (!res.ok) throw new Error('server');
    const json = await res.json();
    json._noScore = true;
    showExpScore(json);
  } catch {
    toast('⚠️ OMR 서버 미연결 — 샘플로 보여드릴게요');
    const demo = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
    showExpScore(sampleToNotation(demo, {
      id: generateId(), title: file.name.replace(/\.[^.]+$/, ''), createdAt: Date.now(), _noScore: true,
    }));
  }
}

// ── 전자 피아노 연결 테스트 ────────────────────────────────────────────────
// 단순히 "MIDI 기기가 잡히는가"만 보면 실제로 안 눌러도 "연결됨"으로 오판할 수 있어서,
// 커스텀 악보 첫 음에 해당하는 건반을 실제로 눌러보게 해서 신호가 오는지까지 확인한다.
function testMidiConnection(targetNote, { timeoutMs = 6000 } = {}) {
  return new Promise(resolve => {
    if (!targetNote || !midi.isSupported()) { resolve(false); return; }
    midi.requestAccess().then(() => {
      const inputs = midi.listInputs();
      if (!inputs.length) { resolve(false); return; }
      let done = false;
      const timer = setTimeout(() => finish(false), timeoutMs);
      function finish(ok) {
        if (done) return;
        done = true;
        clearTimeout(timer);
        resolve(ok);
      }
      midi.setInput(inputs[0].id, {
        onNoteOn: pitch => { if (pitch === targetNote) finish(true); },
        onNoteOff: () => {},
      });
    }).catch(() => resolve(false));
  });
}

// notes를 beat===1 기준으로 마디 배열로 쪼갠다: [[마디1 음표들], [마디2 음표들], ...]
function splitMeasures(notes) {
  if (!notes?.length) return [];
  const measures = [];
  let cur = [];
  notes.forEach(n => {
    if (n.beat === 1 && cur.length) { measures.push(cur); cur = []; }
    cur.push(n);
  });
  if (cur.length) measures.push(cur);
  return measures;
}

// ── 3/5: 오른손만 / 양손 선택 ─────────────────────────────────────────────
function showExpHandMode() {
  hideExpScreens();
  document.getElementById('screen-exp-handmode')?.classList.remove('hidden');
  const bothBtn = document.getElementById('exp-handmode-both');
  bothBtn?.classList.toggle('exp-handmode-disabled', !(state.expScoreData?.staves?.length >= 2));
}

// ── 4/5: 전자피아노 연결 확인 + 닉네임 대기 화면 ───────────────────────────
function showExpWait(handMode) {
  state.expHandMode = handMode;
  hideExpScreens();
  document.getElementById('screen-exp-wait')?.classList.remove('hidden');

  const statusEl = document.getElementById('exp-wait-midi-status');
  const data = state.expScoreData;
  const firstNote = firstMeasure(data?.staves?.[0]?.notes ?? data?.notes)?.[0]?.pitch;

  state.expMidiConfirmed = false;
  if (!firstNote) {
    statusEl.textContent = '📱 화면 건반으로 연주해요';
  } else {
    statusEl.textContent = `🎹 전자 피아노에서 ${solfegeOf(firstNote)} 음을 눌러 연결을 확인해주세요`;
    testMidiConnection(firstNote).then(ok => {
      state.expMidiConfirmed = ok;
      statusEl.textContent = ok
        ? '🎹 전자 피아노 연결을 확인했어요 — 화면 건반 없이 연주해요'
        : '📱 화면 건반으로 연주해요 (전자 피아노 신호를 못 받았어요)';
    });
  }
}

// ── 5/5: 연주 진행(곡 전체, 마디마다 자연스럽게 다음 마디로) + 점수 ─────────
function startExpPerform(nickname) {
  const data = state.expScoreData;
  if (!data) return;
  const handMode = state.expHandMode === 'both' && data.staves?.length >= 2 ? 'both' : 'right';

  const trebleNotes = (data.staves?.[0]?.notes ?? data.notes ?? []).filter(n => !n.isRest);
  const bassNotes   = handMode === 'both' ? (data.staves[1].notes ?? []).filter(n => !n.isRest) : [];

  state.expPerform = {
    nickname, handMode,
    maxScore: handMode === 'both' ? 150 : 100,
    noScore: !!data._noScore,
    trebleMeasures: splitMeasures(trebleNotes),
    bassMeasures:   splitMeasures(bassNotes),
    measureIdx: 0, tIdx: 0, bIdx: 0,
    tHit: new Set(), bHit: new Set(), // 현재 스텝(화음 포함)에서 이미 맞힌 음들 — 화음은 전부 눌러야 그 손이 다음으로 넘어감
    correct: 0, wrong: 0,
    pianoCtrl: null,
  };

  hideExpScreens();
  document.getElementById('screen-exp-perform')?.classList.remove('hidden');
  document.getElementById('exp-perform-title').textContent = data.title || '';
  document.getElementById('exp-perform-result').classList.add('hidden');

  const notationEl = document.getElementById('exp-perform-notation');

  function currentMeasures() {
    const p = state.expPerform;
    return { t: p.trebleMeasures[p.measureIdx] ?? [], b: p.bassMeasures[p.measureIdx] ?? [] };
  }
  // 지금 이 스텝에서 요구되는 전체 음 목록(주음 + 화음 딸린 음) — 화음이면 원래 음정
  // 그대로 여러 개, 아니면 1개.
  function stepPitches(measureNotes, idx) {
    if (!measureNotes || idx >= measureNotes.length) return [];
    const n = measureNotes[idx];
    return [n.pitch, ...(n.chordNotes || [])];
  }
  function renderMeasure() {
    const p = state.expPerform;
    const { t, b } = currentMeasures();
    notationEl.innerHTML = '';
    if (p.handMode === 'both') {
      renderGrandStaff(notationEl, [{ clef: 'treble', notes: t }, { clef: 'bass', notes: b }], {
        expectedIdxByClef: { treble: p.tIdx, bass: p.bIdx },
      });
    } else {
      renderNotation(notationEl, t, { expectedIdx: p.tIdx });
    }
    autoFitExpScore('exp-perform-notation');
  }
  // 화면 건반엔 "아직 안 누른" 음들만 표시 — 화음 중 이미 누른 음은 더 이상 안내하지 않음.
  function updateHighlight() {
    const p = state.expPerform;
    const { t, b } = currentMeasures();
    const tRemain = stepPitches(t, p.tIdx).filter(n => !p.tHit.has(n));
    const bRemain = p.handMode === 'both' ? stepPitches(b, p.bIdx).filter(n => !p.bHit.has(n)) : [];
    p.pianoCtrl?.setExpected(tRemain[0] ?? bRemain[0] ?? null);
    if (p.handMode === 'both') {
      p.pianoCtrl?.setDots([
        ...tRemain.map(n => ({ note: n, color: '#0076CE' })), // 오른손 = 파랑
        ...bRemain.map(n => ({ note: n, color: '#FF8A3D' })), // 왼손 = 주황
      ]);
    }
  }
  renderMeasure();

  const pianoWrap = document.getElementById('exp-perform-piano');
  pianoWrap.innerHTML = '';
  // 전자 피아노 연결이 확인됐으면 화면 건반은 띄우지 않음(실물로 연주) — 아니면 화면 건반 표시.
  pianoWrap.classList.toggle('hidden', state.expMidiConfirmed);

  // 오른손/왼손이 동시에 눌려야 하는 화음도 "어느 손이 지금 이 음을 낼 차례인가"를
  // 헷갈리지 않게 판단 — 오른손(트레블) 스텝에 아직 안 채워진 음이면 오른손으로,
  // 아니면 왼손 스텝을 본다. 둘 다 아니면 오답.
  function handleExpNote(pitch) {
    const p = state.expPerform;
    const { t, b } = currentMeasures();
    const tStep = stepPitches(t, p.tIdx);
    const bStep = p.handMode === 'both' ? stepPitches(b, p.bIdx) : [];

    let matched = false;
    if (tStep.includes(pitch) && !p.tHit.has(pitch)) {
      p.tHit.add(pitch);
      matched = true;
      if (p.tHit.size >= tStep.length) { p.tIdx++; p.tHit.clear(); }
    } else if (bStep.includes(pitch) && !p.bHit.has(pitch)) {
      p.bHit.add(pitch);
      matched = true;
      if (p.bHit.size >= bStep.length) { p.bIdx++; p.bHit.clear(); }
    }

    if (!matched) {
      p.wrong++;
      p.pianoCtrl?.flashWrong(pitch);
      return;
    }
    p.correct++;
    p.pianoCtrl?.flashCorrect(pitch);

    const tDone = p.tIdx >= t.length;
    const bDone = p.handMode !== 'both' || p.bIdx >= b.length;
    if (tDone && bDone) {
      const totalMeasures = Math.max(p.trebleMeasures.length, p.bassMeasures.length);
      if (p.measureIdx + 1 >= totalMeasures) { finishExpPerform(); return; }
      // 한 마디를 다 치면 자연스럽게 다음 마디로 — 양손 모드면 위/아래 악보가 같이 넘어감
      p.measureIdx++; p.tIdx = 0; p.bIdx = 0; p.tHit.clear(); p.bHit.clear();
    }
    renderMeasure();
    updateHighlight();
  }

  if (state.expMidiConfirmed) {
    const inputs = midi.listInputs();
    if (inputs.length) midi.setInput(inputs[0].id, { onNoteOn: handleExpNote, onNoteOff: () => {} });
  } else {
    const wrap = document.createElement('div');
    wrap.className = 'piano-wrapper mini-piano';
    const pianoEl = document.createElement('div');
    pianoEl.className = 'piano';
    wrap.appendChild(pianoEl);
    pianoWrap.appendChild(wrap);
    state.expPerform.pianoCtrl = buildPiano(pianoEl, wrap, { showLabels: true, onPress: handleExpNote });
  }
  updateHighlight();
}

function finishExpPerform() {
  const p = state.expPerform;
  if (!p) return;
  if (p.noScore) {
    document.getElementById('exp-perform-score-text').textContent = `${p.nickname}님, 수고하셨어요! 🎉`;
    document.getElementById('exp-perform-result').classList.remove('hidden');
    return;
  }
  const total = p.correct + p.wrong;
  const score = total > 0 ? Math.round((p.correct / total) * p.maxScore * 10) / 10 : 0;
  document.getElementById('exp-perform-score-text').textContent =
    `${p.nickname}님, ${score}점이에요! (${p.maxScore}점 만점)`;
  document.getElementById('exp-perform-result').classList.remove('hidden');
  saveLeaderboardEntry(state.expScoreData?.title, p.nickname, score);
}

function initExpFlow() {
  const grid = document.getElementById('exp-sample-grid');
  SAMPLES.forEach(s => {
    const card = document.createElement('button');
    card.className = 'exp-sample-card';
    card.innerHTML = `<span class="exp-sample-emoji">${s.emoji}</span><span class="exp-sample-title">${s.title}</span>`;
    card.addEventListener('click', () => {
      showExpScore(sampleToNotation(s, { id: generateId(), createdAt: Date.now() }));
    });
    grid.appendChild(card);
  });

  setupCameraCapture({
    openBtn: 'exp-camera-btn', cancelBtn: 'exp-camera-cancel', shutterBtn: 'exp-camera-shutter',
    captureBox: 'exp-camera-capture', video: 'exp-camera-video', canvas: 'exp-camera-canvas',
    guideCanvas: 'exp-camera-guide-canvas', guideHint: 'exp-camera-guide-hint', error: 'exp-camera-error',
  }, handleExpCameraCapture);

  document.getElementById('exp-handmode-right')?.addEventListener('click', () => showExpWait('right'));
  document.getElementById('exp-handmode-both')?.addEventListener('click', () => {
    if (state.expScoreData?.staves?.length >= 2) showExpWait('both');
  });

  document.getElementById('exp-play-btn')?.addEventListener('click', playExpScore);
  document.getElementById('exp-perform-btn')?.addEventListener('click', showExpHandMode);

  document.getElementById('exp-start-perform-btn')?.addEventListener('click', () => {
    const nickname = document.getElementById('exp-nickname-input').value.trim() || '익명';
    startExpPerform(nickname);
  });
  document.getElementById('exp-perform-done-btn')?.addEventListener('click', () => {
    showExpScore(state.expScoreData); // 순위표 갱신 포함해서 악보 화면으로 복귀
  });

  const goHome = () => { hideExpScreens(); showLanding(); };
  document.getElementById('exp-select-home')?.addEventListener('click', goHome);
  document.getElementById('exp-score-home')?.addEventListener('click', goHome);
  document.getElementById('exp-handmode-home')?.addEventListener('click', goHome);
  document.getElementById('exp-wait-home')?.addEventListener('click', goHome);
  document.getElementById('exp-perform-home')?.addEventListener('click', goHome);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 12음 참고 돋보기 — 규칙2 이후 튜토리얼 + 체험하기(변환/연주)에서 우측 중앙 고정.
// 호버(데스크톱)/탭(터치)하면 규칙0에서 봤던 라벨드 옥타브를 다시 띄워 12음 매핑을 상기시킴.
// ═══════════════════════════════════════════════════════════════════════════════
let magnifierPopupBuilt = false;

function ensureMagnifierPopup() {
  if (magnifierPopupBuilt) return;
  magnifierPopupBuilt = true;
  const popup = document.getElementById('magnifier-popup');
  popup.innerHTML = `
    <p class="magnifier-popup-title">12음 참고표</p>
    <div class="magnifier-popup-scale"><div class="magnifier-popup-scale-inner" id="magnifier-octave"></div></div>`;
  renderLabeledOctave(document.getElementById('magnifier-octave'), { oct: 4 });
}

function initOctaveMagnifier() {
  const wrap  = document.getElementById('octave-magnifier');
  const btn   = document.getElementById('magnifier-btn');
  const popup = document.getElementById('magnifier-popup');
  if (!wrap || !btn || !popup) return;

  const show = () => { ensureMagnifierPopup(); popup.classList.add('visible'); };
  const hide = () => popup.classList.remove('visible');

  btn.addEventListener('mouseenter', show);
  btn.addEventListener('mouseleave', hide);
  // 터치 기기는 hover가 없으므로 탭으로 토글
  btn.addEventListener('click', e => {
    e.stopPropagation();
    ensureMagnifierPopup();
    popup.classList.toggle('visible');
  });
  document.addEventListener('click', e => {
    if (!wrap.contains(e.target)) hide();
  });
}

// 돋보기 표시 여부 전환 — 튜토리얼 규칙2 이후 페이지, 체험하기 모드에서 호출.
function setMagnifierVisible(visible) {
  document.getElementById('octave-magnifier')?.classList.toggle('hidden', !visible);
}

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
// ── 튜토리얼: 태블릿1 대기열 온보딩 킨스크(?kiosk=1) 전용 화면. "단순함이 무기" —
// 화면을 위(악보)/아래(피아노) 절반으로만 나누고, 설명은 상단 한 줄 캡션으로 줄인다.
// 힌트 없이도 건반을 눌러보며 "어떤 음을 눌러야 하는지" 스스로 알아낼 수 있게
// 마지막 3장(블라인드 퀴즈)은 정답을 숨긴다.
const SOLFEGE = {
  C: '도', 'C#': '도#', D: '레', 'D#': '레#', E: '미', F: '파', 'F#': '파#',
  G: '솔', 'G#': '솔#', A: '라', 'A#': '라#', B: '시',
};
const solfegeOf = pitch => SOLFEGE[pitch.slice(0, -1)] ?? pitch;

function hexToRgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

// 오른손(트레블) 3존 + 왼손(베이스) 4존(한 옥타브씩) — 88건반 위 색 띠 + 대보표 예시 음에
// 공용으로 쓴다. 왼손 "최저"(0옥, A0~B0)는 스피커에서 잘 안 들릴 수 있지만 옥타브 구분을
// 위해 일단 별도 색으로 넣어둠 — 왼손도 오른손처럼 한 옥타브 = 한 색 원칙으로 통일.
const HAND_ZONES = [
  { hand: '왼손',  zone: '최저', from: 'A0', to: 'B0', hex: '#E8590C', note: 'A0' },
  { hand: '왼손',  zone: '낮음', from: 'C1', to: 'B1', hex: '#FF8A3D', note: 'C1' },
  { hand: '왼손',  zone: '중간', from: 'C2', to: 'B2', hex: '#FFA85C', note: 'G2' },
  { hand: '왼손',  zone: '높음', from: 'C3', to: 'B3', hex: '#FFC98A', note: 'C3' },
  { hand: '오른손', zone: '낮음', from: 'C4', to: 'B4', hex: '#5BB8F5', note: 'C4' },
  { hand: '오른손', zone: '중간', from: 'C5', to: 'B5', hex: '#3A9EE0', note: 'G5' },
  { hand: '오른손', zone: '높음', from: 'C6', to: 'C8', hex: '#0076CE', note: 'C6' },
];

// 규칙1 전용 — 악보 존 배경을 하단 피아노 존 밴드와 같은 색으로(z0=높음→z2=낮음 순).
// HAND_ZONES에서 그대로 뽑아 쓰므로 팔레트가 어긋날 일이 없음.
function zoneColorsForHand(hand) {
  return ['높음', '중간', '낮음'].map(zone => HAND_ZONES.find(z => z.hand === hand && z.zone === zone).hex);
}
const TREBLE_ZONE_COLORS = zoneColorsForHand('오른손');
const BASS_ZONE_COLORS   = zoneColorsForHand('왼손');

// 옥타브 내비게이션 + 피아노를 container 안에 새로 만들고 state.tutorialPianoCtrl에 연결.
// (페이지 전환마다 renderTutPage()가 이전 피아노를 destroy()하므로 리스너가 쌓이지 않는다.)
function buildTutPiano(container, opts = {}) {
  const wrap = document.createElement('div');
  wrap.innerHTML = `
    <div class="octave-nav">
      <button class="octave-btn" data-role="down">◀</button>
      <span data-role="label">C3 ~ B4 영역</span>
      <button class="octave-btn" data-role="up">▶</button>
      <span class="octave-hint">옥타브 이동</span>
    </div>
    <div class="piano-wrapper mini-piano"><div class="piano"></div></div>`;
  container.appendChild(wrap);

  const ctrl = buildPiano(wrap.querySelector('.piano'), wrap.querySelector('.piano-wrapper'), {
    showLabels: true,
    navPrevEl:  wrap.querySelector('[data-role="down"]'),
    navNextEl:  wrap.querySelector('[data-role="up"]'),
    navLabelEl: wrap.querySelector('[data-role="label"]'),
    ...opts,
  });
  state.tutorialPianoCtrl = ctrl;
  ctrl.setArrows(REF_ARROWS);
  ctrl.setDots([{ note: 'C4', color: '#FF4444' }]);
  return ctrl;
}

// "다음 음: 도" 배지 + 피아노로 순서대로 연주해보는 연습 독 (규칙2+3 페이지 전용).
function buildPracticeDock(container, pitchSeq) {
  container.innerHTML = `
    <div class="tut-practice-head">
      <strong>실시간으로 연주해보기</strong>
      <span class="tut-next-note-badge" id="tut-dock-badge"></span>
      <span id="tut-dock-feedback"></span>
    </div>
    <div id="tut-dock-piano"></div>`;

  let idx = 0;
  const badge    = document.getElementById('tut-dock-badge');
  const feedback = document.getElementById('tut-dock-feedback');
  const updateBadge = () => { badge.textContent = `다음 음: ${solfegeOf(pitchSeq[idx])}`; };
  updateBadge();

  const ctrl = buildTutPiano(document.getElementById('tut-dock-piano'), {
    onPress(note) {
      if (note === pitchSeq[idx]) {
        ctrl.flashCorrect(note);
        feedback.innerHTML = '<span class="tut-feedback-ok">✓</span>';
        idx = (idx + 1) % pitchSeq.length;
        updateBadge();
        ctrl.setExpected(pitchSeq[idx]); // 다음 음으로 하늘색 기대 하이라이트를 옮겨줌 (버그: 누락되어 있었음)
      } else {
        ctrl.flashWrong(note);
        feedback.innerHTML = '<span class="tut-feedback-bad">✕</span>';
      }
      setTimeout(() => { feedback.innerHTML = ''; }, 500);
    },
  });
  ctrl.setExpected(pitchSeq[idx]);
}

// spec: { type: 'note', note } | { type: 'chord', notes:[n1,n2] } | { type: 'sequence', notes:[...] }
// 셋 다 공통으로: 정답을 맞히면 짧게 ✓를 보여준 뒤 자동으로 다음 테스트로 넘어간다
// (규칙2 연습 도크와 달리 "테스트"라 정답 확인이 곧 다음 문제로 이어지는 게 자연스러움).
function buildQuizPage(order, spec) {
  const isChord = spec.type === 'chord';
  const isSeq   = spec.type === 'sequence';
  const targets = spec.type === 'note' ? [spec.note] : spec.notes;

  return {
    chip: `테스트 ${order} / 5`,
    caption: isChord ? '어느 화음을 눌러야 할까요? (두 음을 동시에)'
           : isSeq   ? '한 마디 — 순서대로 눌러보세요'
           : '어느 건반을 눌러야 할까요?',
    render(top, bottom) {
      top.innerHTML = '<div class="notation-container" id="tut-quiz-notation"></div>';
      const noteData = isChord
        ? [{ pitch: targets[0], duration: 2, beat: 1, chordNotes: targets.slice(1) }]
        : isSeq
          ? targets.map((p, i) => ({ pitch: p, duration: 1, beat: i + 1 }))
          : [{ pitch: targets[0], duration: 2, beat: 1 }];
      renderNotation(document.getElementById('tut-quiz-notation'), noteData,
        isSeq ? { expectedIdx: 0 } : {});

      bottom.innerHTML = `
        <p class="tut-feedback-hint" id="tut-quiz-feedback">${isSeq ? '첫 음부터 순서대로 눌러보세요' : '건반을 눌러 확인해보세요'}</p>
        <div id="tut-quiz-piano"></div>`;
      const feedback = document.getElementById('tut-quiz-feedback');

      let seqIdx = 0;
      const chordPressed = new Set();

      function goNext() {
        setTimeout(() => {
          if (tutPageIdx < TUT_PAGES.length - 1) goToTutPage(tutPageIdx + 1);
        }, 700); // ✓ 표시를 잠깐 보여준 뒤 자동 전환
      }

      const ctrl = buildTutPiano(document.getElementById('tut-quiz-piano'), {
        onPress(note) {
          audio.unlock(); audio.playNote(note, 0.4);

          if (isChord) {
            if (targets.includes(note)) {
              chordPressed.add(note);
              ctrl.flashCorrect(note);
              if (targets.every(n => chordPressed.has(n))) {
                feedback.innerHTML = '<span class="tut-feedback-ok">✓ 정답이에요!</span>';
                goNext();
              }
            } else {
              chordPressed.clear();
              ctrl.flashWrong(note);
              feedback.innerHTML = '<span class="tut-feedback-bad">✕ 다시 시도해보세요</span>';
            }
            return;
          }

          if (isSeq) {
            if (note === targets[seqIdx]) {
              ctrl.flashCorrect(note);
              seqIdx++;
              renderNotation(document.getElementById('tut-quiz-notation'), noteData, { expectedIdx: seqIdx });
              if (seqIdx >= targets.length) {
                feedback.innerHTML = '<span class="tut-feedback-ok">✓ 정답이에요!</span>';
                goNext();
              } else {
                feedback.innerHTML = `<span class="tut-feedback-ok">✓ (${seqIdx}/${targets.length})</span>`;
                ctrl.setExpected(targets[seqIdx]);
              }
            } else {
              ctrl.flashWrong(note);
              feedback.innerHTML = '<span class="tut-feedback-bad">✕ 다시 시도해보세요</span>';
            }
            return;
          }

          // 단일 음
          if (note === targets[0]) {
            ctrl.flashCorrect(note);
            feedback.innerHTML = '<span class="tut-feedback-ok">✓ 정답이에요!</span>';
            goNext();
          } else {
            ctrl.flashWrong(note);
            feedback.innerHTML = '<span class="tut-feedback-bad">✕ 다시 시도해보세요</span>';
          }
        },
      });

      if (isSeq) ctrl.setExpected(targets[0]);
    },
  };
}

const TUT_PAGES = [
  {
    chip: '시작하기 전에',
    caption: '우리의 목표 — 조표·옥타브 번호 같은 오선지의 복잡한 규칙 없이, 색과 위치로 바로 읽는 악보',
    splitDirection: 'row', // 위/아래 대신 좌/우로 두 악보를 나란히 비교
    render(top, bottom) {
      top.innerHTML = `
        <p class="tut-compare-label">전통 오선 악보</p>
        <img class="tut-compare-img" src="assets/tut-intro-original.png" alt="전통 오선 악보 원본">`;
      bottom.innerHTML = `
        <p class="tut-compare-label">커스텀 악보 (같은 곡)</p>
        <img class="tut-compare-img" src="assets/tut-intro-custom.png" alt="같은 곡을 커스텀 악보로 변환한 결과">`;
    },
  },
  {
    chip: '규칙 0',
    caption: '흰 건반 도~시, 검은 건반 1~5 — 눌러서 들어보세요',
    render(top, bottom) {
      top.innerHTML = `
        <p class="tut-compare-label">방금 본 커스텀 악보의 높은음자리(오른손) 부분</p>
        <img class="tut-compare-img" src="assets/rule0-treble.png" alt="커스텀 악보 높은음자리 부분 — 음 이름이 D, G, B 등으로 표시됨">
        <p style="font-size:16px;color:var(--text-dim);text-align:center;max-width:520px;line-height:1.6;margin-top:6px;">
          여기 쓰인 음 이름들은 한 옥타브 = 흰 건반 7개 + 검은 건반 5개, 총 12음에서 나와요
        </p>`;
      renderLabeledOctave(bottom, { oct: 4 });
    },
  },
  {
    chip: '규칙 1',
    caption: '세로 위치 = 음 높이 (위 오른손 · 아래 왼손)',
    render(top, bottom) {
      const trebleNotes = [
        { pitch: 'C6', duration: 1, beat: 1 },
        { pitch: 'G5', duration: 1, beat: 2 },
        { pitch: 'C4', duration: 1, beat: 3 },
      ];
      const bassNotes = [
        { pitch: 'C3', duration: 1, beat: 1 },
        { pitch: 'G2', duration: 1, beat: 2 },
        { pitch: 'C1', duration: 1, beat: 3 },
      ];
      top.innerHTML = '<div id="tut-r1-grand" style="width:100%; height:100%;"></div>';
      // 이 화면 한정: 존 배경을 하단 피아노 존 밴드와 같은 색으로 칠하고, 음표 자체(밑줄
      // 포함)는 아예 안 그려서 순수 존 색상만 보이게 함. 왼쪽=낮은음자리, 오른쪽=높은음자리로
      // 나란히 배치 — 가로로 눕힌 화면에서 세로 공간을 절반만 써서 스크롤 없이 들어오게.
      renderGrandStaff(document.getElementById('tut-r1-grand'), [
        { clef: 'bass',   notes: bassNotes },
        { clef: 'treble', notes: trebleNotes },
      ], {
        hideNotes: true,
        layout: 'row',
        zoneColorsByClef: { treble: TREBLE_ZONE_COLORS, bass: BASS_ZONE_COLORS },
      });

      bottom.innerHTML = '<div class="tut-zone-legend" id="tut-r1-legend"></div><div id="tut-r1-piano"></div>';
      const legend = document.getElementById('tut-r1-legend');
      HAND_ZONES.forEach(z => {
        legend.insertAdjacentHTML('beforeend',
          `<span><i style="background:${z.hex}"></i>${z.hand} ${z.zone}</span>`);
      });
      const ctrl = buildTutPiano(document.getElementById('tut-r1-piano'), {
        onPress(note) { audio.unlock(); audio.playNote(note, 0.4); },
      });
      ctrl.setZoneBands(HAND_ZONES.map(z => ({ fromNote: z.from, toNote: z.to, color: hexToRgba(z.hex, 0.4) })));
    },
  },
  {
    chip: '규칙 2',
    caption: '셀 너비 = 음 길이, 테두리 색 = 박자',
    render(top, bottom) {
      const rule23Notes = [
        { pitch: 'G4', duration: 0.5, beat: 1 },
        { pitch: 'C5', duration: 1,   beat: 2 },
        { pitch: 'E4', duration: 2,   beat: 3 },
      ];
      top.innerHTML = `
        <div class="notation-container" id="tut-r23-notation"></div>
        <div class="tut-zone-legend" id="tut-beat-legend"></div>`;
      // 존별 다른 색 + 음표 글자 숨김은 규칙1 전용 — 규칙2는 기본 반투명 존 배경 +
      // 음표 글자(G, C, E) 표시 그대로.
      renderNotation(document.getElementById('tut-r23-notation'), rule23Notes, {});
      const legend = document.getElementById('tut-beat-legend');
      [1, 2, 3, 4].forEach(b => {
        legend.insertAdjacentHTML('beforeend', `<span><i style="background:${BEAT_COLORS[b]}"></i>${b}박</span>`);
      });

      bottom.innerHTML = '<div class="tut-practice-dock" id="tut-practice-dock"></div>';
      buildPracticeDock(document.getElementById('tut-practice-dock'), rule23Notes.map(n => n.pitch));
    },
  },
  {
    chip: '규칙 3',
    caption: '화음 = 두 음을 동시에 눌러요',
    render(top, bottom) {
      const chordNote = { pitch: 'C4', duration: 2, beat: 1, chordNotes: ['E4'] };
      top.innerHTML = '<div class="notation-container" id="tut-chord-notation"></div>';
      renderNotation(document.getElementById('tut-chord-notation'), [chordNote], {});

      bottom.innerHTML = '<div id="tut-chord-piano"></div>';
      const target = [chordNote.pitch, ...chordNote.chordNotes];
      const ctrl = buildTutPiano(document.getElementById('tut-chord-piano'), {
        onPress(note) { audio.unlock(); audio.playNote(note, 0.5); },
      });
      ctrl.setDots(target.map(n => ({ note: n, color: '#0076CE' })));
    },
  },
  buildQuizPage(1, { type: 'note', note: 'E4' }),
  buildQuizPage(2, { type: 'note', note: 'G#4' }),
  buildQuizPage(3, { type: 'note', note: 'C5' }),
  buildQuizPage(4, { type: 'chord', notes: ['C4', 'E4'] }),
  buildQuizPage(5, { type: 'sequence', notes: ['C4', 'D4', 'E4', 'F4'] }),
];

let tutPageIdx = 0;

function renderTutDots() {
  const wrap = document.getElementById('tut-dots');
  wrap.innerHTML = '';
  TUT_PAGES.forEach((_, i) => {
    const d = document.createElement('span');
    d.className = 'tut-dot' + (i === tutPageIdx ? ' active' : '');
    wrap.appendChild(d);
  });
}

// 화면 고정(상하 스크롤 금지) 원칙을 지키기 위해, 위/아래 박스 내용이 박스 높이보다
// 크면(특히 가로로 눕힌 짧은 화면에서 규칙1/2/3처럼 내용이 많은 페이지) 자동으로 축소해서
// 맞춘다. 페이지별로 값을 하드코딩하지 않고 실제 렌더된 높이를 재서 계산 — 어떤 화면
// 크기에서도 같은 원리로 동작한다. 좌우(피아노/오선) 스크롤은 그대로 유지.
function autoFitTutBoxes() {
  [document.getElementById('tut-split-top'), document.getElementById('tut-split-bottom')]
    .forEach(box => {
      if (!box) return;
      Array.from(box.children).forEach(c => { c.style.zoom = ''; });
      const overflow = box.scrollHeight - box.clientHeight;
      if (overflow > 4 && box.clientHeight > 0) {
        const zoom = Math.max(0.34, (box.clientHeight / box.scrollHeight) * 0.95);
        Array.from(box.children).forEach(c => { c.style.zoom = zoom; });
      }
    });
}

function renderTutPage(idx) {
  tutPageIdx = idx;
  resetShellScroll();
  document.getElementById('screen-tutorial')?.scrollTo(0, 0);
  state.tutorialPianoCtrl?.destroy();
  state.tutorialPianoCtrl = null;

  const page = TUT_PAGES[idx];
  document.getElementById('tut-page-chip').textContent = page.chip;
  document.getElementById('tut-caption').textContent = page.caption;
  const top = document.getElementById('tut-split-top');
  const bottom = document.getElementById('tut-split-bottom');
  top.innerHTML = '';
  bottom.innerHTML = '';
  document.getElementById('tut-page-frame')
    .classList.toggle('tut-split-row', page.splitDirection === 'row');
  page.render(top, bottom);
  autoFitTutBoxes();

  renderTutDots();
  // 랜딩에서 들어온 흐름은 첫 페이지 "이전"이 비활성화 대신 랜딩으로 나가는 종료 버튼이 됨
  const atFirst = idx === 0;
  const prevBtn = document.getElementById('tut-prev');
  prevBtn.disabled = atFirst && !state.flowFromLanding;
  prevBtn.textContent = atFirst && state.flowFromLanding ? '← 처음으로' : '← 이전';

  const atLast = idx === TUT_PAGES.length - 1;
  // 튜토리얼 단독 흐름(체험하기로 안 이어짐)에서는 마지막 페이지도 "변환 시작"이 아니라 처음으로
  const tutorialOnly = state.flowLock && !state.flowLock.includes('convert');
  document.getElementById('tut-next').textContent =
    atLast ? (tutorialOnly ? '이해했어요 — 처음으로' : '이해했어요 — 변환 시작하기 →') : '다음 →';

  // 규칙2(인덱스 3)부터 12음 참고 돋보기 노출 — 그 전 페이지들은 아직 12음/옥타브를
  // 처음 배우는 단계라 돋보기가 오히려 혼란을 줄 수 있어 숨긴다.
  setMagnifierVisible(idx >= 3);
}

// 버튼 누를 때마다 화면이 뚝 끊기지 않도록, 나가는 방향으로 살짝 페이드아웃 →
// 내용 교체 → 반대쪽에서 페이드인. 태블릿에서 체감이 크게 달라지는 부분이라 별도 함수로 분리.
const TUT_ANIM_MS = 180;
function goToTutPage(idx) {
  if (idx < 0 || idx >= TUT_PAGES.length || idx === tutPageIdx) return;
  const dir = idx > tutPageIdx ? 1 : -1; // 1=다음(왼쪽으로 퇴장), -1=이전(오른쪽으로 퇴장)
  const frame = document.getElementById('tut-page-frame');

  frame.style.transition = `opacity ${TUT_ANIM_MS}ms ease, transform ${TUT_ANIM_MS}ms ease`;
  frame.style.opacity = '0';
  frame.style.transform = `translateX(${dir * -18}px)`;

  setTimeout(() => {
    renderTutPage(idx);
    frame.style.transition = 'none';
    frame.style.transform = `translateX(${dir * 18}px)`;
    void frame.offsetWidth; // 강제 리플로우 — 위 스타일을 실제로 적용시킨 뒤 아래 트랜지션 시작
    frame.style.transition = `opacity ${TUT_ANIM_MS}ms ease, transform ${TUT_ANIM_MS}ms ease`;
    frame.style.opacity = '1';
    frame.style.transform = 'translateX(0)';
  }, TUT_ANIM_MS);
}

function initTutorial() {
  document.getElementById('tut-prev').addEventListener('click', () => {
    if (tutPageIdx > 0) { goToTutPage(tutPageIdx - 1); return; }
    if (state.flowFromLanding) exitFlowToLanding(); // 첫 페이지의 "이전" = 랜딩으로 나가기
  });
  document.getElementById('tut-next').addEventListener('click', () => {
    if (tutPageIdx < TUT_PAGES.length - 1) { goToTutPage(tutPageIdx + 1); return; }
    // 마지막 페이지: 튜토리얼 단독 흐름이면 랜딩으로, 아니면(자유 탐색/구 킨스크) 변환 화면으로
    if (state.flowLock && !state.flowLock.includes('convert')) exitFlowToLanding();
    else navigate('convert');
  });
  renderTutPage(0);
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
  document.getElementById('btn-show-qr').addEventListener('click', showQrForCurrentResult);

  // 전시 부스 진행요원 전용 — 실시간 AI 인식이 조명 등으로 꼬였을 때 즉시 대체 결과로 전환.
  document.getElementById('btn-cheat-fallback')?.addEventListener('click', () => {
    const s = SAMPLES.find(x => x.id === 'sample_butterfly') || SAMPLES[0];
    const data = sampleToNotation(s, { id: generateId(), createdAt: Date.now() });
    showLoading(); // output-placeholder를 확실히 숨긴 뒤(동기 실행) 바로 결과로 전환
    showResult(data);
  });

  initCameraCapture();
}

// 전시 부스: 방금 변환한 악보를 방문자 폰으로 가져갈 수 있게 QR 코드를 띄운다.
function showQrForCurrentResult() {
  const data = state.convertResult;
  if (!data?.id) return;
  const panel = document.getElementById('qr-panel');
  const img   = document.getElementById('qr-img');
  const shareUrl = `${location.origin}${location.pathname.replace(/[^/]*$/, '')}index.html?score=${encodeURIComponent(data.id)}`;
  img.src = `/api/qr?data=${encodeURIComponent(shareUrl)}`;
  panel.classList.toggle('hidden');
}

// 폰 카메라로 악보를 직접 촬영해서 업로드 — 기존 handleUpload() 흐름에 그대로 합류시킨다.
// lib/screens/guided_camera_screen.dart의 뷰파인더 가이드를 그대로 이식 — 오선 1개/대보표(2개)
// 토글, 어두운 마스크 + 코너 브래킷 + 실제 오선 줄 가이드, 촬영 시 그 박스 영역만 크롭해서
// 넘긴다(자동 오선 검출에 온전히 맡기지 않고 촬영 UX로 프레이밍을 유도 — 인식률에 직결).
const CAMERA_GUIDE_W_FRAC = 0.88;
function cameraGuideHFrac(grandStaff) { return grandStaff ? 0.34 : 0.16; }

function cameraGuideRectNative(vw, vh, grandStaff) {
  const w = vw * CAMERA_GUIDE_W_FRAC;
  const h = vh * cameraGuideHFrac(grandStaff);
  return { x: (vw - w) / 2, y: (vh - h) / 2, w, h };
}

// object-fit:contain으로 보이는 비디오는 wrap 안에서 letterbox될 수 있으므로,
// native(원본 해상도) 좌표 사각형을 실제 화면에 그려질 위치로 변환한다.
function cameraNativeToDisplay(rect, vw, vh, ww, wh) {
  const scale = Math.min(ww / vw, wh / vh);
  const offX = (ww - vw * scale) / 2;
  const offY = (wh - vh * scale) / 2;
  return { x: offX + rect.x * scale, y: offY + rect.y * scale, w: rect.w * scale, h: rect.h * scale };
}

function cameraRoundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function cameraDrawStaffLines(ctx, x, y, w, h) {
  const gap = h / 4;
  for (let i = 0; i < 5; i++) {
    const ly = y + gap * i;
    ctx.beginPath(); ctx.moveTo(x, ly); ctx.lineTo(x + w, ly); ctx.stroke();
  }
}

function drawCameraGuideOverlay(canvas, wrap, video, grandStaff) {
  const ww = wrap.clientWidth, wh = wrap.clientHeight;
  if (!ww || !wh) return;
  canvas.width = ww; canvas.height = wh;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, ww, wh);

  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  const rect = cameraNativeToDisplay(cameraGuideRectNative(vw, vh, grandStaff), vw, vh, ww, wh);

  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(0, 0, ww, wh);
  ctx.save();
  cameraRoundRectPath(ctx, rect.x, rect.y, rect.w, rect.h, 8);
  ctx.clip();
  ctx.clearRect(0, 0, ww, wh);
  ctx.restore();

  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2.5;
  cameraRoundRectPath(ctx, rect.x, rect.y, rect.w, rect.h, 8);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.lineWidth = 1.6;
  if (grandStaff) {
    cameraDrawStaffLines(ctx, rect.x, rect.y + rect.h * 0.10, rect.w, rect.h * 0.32);
    cameraDrawStaffLines(ctx, rect.x, rect.y + rect.h * 0.58, rect.w, rect.h * 0.32);
  } else {
    cameraDrawStaffLines(ctx, rect.x, rect.y + rect.h * 0.30, rect.w, rect.h * 0.40);
  }

  const len = 22;
  ctx.strokeStyle = '#0076CE';
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  [[rect.x, rect.y, 1, 1], [rect.x + rect.w, rect.y, -1, 1],
   [rect.x, rect.y + rect.h, 1, -1], [rect.x + rect.w, rect.y + rect.h, -1, -1]]
    .forEach(([cx, cy, sx, sy]) => {
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + len * sx, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + len * sy); ctx.stroke();
    });
}

// 가이드 오버레이 촬영 UI 조립 — 기존 변환 화면과 체험하기 화면 둘 다 이 함수로 만든다
// (ids만 다르게, 촬영 완료 시 onCaptured(file)로 각자 원하는 곳으로 라우팅).
function setupCameraCapture(ids, onCaptured) {
  const openBtn     = document.getElementById(ids.openBtn);
  const cancelBtn   = document.getElementById(ids.cancelBtn);
  const shutterBtn  = document.getElementById(ids.shutterBtn);
  const captureBox  = document.getElementById(ids.captureBox);
  const video       = document.getElementById(ids.video);
  const canvas      = document.getElementById(ids.canvas);
  const guideCanvas = document.getElementById(ids.guideCanvas);
  const guideHint   = document.getElementById(ids.guideHint);
  const errorEl     = document.getElementById(ids.error);
  if (!openBtn || !captureBox) return;
  const guideWrap = captureBox.querySelector('.camera-video-wrap');
  const modeChips = captureBox.querySelectorAll('.camera-mode-chip');

  let grandStaff = true; // flutter 쪽 기본값과 동일 — 피아노 악보는 대보표가 더 흔함

  function updateHint() {
    guideHint.textContent = grandStaff
      ? '대보표(높은음자리+낮은음자리)를 박스 안에 맞춰주세요'
      : '오선 하나를 박스 안에 맞춰주세요';
  }
  function redrawGuide() { drawCameraGuideOverlay(guideCanvas, guideWrap, video, grandStaff); }

  modeChips.forEach(chip => {
    chip.addEventListener('click', () => {
      grandStaff = chip.dataset.grand === '1';
      modeChips.forEach(c => c.classList.toggle('active', c === chip));
      updateHint();
      redrawGuide();
    });
  });

  video.addEventListener('loadedmetadata', redrawGuide);
  new ResizeObserver(redrawGuide).observe(guideWrap);

  function cleanup() {
    state.cameraStream?.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
    video.srcObject = null;
    captureBox.classList.add('hidden');
    state.activeCameraStop = null;
  }

  async function openCamera() {
    errorEl.classList.add('hidden');
    captureBox.classList.remove('hidden');
    state.activeCameraStop = cleanup;
    updateHint();

    if (!navigator.mediaDevices?.getUserMedia) {
      errorEl.textContent = '이 브라우저/연결에서는 카메라를 쓸 수 없습니다 (HTTPS 또는 localhost 접속이 필요해요) — 파일 선택을 이용해주세요';
      errorEl.classList.remove('hidden');
      return;
    }
    try {
      state.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1440 } },
        audio: false,
      });
      video.srcObject = state.cameraStream;
    } catch (e) {
      errorEl.textContent = '카메라 권한을 허용해주세요: ' + e.message;
      errorEl.classList.remove('hidden');
    }
  }

  openBtn.addEventListener('click', openCamera);
  cancelBtn.addEventListener('click', () => state.activeCameraStop?.());

  shutterBtn.addEventListener('click', () => {
    if (!state.cameraStream || !video.videoWidth) return;
    const rect = cameraGuideRectNative(video.videoWidth, video.videoHeight, grandStaff);
    canvas.width  = rect.w;
    canvas.height = rect.h;
    canvas.getContext('2d').drawImage(video, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);
    canvas.toBlob(blob => {
      if (!blob) return;
      state.activeCameraStop?.();
      onCaptured(new File([blob], 'capture.jpg', { type: 'image/jpeg' }));
    }, 'image/jpeg', 0.92);
  });
}

function initCameraCapture() {
  setupCameraCapture({
    openBtn: 'btn-camera-open', cancelBtn: 'btn-camera-cancel', shutterBtn: 'btn-camera-shutter',
    captureBox: 'camera-capture', video: 'camera-video', canvas: 'camera-canvas',
    guideCanvas: 'camera-guide-canvas', guideHint: 'camera-guide-hint', error: 'camera-error',
  }, handleUpload);
}

// SAMPLES 항목을 화면에서 쓰는 notation 데이터 모양으로 변환 — staves(그랜드 스태프)가
// 있으면 notes(=treble, 재생/미리보기용)도 함께 채운다. 이 변환을 여러 곳에서 각자
// 손으로 반복하다가 한 곳(연주하기 초기 자동 로드)에서 staves를 빠뜨렸던 적이 있어
// (반주 모드가 항상 비활성으로 보이는 버그로 발견) 한 곳으로 모았다.
function sampleToNotation(s, overrides = {}) {
  const base = { title: s.title, tempo: s.tempo, timeSignature: s.timeSignature };
  return s.staves
    ? { ...base, staves: s.staves, notes: s.staves[0].notes, ...overrides }
    : { ...base, notes: s.notes, ...overrides };
}

function startConversion(sample) {
  showLoading();
  const data = sampleToNotation(sample, { id: generateId(), createdAt: Date.now() });
  setTimeout(() => showResult(data), 2200);
}

async function handleUpload(file) {
  if (!file.type.startsWith('image/')) { toast('이미지 파일을 선택해주세요'); return; }
  if (file.size > 10 * 1024 * 1024)   { toast('파일 크기가 10MB를 초과합니다'); return; }
  showLoading();

  const model = document.getElementById('model-select')?.value || 'andromr';
  const form  = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`/api/recognize?model=${model}`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `서버 오류 (${res.status})`);
    }
    const json = await res.json();
    json._fromServer = true; // QR "테이크아웃" 버튼은 서버에 실제 저장된 결과에만 노출
    showResult(json);
  } catch (e) {
    if (e instanceof TypeError) {
      // 서버 미연결 → 샘플 데이터로 폴백
      toast('⚠️ OMR 서버 미연결 — 샘플로 시연합니다');
      const demo = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
      const data = sampleToNotation(demo, {
        id: generateId(), title: file.name.replace(/\.[^.]+$/, ''), createdAt: Date.now(),
      });
      setTimeout(() => showResult(data), 800);
    } else {
      document.getElementById('output-loading').classList.add('hidden');
      document.getElementById('output-placeholder').classList.remove('hidden');
      toast('❌ ' + e.message);
    }
  }
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

  document.getElementById('qr-panel')?.classList.add('hidden');
  const qrBtn = document.getElementById('btn-show-qr');
  qrBtn?.classList.toggle('hidden', !data._fromServer);

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
  if (state.user) saveScoreCloud(toSave).catch(console.error);
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
    const notation = sampleToNotation(s, { id: s.id, createdAt: 0 });
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
    const notation = sampleToNotation(s, { id: s.id, createdAt: 0 });
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

  document.getElementById('toggle-accompaniment').addEventListener('change', e => {
    state.accompanimentMode = e.target.checked;
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

        // Wait Mode 정답 — 악보 쪽도 짧게 초록 이펙트 (스코어가 스르륵 넘어가기 전 성취감 피드백)
        const ncOk = document.getElementById('play-notation');
        if (ncOk) {
          ncOk.classList.add('notation-correct-flash');
          setTimeout(() => ncOk.classList.remove('notation-correct-flash'), 400);
        }

        const bpm    = parseInt(document.getElementById('tempo-slider').value);
        const beatMs = (60 / bpm) * expNote.duration * 1000;

        state.playedIdx = expIdx;
        renderPlay(expIdx, -1);

        showCountdown(beatMs, () => setExpectedNote(expIdx + 1));
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

// 실제 사진에서 변환된 악보는 쉼표를 포함한다 — 쉼표는 키를 누를 필요가 없으므로
// 여기서 다음 "실제로 눌러야 하는" 음표를 찾을 때까지 쉼표를 박자 길이만큼만
// 자동으로 건너뛴다 (연속된 쉼표도 재귀적으로 처리).
function setExpectedNote(idx) {
  if (!state.playNotation) return;
  const notes = state.playNotation.notes;
  if (idx >= notes.length) {
    state.playNoteIdx = -1;
    state.playedIdx   = -1;
    state.playPianoCtrl?.clearExpected();
    renderPlay(-1, -1);
    toast('🎉 연주 완료! 처음부터 다시 하려면 ⏹ 버튼을 누르세요');
    return;
  }

  const note = notes[idx];
  state.playNoteIdx = idx;
  state.playedIdx   = -1;
  renderPlay(-1, idx);

  if (note.isRest) {
    state.playPianoCtrl?.clearExpected();
    const bpm    = parseInt(document.getElementById('tempo-slider').value);
    const restMs = (60 / bpm) * note.duration * 1000;
    showCountdown(restMs, () => setExpectedNote(idx + 1));
  } else {
    state.playPianoCtrl?.setExpected(note.pitch);
  }
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

  // 반주 모드는 왼손(베이스) 보표가 있을 때만 의미가 있음 — 없으면 꺼서 숨긴다.
  const hasBass = !!notation.staves?.[1];
  const accompToggle = document.getElementById('toggle-accompaniment');
  const accompWrap    = document.getElementById('accompaniment-toggle-wrap');
  if (accompToggle) {
    accompToggle.disabled = !hasBass;
    if (!hasBass) { accompToggle.checked = false; state.accompanimentMode = false; }
  }
  accompWrap?.classList.toggle('hidden', !hasBass);

  setExpectedNote(0);
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

    const staffMeta = state.playNotation.staves.map((stave, si) => ({
      label: si === 0 ? '🎵 높은음자리 (Treble)' : '🎻 낮은음자리 (Bass)',
      color: si === 0 ? '#0076CE' : '#5BB8F5',
      notes: stave.notes,
      clef:  stave.clef ?? (si === 0 ? 'treble' : 'bass'),
      interactive: si === 0,
    }));

    let trebleCtrl = null;
    staffMeta.forEach(({ label, color, notes, clef, interactive }) => {
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
        clef,
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

// 전자 피아노(MIDI) 연결 UI — 한 번만 초기화. 입력 콜백은 그때그때의
// state.playPianoCtrl을 참조하므로 연주하기 화면을 다시 그려도(피아노 재생성) 계속 동작한다.
function initMidiControls() {
  const btn       = document.getElementById('btn-midi-connect');
  const dot       = document.getElementById('midi-dot');
  const statusEl  = document.getElementById('midi-status');
  const inputSel  = document.getElementById('midi-input-select');
  const outputSel = document.getElementById('midi-output-select');
  if (!btn) return;

  function fillSelect(sel, devices, emptyLabel) {
    sel.innerHTML = '';
    if (devices.length === 0) {
      sel.innerHTML = `<option value="">${emptyLabel}</option>`;
      sel.disabled = true;
      return;
    }
    sel.disabled = false;
    devices.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.id;
      opt.textContent = d.name || d.id;
      sel.appendChild(opt);
    });
  }

  function connectSelected() {
    if (inputSel.value) {
      midi.setInput(inputSel.value, {
        onNoteOn:  pitch => state.playPianoCtrl?.press(pitch, { silent: true }),
        onNoteOff: pitch => state.playPianoCtrl?.release(pitch),
      });
    }
    midi.setOutput(outputSel.value || null);
  }

  btn.addEventListener('click', async () => {
    if (!midi.isSupported()) {
      statusEl.textContent = '이 브라우저는 Web MIDI를 지원하지 않아요 (Chrome/Edge + HTTPS 또는 localhost 필요)';
      return;
    }
    try {
      await midi.requestAccess();
      const inputs  = midi.listInputs();
      const outputs = midi.listOutputs();
      fillSelect(inputSel,  inputs,  '입력 기기 없음');
      fillSelect(outputSel, outputs, '출력 기기 없음');
      inputSel.classList.remove('hidden');
      outputSel.classList.remove('hidden');
      connectSelected();
      if (inputs.length || outputs.length) {
        dot.classList.add('connected');
        statusEl.textContent = `입력: ${inputs[0]?.name ?? '없음'} · 출력: ${outputs[0]?.name ?? '없음'}`;
      } else {
        statusEl.textContent = 'USB로 피아노를 연결한 뒤 다시 눌러주세요';
      }
    } catch (e) {
      statusEl.textContent = 'MIDI 연결 실패: ' + e.message;
    }
  });

  inputSel.addEventListener('change', connectSelected);
  outputSel.addEventListener('change', connectSelected);
}

// 자동재생 한 음마다: 화면 건반을 잠깐 파랗게 켜고, 전자 피아노가 연결돼 있으면
// MIDI로도 실제 소리를 낸다. 미리듣기/반주 모드 둘 다 이 함수를 공유한다.
function flashAutoPlayNote(note, bpm) {
  const k = document.querySelector(`[data-note="${note.pitch}"]`);
  if (k) {
    const isB = k.classList.contains('bk');
    k.style.background = '#0076CE';
    setTimeout(() => { k.style.background = isB ? '#1c1c1c' : '#f4efe6'; },
      note.duration * (60 / bpm) * 900);
  }
  if (midi.hasOutput() && note.pitch) {
    midi.sendNoteOn(note.pitch, 100);
    setTimeout(() => midi.sendNoteOff(note.pitch), note.duration * (60 / bpm) * 900);
  }
}

function toggleAutoPlay() {
  const btn = document.getElementById('btn-auto-play');
  if (state.playCancel) {
    stopAutoPlay();
    return;
  }
  if (!state.playNotation) return;
  const bpm = parseInt(document.getElementById('tempo-slider').value);

  // 반주 모드: 왼손(베이스)만 처음부터 자체 템포로 독립 재생 — 오른손 Wait Mode
  // 상태(state.playNoteIdx, 기대 음 하이라이트 등)는 그대로 둬서 동시에 연주할 수 있게 한다.
  const bassNotes = state.playNotation.staves?.[1]?.notes;
  if (state.accompanimentMode && bassNotes) {
    btn.textContent = '⏸';
    state.playCancel = audio.playSequence(
      bassNotes, bpm,
      stepIdx => flashAutoPlayNote(bassNotes[stepIdx], bpm),
      () => {
        state.playCancel = null;
        btn.textContent  = '▶';
        toast('🎼 반주 재생 완료');
      },
    );
    return;
  }

  // 미리듣기: 트레블(또는 단일 보표) 전체를 자동 재생.
  btn.textContent = '⏸';
  clearCountdown();
  state.playPianoCtrl?.clearExpected();

  const start = Math.max(0, state.playNoteIdx < 0 ? 0 : state.playNoteIdx);
  const notes = state.playNotation.notes.slice(start);

  state.playCancel = audio.playSequence(
    notes, bpm,
    stepIdx => {
      const gIdx = start + stepIdx;
      state.playNoteIdx = gIdx;
      state.playedIdx   = gIdx;
      renderPlay(gIdx, -1);
      state.playPianoCtrl?.clearExpected();
      flashAutoPlayNote(state.playNotation.notes[gIdx], bpm);
    },
    () => {
      state.playCancel = null;
      btn.textContent  = '▶';
      toast('🎵 재생 완료');
    },
  );
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
  state.playPianoCtrl?.clearHighlights();
  if (state.playNotation) setExpectedNote(0);
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
        deleteNotation(id);
        if (state.user) deleteScoreCloud(id).catch(console.error);
        toast('삭제되었습니다'); initLibraryScreen();
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Firebase Auth UI
// ═══════════════════════════════════════════════════════════════════════════════

function updateAuthUI(user) {
  state.user = user;
  const area = document.getElementById('auth-area');
  if (!area) return;

  if (user) {
    const avatar = user.photoURL
      ? `<img src="${user.photoURL}" class="user-avatar" alt="" referrerpolicy="no-referrer">`
      : '';
    const name = user.displayName || user.email || '사용자';
    area.innerHTML = `
      <div class="user-chip">
        ${avatar}
        <span class="user-name">${name}</span>
        <button class="btn-logout" id="btn-logout">로그아웃</button>
      </div>`;
    document.getElementById('btn-logout').addEventListener('click', () => {
      signOutUser().catch(console.error);
    });
    syncFromCloud().catch(console.error);
  } else {
    area.innerHTML = `<button class="btn-login" id="btn-login">🔑 구글 로그인</button>`;
    document.getElementById('btn-login').addEventListener('click', () => {
      signInWithGoogle().catch(err => {
        if (err.code !== 'auth/popup-closed-by-user') toast('로그인 실패: ' + err.message);
      });
    });
  }
}

async function checkServerStatus() {
  const dot    = document.getElementById('server-dot');
  const badge  = document.getElementById('omr-badge');
  const msg    = document.getElementById('omr-status-msg');
  try {
    const res  = await fetch('/api/status');
    const s    = await res.json();
    const both = s.andromr && s.custom;
    const any  = s.andromr || s.custom;
    if (dot)   { dot.classList.add('connected'); dot.title = '서버 연결됨'; }
    if (badge) badge.textContent = '✅ OMR 서버 연결됨';
    if (msg)   msg.textContent = `Andromr: ${s.andromr ? '✓' : '✗'}  커스텀: ${s.custom ? '✓' : '✗'}`;
    // 연결된 모델만 select에 표시
    const sel = document.getElementById('model-select');
    if (sel) {
      [...sel.options].forEach(opt => {
        opt.disabled = !s[opt.value];
        if (opt.disabled && sel.value === opt.value) sel.value = s.andromr ? 'andromr' : 'custom';
      });
    }
  } catch {
    if (dot)   dot.title = '서버 미연결';
    if (badge) badge.textContent = '🔴 서버 미연결';
    if (msg)   msg.textContent = 'python server.py 실행 후 새로고침하면 실제 OMR이 동작합니다';
  }
}

async function syncFromCloud() {
  try {
    const cloudScores = await loadScoresCloud();
    cloudScores.forEach(score => saveNotation(score));
    if (state.screen === 'library') initLibraryScreen();
  } catch (e) {
    console.error('[Firebase] 동기화 오류:', e);
  }
}

// ── 전역 이벤트 위임 ───────────────────────────────────────────────────────────
document.addEventListener('click', e => {
  const nav = e.target.dataset.nav || e.target.closest('[data-nav]')?.dataset.nav;
  if (nav) { wheelLocked = false; navigate(nav); return; }

  // 하단 탭바 클릭(아이콘/라벨 span 클릭 포함)은 wheelLocked 무시하고 즉시 전환
  const tabBtn = e.target.closest('.nav-tab');
  const scr = tabBtn?.dataset.screen ?? e.target.dataset.screen;
  if (scr) {
    if (tabBtn) wheelLocked = false;
    navigate(scr);
  }
});

// ── 부팅 ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // 전시 부스 킨스크 모드 — 태블릿1: ?kiosk=1(튜토리얼 고정), 태블릿2: ?kiosk=convert
  // (악보 변환 화면 고정). 지정한 화면 하나만 남기고 다른 화면/로그인은 다 숨긴다.
  const kioskParam = new URLSearchParams(location.search).get('kiosk');
  state.kioskMode = kioskParam === null ? false : (kioskParam === 'convert' ? 'convert' : 'tutorial');
  document.body.classList.toggle('kiosk-mode', !!state.kioskMode);
  if (state.kioskMode) document.body.dataset.kiosk = state.kioskMode;

  // 첫 화면은 transition 없이 즉시 표시 — HTML에는 tutorial에만 active가 박혀있으므로
  // (kiosk=convert일 때) 다른 화면을 켜기 전에 먼저 다 꺼준다.
  const firstScreenName = state.kioskMode === 'convert' ? 'convert' : 'tutorial';
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const firstScreen = document.getElementById('screen-' + firstScreenName);
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
  notationNavUpdate.convert  = makeNotationNav('convert-notation',  'convert-notation-prev',  'convert-notation-next');
  notationNavUpdate.play     = makeNotationNav('play-notation',     'play-notation-prev',     'play-notation-next');

  onAuthChange(updateAuthUI);
  checkServerStatus();
  initTutorial();
  initConvert();
  initMidiControls();
  initOctaveMagnifier();
  initLanding();
  initExpFlow();
  initAboutModal();

  // navigate는 이미 active를 설정했으므로 state만 맞춤
  function settleOn(name) {
    state.screen = name;
    document.querySelectorAll('.nav-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.screen === name));
  }

  if (state.kioskMode) {
    // 전시 부스 물리 킨스크(URL) — 랜딩 없이 지정된 화면으로 바로 고정
    settleOn(firstScreenName);
    hideLanding();
  } else {
    loadSharedScoreFromQuery().then(loaded => {
      if (loaded) hideLanding(); // QR로 결과를 바로 열람 — 랜딩 건너뜀 (navigate('convert')는 그 안에서 이미 처리됨)
      else settleOn('tutorial'); // 기본 상태 — 기본 상태(HTML 기본값)인 랜딩이 그대로 보임
    });
  }
});

// 전시 부스 QR "테이크아웃" — ?score=<id>로 접속하면 그 결과를 바로 열어서 보여준다
// (서버가 /api/recognize 성공 시 함께 저장해 둔 결과를 조회).
async function loadSharedScoreFromQuery() {
  const id = new URLSearchParams(location.search).get('score');
  if (!id) return false;
  try {
    const res = await fetch(`/api/score/${encodeURIComponent(id)}`);
    if (!res.ok) { toast('⚠️ 공유된 악보를 찾을 수 없습니다 (만료되었을 수 있어요)'); return false; }
    const data = await res.json();
    navigate('convert', { instant: true });
    showResult(data);
    toast('🎁 나만의 커스텀 악보를 불러왔어요!');
    return true;
  } catch {
    return false;
  }
}
