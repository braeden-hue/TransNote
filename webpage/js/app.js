import { SAMPLES, BEAT_COLORS, noteToFrequency } from './samples.js';
import { audio }                           from './audio.js';
import { renderNotation, renderGrandStaff, renderDigitalStaff } from './notation.js';
import { buildPiano, renderLabeledOctave } from './piano.js';
import { recognizeImage }                  from './recognize.js';
import { loadAll, saveNotation, deleteNotation, generateId, getById } from './storage.js';
import { signInWithGoogle, signOutUser, onAuthChange,
         saveScoreCloud, loadScoresCloud, deleteScoreCloud,
         isNicknameTakenInSong, saveLeaderboardEntryCloud, loadLeaderboardCloud } from './firebase.js';
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
  expPlayCancel:     [],    // 체험하기 "연주 듣기" 중인 재생의 취소 함수들(대보표면 최대 2개) — 중첩 재생 방지용
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

// 킨스크(?kiosk=)별로 벗어날 수 없는 화면 집합. 태블릿2(convert)는 저장/이어붙이기 +
// 다른 참여자가 저장해둔 악보 목록(내 악보함)/연주까지 오갈 수 있어야 해서 convert
// 하나가 아니라 셋을 다 허용(2026-08-10) -- 태블릿1(tutorial)은 기존대로 그대로 고정.
const KIOSK_ALLOWED_SCREENS = {
  tutorial: ['tutorial'],
  convert:  ['convert', 'library', 'play'],
};
let wheelLocked = false;

// 화면 전환 컨테이너(.app-main/.screens-wrapper)는 항상 scrollTop=0이어야 하는데,
// 키보드 포커스 이동 등으로 브라우저가 자동 스크롤을 걸어버리는 경우가 있어(특히
// 콘텐츠가 긴 화면에서) 화면이 바뀔 때마다 강제로 되돌린다.
function resetShellScroll() {
  document.querySelector('.app-main')?.scrollTo(0, 0);
  document.querySelector('.screens-wrapper')?.scrollTo(0, 0);
}

function navigate(name, { instant = false } = {}) {
  if (state.kioskMode && !KIOSK_ALLOWED_SCREENS[state.kioskMode]?.includes(name)) return; // 킨스크: 허용된 화면 밖으로 못 나가게 고정
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
// 모바일에서 세로/가로 회전 시 resize 이벤트가 새 크기 확정 전에 먼저 발생하거나 아예
// 안 오는 경우가 있어(브라우저·OS별로 제각각), orientationchange도 별도로 듣고 값이
// 안정된 뒤(300ms, 카메라 가이드 재계산과 동일 지연) 다시 계산한다 — "세로/가로에 따라
// 화면 크기가 어긋나 보이는" 버그의 주 원인이었다.
window.addEventListener('orientationchange', () => setTimeout(fitLandingFrame, 300));

// ── 화면 전환 연출: 누른 버튼 쪽으로 카메라가 줌인하는 느낌의 확대 전환 ──────────────
// 랜딩(메인 화면) 자체가 그 버튼 위치를 중심으로 확대되며 흐려지고(under 레이어), 그
// 위로 목적지 사진(튜토.png/plus.png, over 레이어)이 한 박자 늦게 서서히 크로스페이드로
// 나타나 또렷해진다 — "곧바로 다음 이미지로 넘어가는" 게 아니라 메인 화면에서 자연스럽게
// 이어지는 줌인처럼 보이게 하는 게 핵심. 두 가지 쓰임이 있다:
//   - stay:false(기본, 튜토리얼) — 다 다가간 시점에 실제 목적지 화면으로 바꿔치기하고
//     사진을 다시 걷어내며 그 화면을 드러낸다.
//   - stay:true(AI 모델) — 다 다가간 뒤에도 사진을 그대로 띄워둔 채(그 앞에 도착해서 멈춰
//     선 상태) 모달만 그 위에 띄운다. playWalkOut()으로 축소하며 다시 랜딩으로 돌아간다.
// CSS(.walkin-overlay)의 트랜지션 시간(WALKIN_MS)과 반드시 맞물려야 한다.
const WALKIN_MS = 1800;
let walkinInProgress = false;
function playWalkIn(imgSrc, { onArrive, stay = false, originXPct = 50, originYPct = 50 } = {}) {
  if (walkinInProgress) return;
  walkinInProgress = true;
  const overlay = document.getElementById('walkin-overlay');
  const under   = document.getElementById('walkin-img-under');
  const over    = document.getElementById('walkin-img-over');
  const origin  = `${originXPct}% ${originYPct}%`;
  under.style.transformOrigin = origin;
  over.style.transformOrigin  = origin;
  over.src = imgSrc;
  overlay.classList.remove('walking');
  overlay.classList.add('visible');
  void overlay.offsetWidth; // 강제 리플로우 — walking 트랜지션이 매번 처음부터 재생되게
  requestAnimationFrame(() => overlay.classList.add('walking'));
  setTimeout(() => {
    onArrive();
    if (!stay) {
      overlay.classList.remove('visible', 'walking');
      walkinInProgress = false;
    }
    // stay:true면 walkinInProgress를 계속 true로 둬서(=사진이 떠 있는 동안) 다른 핫스팟이
    // 끼어들지 못하게 막는다 — playWalkOut()이 끝나야 풀린다.
  }, WALKIN_MS);
}

// stay:true로 멈춰 서 있던 사진을 다시 축소하며 걷어내고 랜딩으로 돌아간다.
function playWalkOut() {
  const overlay = document.getElementById('walkin-overlay');
  overlay.classList.remove('walking'); // 확대 상태 해제 → 원래 크기로 줄어드는 트랜지션 재생
  setTimeout(() => {
    overlay.classList.remove('visible');
    walkinInProgress = false;
  }, WALKIN_MS);
}

function initLanding() {
  fitLandingFrame();
  document.querySelectorAll('.landing-hotspot').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.landing;
      // 버튼 위치(핫스팟 %) = 줌인 카메라가 향하는 지점(transform-origin)
      const originXPct = parseFloat(btn.style.left) || 50;
      const originYPct = parseFloat(btn.style.top) || 50;
      if (target === 'about') { showAboutModal(); return; }
      if (target === 'ai') {
        // 도착해서 멈춰 선 채(stay:true) 모달을 자동으로 띄우지 않는다 — 책 위 별도
        // 핫스팟(#walkin-book-hotspot)을 직접 눌러야 모델 설명 책이 열린다.
        playWalkIn('assets/walkin-desk.jpg', {
          stay: true, originXPct, originYPct,
          onArrive: () => document.getElementById('walkin-overlay')?.classList.add('desk-stay'),
        });
        return;
      }
      if (target === 'tutorial') {
        playWalkIn('assets/walkin-piano.jpg', {
          originXPct, originYPct,
          onArrive: () => {
            hideLanding();
            enterFlow(['tutorial']);
            renderTutPage(0);
            navigate('tutorial', { instant: true });
          },
        });
        return;
      }
      hideLanding();
      if (target === 'experience') showExpSelect();
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
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hideAboutModal(); });
}

// ── TransNote(책상) 워크인 — 도착해서 멈춰 선 상태 나가기 + 모델 설명 책 ──────────────
// "← 처음으로"는 desk-stay 상태(책상 사진 앞에 멈춰 서 있는 동안)에만 보이고, 누르면
// 책이 열려 있었다면 먼저 닫고 playWalkOut()으로 축소하며 랜딩으로 돌아간다.
function hideDeskStay() {
  const overlay = document.getElementById('walkin-overlay');
  if (!overlay || !overlay.classList.contains('desk-stay')) return;
  hideBookModal();
  overlay.classList.remove('desk-stay');
  playWalkOut();
}

function showBookModal() { document.getElementById('book-modal')?.classList.remove('hidden'); }
function hideBookModal() { document.getElementById('book-modal')?.classList.add('hidden'); }

function initWalkinBack() {
  document.getElementById('walkin-back-btn')?.addEventListener('click', hideDeskStay);
  document.getElementById('walkin-book-hotspot')?.addEventListener('click', showBookModal);
  document.getElementById('book-close')?.addEventListener('click', hideBookModal);
  document.getElementById('book-modal-backdrop')?.addEventListener('click', hideBookModal);
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    const modal = document.getElementById('book-modal');
    if (modal && !modal.classList.contains('hidden')) { hideBookModal(); return; }
    hideDeskStay();
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// 체험하기 — 랜딩에서만 진입하는 완전히 독립된 흐름(기존 변환/연주 화면과 무관).
// 1) 샘플 3곡 + 촬영 버튼만 있는 심플한 선택 화면
// 2) 선택한 곡의 커스텀 악보 첫 마디 + 연주 듣기(전체 곡)/연주하기 + 순위표
// 3) 연주하기 진입 시: 전자피아노 연결 테스트 + 닉네임 입력 대기 화면
// 4) 연주 진행(전자피아노 연결 시 화면 건반 숨김) + 점수 결과
// ═══════════════════════════════════════════════════════════════════════════════
function stopExpPlayback() {
  state.expPlayCancel.forEach(cancel => cancel?.());
  state.expPlayCancel = [];
  audio.stopAll(); // 취소 함수가 없어도(예: 아직 재생 시작 전) 혹시 울리고 있는 음까지 확실히 정지
  document.getElementById('exp-play-btn')?.classList.remove('playing');
}

function hideExpScreens() {
  stopExpPlayback(); // 화면이 바뀌면(뒤로가기 포함) 재생 중이던 곡은 자동으로 멈춘다
  closeExpCapturePreview(); // 촬영 미리보기 도중 나가도 오버레이/objectURL이 남지 않게
  ['screen-exp-select', 'screen-exp-score', 'screen-exp-handmode', 'screen-exp-wait', 'screen-exp-perform']
    .forEach(id => document.getElementById(id)?.classList.add('hidden'));
}

function showExpSelect() {
  hideExpScreens();
  setMagnifierVisible(false);
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

// ── 곡별 순위표(상위 3명) — Firestore leaderboard 컬렉션(로그인 불필요, 부스 회전율
// 때문에 로그인 게이트를 뺀 결정, docs/PLAN_booth_companion_page.md 참고). 닉네임
// 유일성은 songKey(곡) 안에서만 검사 — 다른 곡에서는 같은 닉네임을 다시 써도 됨
// (2026-08-10 확정). Firebase 미설정 환경(로컬 개발 등)에서는 firebase.js 쪽 함수들이
// 조용히 빈 값/false를 반환해 화면이 비어있을 뿐 에러로 죽지 않는다.

async function renderLeaderboard(songKey) {
  const el = document.getElementById('exp-leaderboard-list');
  if (!el) return;
  let list = [];
  try {
    list = await loadLeaderboardCloud(songKey, 3);
  } catch (e) {
    // 네트워크/권한 오류 등 -- 빈 목록으로 표시하되, 원인 파악용으로 콘솔에는 남긴다
    // (Firestore 보안규칙 미설정이 원인인 경우가 많음 — firestore.rules 참고).
    console.error('[renderLeaderboard] 순위표 로드 실패:', e);
  }
  el.innerHTML = list.length
    ? list.map((e, i) => `
        <li>
          <span class="exp-lb-rank">${i + 1}</span>
          <span class="exp-lb-name">${e.nickname}</span>
          <span class="exp-lb-score">${e.score}점</span>
        </li>`).join('')
    : '<li class="exp-leaderboard-empty">아직 기록이 없어요</li>';
}

// 첫 마디만 잘라서 보표(대보표/단일보표)를 그려넣는 공통 로직 — 악보 화면과
// 촬영 직후 미리보기(원본 사진과 비교)가 같은 방식으로 렌더링해야 하므로 공용화.
function renderFirstMeasureInto(container, data) {
  container.innerHTML = '';
  if (data.staves?.length >= 2) {
    const trimmed = data.staves.map(s => ({ ...s, notes: firstMeasure(s.notes) }));
    renderGrandStaff(container, trimmed, { noteColorMode: 'octave' });
  } else {
    renderNotation(container, firstMeasure(data.notes), { noteColorMode: 'octave' });
  }
}

function showExpScore(data) {
  state.expScoreData = data;
  hideExpScreens();
  document.getElementById('screen-exp-score')?.classList.remove('hidden');

  document.getElementById('exp-score-title').textContent = data.title || '';
  renderFirstMeasureInto(document.getElementById('exp-score-notation'), data);
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
// 회전 시 resize만으로는 못 잡는 경우가 있어(위 fitLandingFrame 주석 참고)
// orientationchange도 같이 듣는다 — 체험하기 악보/연주 화면 + 튜토리얼 박스 전부 해당.
function refitVisibleBoxesOnResize() {
  if (state.expScoreData) autoFitExpScore('exp-score-notation');
  if (state.expPerform)   autoFitExpScore('exp-perform-notation');
  if (state.screen === 'tutorial') autoFitTutBoxes();
}
window.addEventListener('resize', refitVisibleBoxesOnResize);
window.addEventListener('orientationchange', () => setTimeout(refitVisibleBoxesOnResize, 300));

// 여러 보표(오른손/왼손)를 하나의 절대 시간축으로 합쳐서 재생 — 예전엔 각 손을
// audio.playSequence()로 독립된 setTimeout 체인을 따로 돌렸는데, 이러면 (1) 두 손의
// 마디별 박자 합이 데이터상 미세하게라도 다르거나(나비야에서 실제로 발견된 버그),
// (2) 매 스텝 setTimeout(dur) 체인 자체가 콜백 지연을 누적시키는 것만으로도 시간이
// 지날수록 두 손이 서서히 어긋날 수 있었다. 전체 이벤트를 한 번에 정렬해서 절대
// 경과시간(performance.now() 기준) 대비 남은 지연만 매번 다시 계산하는 방식으로
// 바꿔서 이 어긋남 자체가 구조적으로 생길 수 없게 한다.
function playMergedSequence(partsNotes, tempo, onEnd) {
  const qSec = 60 / tempo;
  const events = [];
  partsNotes.forEach(notes => {
    let t = 0;
    notes.forEach(n => {
      if (n.pitch) events.push({ time: t, pitch: n.pitch, chordNotes: n.chordNotes, durSec: n.duration * qSec * 0.9 });
      t += n.duration * qSec;
    });
  });
  events.sort((a, b) => a.time - b.time);
  if (!events.length) { onEnd?.(); return () => {}; }

  let idx = 0, cancelled = false, timeoutId = null;
  const startWall = performance.now();
  function step() {
    if (cancelled) return;
    const elapsedSec = (performance.now() - startWall) / 1000;
    while (idx < events.length && events[idx].time <= elapsedSec + 0.01) {
      const e = events[idx];
      audio.playNote(e.pitch, e.durSec);
      e.chordNotes?.forEach(p => audio.playNote(p, e.durSec));
      idx++;
    }
    if (idx >= events.length) { onEnd?.(); return; }
    const delayMs = Math.max(0, (events[idx].time - elapsedSec) * 1000);
    timeoutId = setTimeout(step, delayMs);
  }
  step();

  return () => {
    cancelled = true;
    if (timeoutId) clearTimeout(timeoutId);
    audio.stopAll();
  };
}

function playExpScore() {
  const data = state.expScoreData;
  if (!data) return;
  audio.unlock();
  stopExpPlayback(); // 이전에 재생 중이던 게 있으면 먼저 멈추고 새로 시작 — 연타 시 중첩 방지
  const bpm = data.tempo || 90;
  const btn = document.getElementById('exp-play-btn');
  btn?.classList.add('playing');
  const done = () => btn?.classList.remove('playing');

  // 화면엔 첫 마디만 보여주지만, 재생은 곡 전체(모든 마디)를 들려준다.
  const parts = data.staves?.length >= 2
    ? [data.staves[0].notes, data.staves[1].notes]
    : [data.notes || []];
  state.expPlayCancel.push(playMergedSequence(parts, bpm, done));
}

// 인식 성공 시 바로 악보 화면으로 넘기지 않고, 촬영한 사진과 변환된 디지털 악보를
// 나란히 보여주는 미리보기(#exp-preview-overlay)를 한 번 거친다 — 사용자가 눈으로
// 비교해보고 이상하면 재촬영할 수 있게. 확인/재촬영 버튼은 initExpFlow()에서 한 번만 연결.
let pendingExpCapture = null; // { json, photoUrl }

function showExpCapturePreview(json, photoUrl) {
  pendingExpCapture = { json, photoUrl };
  document.getElementById('exp-preview-photo').src = photoUrl;
  // 촬영한 원본 사진과 직접 비교해야 하니, 우리 색상 커스텀 표기가 아니라 정식(디지털)
  // 오선보로 보여준다 — 커스텀 변환은 "이대로 사용" 확정 후 악보 화면에서 보여줌.
  // 사진에 찍힌 마디 전체와 비교해야 하므로 첫 마디만이 아니라 인식된 전체를 보여준다
  // (연습용 악보 화면은 여전히 한 마디씩만 — 이건 비교 전용 화면이라 다름).
  const staves = json.staves?.length >= 2
    ? [{ clef: 'treble', notes: json.staves[0].notes ?? [] }, { clef: 'bass', notes: json.staves[1].notes ?? [] }]
    : [{ clef: 'treble', notes: json.notes ?? [] }];
  renderDigitalStaff(document.getElementById('exp-preview-notation'), staves, json.timeSignature);
  document.getElementById('exp-preview-overlay')?.classList.remove('hidden');
}

function closeExpCapturePreview() {
  document.getElementById('exp-preview-overlay')?.classList.add('hidden');
  if (pendingExpCapture?.photoUrl) URL.revokeObjectURL(pendingExpCapture.photoUrl);
  pendingExpCapture = null;
}

// 체험하기용 촬영 결과 처리 — train/checkpoints의 r15 체크포인트로 서버(RunPod GPU)가
// 실제 인식한다. 인식 중에는 전용 대기 화면(#exp-recognizing-overlay, 임시)을 띄운다.
// 촬영해서 만든 악보는 정해진 3곡과 달리 순위표 대상이 아님(_noScore).
async function handleExpCameraCapture(file, fullFile) {
  if (!file) return;
  const overlay = document.getElementById('exp-recognizing-overlay');
  overlay?.classList.remove('hidden');
  try {
    // 콜드스타트 중엔 IN_QUEUE 상태로 몇 번이고 다시 물어보게 되는데, 그동안 태블릿
    // 화면이 그냥 멈춘 것처럼 보이지 않도록 대기 시간을 토스트로 계속 갱신해준다.
    const json = await recognizeImage(file, {
      model: 'custom',
      onProgress: p => {
        if (p.delayTimeMs != null) toast(`⏱ 대기 중... ${(p.delayTimeMs / 1000).toFixed(0)}s`);
      },
    });
    json._noScore = true;
    // 진단용 — 대기/실행 중 어디서 오래 걸리는지 태블릿에서도 바로 보이게 토스트로
    // 띄운다(devtools 접근 불가). 원인 확정되면 제거할 것.
    if (json._timing) {
      const t = json._timing;
      const s = ms => (ms == null ? '?' : (ms / 1000).toFixed(1));
      toast(`⏱ 대기 ${s(t.delayTimeMs)}s / 실행 ${s(t.executionTimeMs)}s`);
    }
    // 미리보기엔 인식용으로 잘라 보낸 이미지가 아니라 촬영한 프레임 전체(fullFile)를
    // 보여준다 — 없으면(구형 브라우저 등) 잘라낸 이미지로라도 대체.
    showExpCapturePreview(json, URL.createObjectURL(fullFile || file));
  } catch (e) {
    console.error('[handleExpCameraCapture] 인식 실패, 샘플로 대체:', e);
    toast(`⚠️ ${e.message} — 샘플로 보여드릴게요`);
    const demo = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
    showExpScore(sampleToNotation(demo, {
      id: generateId(), title: file.name.replace(/\.[^.]+$/, ''), createdAt: Date.now(), _noScore: true,
    }));
  } finally {
    overlay?.classList.add('hidden');
  }
}

// ── 전자 피아노 연결 테스트 ────────────────────────────────────────────────
// 단순히 "MIDI 기기가 잡히는가"만 보면 실제로 안 눌러도 "연결됨"으로 오판할 수 있어서,
// 커스텀 악보 첫 음에 해당하는 건반을 실제로 눌러보게 해서 신호가 오는지까지 확인한다.
// onDevices/onAnyNote — 태블릿(devtools 없음)에서도 어디서 막히는지 화면에서 바로 보이게
// 하기 위한 진단용 콜백. onDevices(names|null): null이면 requestMIDIAccess 자체가 실패
// (권한 거부 등), []면 기기가 하나도 안 잡힘, 그 외는 감지된 기기 이름 목록.
// onAnyNote(pitch): targetNote와 무관하게 "뭐라도" note-on이 들어올 때마다 호출 — 신호는
// 오는데 원하는 음이 아닌 경우(옥타브 오프셋 등)를 구분하는 데 씀.
function testMidiConnection(targetNote, { timeoutMs = 6000, onDevices, onAnyNote } = {}) {
  return new Promise(resolve => {
    if (!targetNote || !midi.isSupported()) { onDevices?.(null); resolve(false); return; }
    midi.requestAccess().then(() => {
      const inputs = midi.listInputs();
      onDevices?.(inputs.map(i => i.name));
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
        onNoteOn: pitch => { onAnyNote?.(pitch); if (pitch === targetNote) finish(true); },
        onNoteOff: () => {},
      });
    }).catch(() => { onDevices?.(null); resolve(false); });
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
  const bothBtn  = document.getElementById('exp-handmode-both');
  const rightBtn = document.getElementById('exp-handmode-right');
  const noteEl   = document.getElementById('exp-handmode-note');
  const onlyBoth = !!state.expScoreData?.onlyBothHands;
  bothBtn?.classList.toggle('exp-handmode-disabled', !(state.expScoreData?.staves?.length >= 2));
  rightBtn?.classList.toggle('exp-handmode-disabled', onlyBoth);
  if (noteEl) {
    noteEl.textContent = onlyBoth
      ? '※ 이 곡은 양손 연주만 선택할 수 있어요'
      : '※ 단일 오선 악보는 양손 모드를 선택할 수 없어요';
  }
  // 연주하기 버튼을 누른 순간부터(오른손/양손 선택 → 대기 → 연주) 12음 참고 돋보기 노출.
  setMagnifierVisible(true);
}

// ── 4/5: 전자피아노 연결 확인 + 닉네임 대기 화면 ───────────────────────────
function showExpWait(handMode) {
  state.expHandMode = handMode;
  hideExpScreens();
  document.getElementById('screen-exp-wait')?.classList.remove('hidden');
  runMidiCheck();
}

// 전자 피아노 연결 확인 — showExpWait() 진입 시와 "다시 확인" 버튼 클릭 시 공통으로 쓴다.
// 확인이 끝나기 전까지는 state.expMidiConfirmed가 아직 이전 값(또는 기본 false)이라 "시작"을
// 눌러도 실제 연결 여부와 다르게 진행될 수 있어서, 확인 중엔 시작 버튼을 막아둔다.
function runMidiCheck() {
  const statusEl = document.getElementById('exp-wait-midi-status');
  const debugEl  = document.getElementById('exp-wait-midi-debug');
  const retryBtn = document.getElementById('exp-wait-midi-retry');
  const startBtn = document.getElementById('exp-start-perform-btn');
  const data = state.expScoreData;
  const firstNote = firstMeasure(data?.staves?.[0]?.notes ?? data?.notes)?.[0]?.pitch;

  state.expMidiConfirmed = false;
  retryBtn?.classList.add('hidden');
  if (debugEl) debugEl.textContent = '';
  if (!firstNote) {
    statusEl.textContent = '📱 화면 건반으로 연주해요';
    return;
  }
  // Web MIDI는 https:// 또는 http://localhost가 아니면 브라우저가 API 자체를 막는다 —
  // 이 경우 케이블/기기는 멀쩡해도 절대 연결될 수 없으니, 신호를 기다리기 전에 먼저
  // 걸러서 정확한 원인을 알려준다(devtools 없는 태블릿에서 특히 헷갈리기 쉬움).
  if (!midi.isSupported()) {
    if (debugEl) debugEl.textContent = `⚠️ 이 주소(${location.protocol}//${location.host})에서는 MIDI가 막혀 있어요 — https:// 또는 http://localhost 로 접속해야 해요`;
    statusEl.textContent = '📱 화면 건반으로 연주해요 (전자 피아노 신호를 못 받았어요)';
    return;
  }
  statusEl.textContent = `🎹 전자 피아노에서 ${solfegeOf(firstNote)} 음을 눌러 연결을 확인해주세요`;
  if (startBtn) startBtn.disabled = true;
  testMidiConnection(firstNote, {
    // 태블릿엔 devtools가 없어서 콘솔 대신 이 줄로 어디서 막히는지 바로 보여준다.
    onDevices: names => {
      if (!debugEl) return;
      if (names === null) debugEl.textContent = '⚠️ MIDI 권한 요청 실패 — 브라우저가 접근을 막았어요(주소창 자물쇠 아이콘에서 MIDI 권한 확인)';
      else if (!names.length) debugEl.textContent = '🔌 감지된 MIDI 기기 없음 — 케이블/전원을 확인해주세요';
      else debugEl.textContent = `🔌 기기 감지됨: ${names.join(', ')} — 아무 건반이나 눌러보세요`;
    },
    onAnyNote: pitch => {
      if (debugEl) debugEl.textContent = `🔌 신호 수신됨 — 방금 누른 음: ${pitch}`;
    },
  }).then(ok => {
    state.expMidiConfirmed = ok;
    if (startBtn) startBtn.disabled = false;
    // 양손 모드는 전자 피아노 연결이 확인된 경우에만 실제로 진행된다(startExpPerform에서
    // 최종 결정) — 연결이 안 됐으면 시작 버튼을 누르기 전에 미리 안내해서 놀라지 않게.
    if (ok) {
      statusEl.textContent = '🎹 전자 피아노 연결을 확인했어요 — 화면 건반 없이 연주해요';
    } else {
      retryBtn?.classList.remove('hidden');
      statusEl.textContent = state.expHandMode === 'both'
        ? '📱 전자 피아노 연결이 확인되지 않아 오른손만 연주로 진행돼요'
        : '📱 화면 건반으로 연주해요 (전자 피아노 신호를 못 받았어요)';
    }
  });
}

// ── 5/5: 연주 진행(곡 전체, 마디마다 자연스럽게 다음 마디로) + 점수 ─────────
function startExpPerform(nickname) {
  const data = state.expScoreData;
  if (!data) return;
  // 전자 피아노 연결이 확인되지 않으면 양손 모드를 골랐더라도 오른손만으로 진행한다 —
  // 화면 터치 하나로는 양손 동시 입력을 안정적으로 받기 어려워, 실물 피아노가 확인된
  // 경우에만 실제로 양손 모드를 허용한다(showExpWait에서 이미 이 경우 안내 문구를 띄움).
  const handMode = state.expHandMode === 'both' && data.staves?.length >= 2 && state.expMidiConfirmed
    ? 'both' : 'right';

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
    held: new Set(), // 지금 물리적으로 눌려 있는(뗴지 않은) 음들 — 화음이 "동시에" 눌렸는지 판정용
    correct: 0, wrong: 0,
    hintOn: false, hintUsed: false, // 힌트(💡) 상태 — 한 번이라도 켜면 hintUsed는 끝까지 true(최종 점수 -10)
    pianoCtrl: null,
    refreshHighlight: null, // 힌트 버튼 클릭 시 최신 updateHighlight()를 부를 참조(아래에서 채움)
  };
  document.getElementById('exp-hint-btn')?.classList.remove('exp-hint-on');

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
        noteColorMode: 'octave',
      });
    } else {
      renderNotation(notationEl, t, { expectedIdx: p.tIdx, noteColorMode: 'octave' });
    }
    autoFitExpScore('exp-perform-notation');
  }
  // 화면 건반의 기준점(가온다 C4·한 옥타브 아래 도 C3) — 힌트와 무관하게 항상 표시.
  const REF_DOTS = [{ note: 'C4', color: '#E5383B' }, { note: 'C3', color: '#E5383B' }];
  // 힌트를 켰을 때만 "아직 안 누른" 음들을 건반에 표시 — 기본은 스스로 악보를 보고
  // 찾게 하고(파란 하이라이트 없음), 힌트 버튼(💡)을 켰을 때만 다음 음을 보여준다.
  function updateHighlight() {
    const p = state.expPerform;
    const { t, b } = currentMeasures();
    const tRemain = stepPitches(t, p.tIdx).filter(n => !p.tHit.has(n));
    const bRemain = p.handMode === 'both' ? stepPitches(b, p.bIdx).filter(n => !p.bHit.has(n)) : [];
    const hintDots = p.hintOn ? [
      ...tRemain.map(n => ({ note: n, color: '#0076CE' })), // 오른손 = 파랑
      ...bRemain.map(n => ({ note: n, color: '#FF8A3D' })), // 왼손 = 주황
    ] : [];
    p.pianoCtrl?.setDots([...REF_DOTS, ...hintDots]);
    p.pianoCtrl?.setExpected(p.hintOn ? (tRemain[0] ?? bRemain[0] ?? null) : null);
  }
  state.expPerform.refreshHighlight = updateHighlight;
  renderMeasure();

  const pianoWrap = document.getElementById('exp-perform-piano');
  pianoWrap.innerHTML = '';

  // 오른손/왼손이 동시에 눌려야 하는 화음도 "어느 손이 지금 이 음을 낼 차례인가"를
  // 헷갈리지 않게 판단 — 오른손(트레블) 스텝에 아직 안 채워진 음이면 오른손으로,
  // 아니면 왼손 스텝을 본다. 둘 다 아니면 오답.
  //
  // 화음(2음 이상)은 "동시에" 눌러야 정답 — 한 음만 누른 순간엔 정답/오답 판정을
  // 보류하고(p.held로 지금 물리적으로 눌려 있는 음들을 추적), 나머지 화음 구성음이
  // 전부 같이 눌려 있을 때만 한꺼번에 정답 처리한다. 하나만 누르고 떼면(동시가 아니면)
  // 그 음은 무효가 되어 다시 처음부터 같이 눌러야 한다.
  //
  // 양손 모드에서 두 보표의 현재 음이 같은 박자(beat)에서 시작하면, 실제 악보처럼 두
  // 손을 동시에 눌러야 하는 순간이므로 두 보표의 음을 하나의 화음처럼 합쳐서 동시
  // 입력을 요구한다(synced 분기). 박자가 다르면(한쪽이 더 긴 음을 들고 있는 등) 기존
  // 처럼 손마다 독립적으로 진행한다.
  function handleExpNotePress(pitch) {
    const p = state.expPerform;
    p.held.add(pitch);
    const { t, b } = currentMeasures();
    const tNote  = t[p.tIdx] ?? null;
    const bNote  = p.handMode === 'both' ? (b[p.bIdx] ?? null) : null;
    const synced = !!(tNote && bNote && tNote.beat === bNote.beat);
    const tStep  = stepPitches(t, p.tIdx);
    const bStep  = p.handMode === 'both' ? stepPitches(b, p.bIdx) : [];

    // 태블릿 진단용 — 실제로 어떤 음이 들어왔고 지금 채보에서 요구하는 음과 맞는지 텍스트로
    // 바로 보여준다(건반 플래시는 짧고 화면 밖으로 스크롤돼 있을 수 있어 놓치기 쉬움).
    const debugEl = document.getElementById('exp-perform-debug');
    if (debugEl) {
      const expected = [...tStep, ...bStep];
      const ok = expected.includes(pitch);
      debugEl.textContent = `🎹 누른 음: ${pitch} · 지금 요구되는 음: ${expected.join('/') || '-'} → ${ok ? '✅ 일치' : '❌ 불일치'}`;
    }

    if (synced) {
      const combined = [...tStep, ...bStep];
      const alreadyHit = n => p.tHit.has(n) || p.bHit.has(n);
      if (!combined.includes(pitch) || alreadyHit(pitch)) {
        p.wrong++;
        p.pianoCtrl?.flashWrong(pitch);
        return;
      }
      if (!combined.every(n => p.held.has(n) || alreadyHit(n))) return; // 양손 화음이 아직 같이 안 눌림 — 보류

      tStep.forEach(n => { if (!p.tHit.has(n)) { p.tHit.add(n); p.correct++; p.pianoCtrl?.flashCorrect(n); } });
      bStep.forEach(n => { if (!p.bHit.has(n)) { p.bHit.add(n); p.correct++; p.pianoCtrl?.flashCorrect(n); } });
      if (p.tHit.size >= tStep.length) { p.tIdx++; p.tHit.clear(); }
      if (p.bHit.size >= bStep.length) { p.bIdx++; p.bHit.clear(); }
    } else {
      let step = null, hitSet = null, advance = null;
      if (tStep.includes(pitch) && !p.tHit.has(pitch)) {
        step = tStep; hitSet = p.tHit; advance = () => { p.tIdx++; p.tHit.clear(); };
      } else if (bStep.includes(pitch) && !p.bHit.has(pitch)) {
        step = bStep; hitSet = p.bHit; advance = () => { p.bIdx++; p.bHit.clear(); };
      }

      if (!step) {
        p.wrong++;
        p.pianoCtrl?.flashWrong(pitch);
        return;
      }

      // MIDI 없이 화면 건반(마우스 커서 하나)으로 연주할 때는 화음 여러 음을 동시에
      // 누를 방법이 없다 — 화음이면 그중 가장 높은 음 하나만 맞으면 전체를 통과시킨다.
      // (양손 동기화 화음은 여기 안 옴 — 양손 모드는 MIDI 확인된 경우에만 진입하므로
      // 이 else 분기는 항상 한 손 화음만 다룸.) MIDI로 실제 피아노를 칠 땐 진짜 화음을
      // 눌러야 하므로 기존 동시입력 판정을 그대로 유지.
      if (!state.expMidiConfirmed && step.length > 1) {
        const highest = step.reduce((a, b) => (noteToFrequency(b) > noteToFrequency(a) ? b : a));
        if (pitch !== highest) return; // 최고음이 아니면 오답 처리 없이 그냥 대기
        step.forEach(n => { if (!hitSet.has(n)) { hitSet.add(n); p.correct++; } });
        p.pianoCtrl?.flashCorrect(pitch);
        advance();
      } else {
        if (!step.every(n => p.held.has(n) || hitSet.has(n))) return; // 화음 전체가 아직 같이 안 눌림 — 보류
        step.forEach(n => { if (!hitSet.has(n)) { hitSet.add(n); p.correct++; p.pianoCtrl?.flashCorrect(n); } });
        if (hitSet.size >= step.length) advance();
      }
    }

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

  function handleExpNoteRelease(pitch) {
    state.expPerform?.held.delete(pitch);
  }

  // 화면 건반은 MIDI 연결 여부와 상관없이 항상 만든다 — 실물 피아노로 칠 때도 지금 누른
  // 음이 화면에 그대로 반영돼야(정답/오답 플래시 포함) "같은 음이 눌렸는지" 눈으로 확인할
  // 수 있다(전엔 MIDI 확인 시 건반을 아예 안 그려서 pianoCtrl이 null이라 어떤 시각 효과도
  // 안 나갔음).
  const wrap = document.createElement('div');
  wrap.className = 'piano-wrapper mini-piano';
  const pianoEl = document.createElement('div');
  pianoEl.className = 'piano';
  wrap.appendChild(pianoEl);
  pianoWrap.appendChild(wrap);
  state.expPerform.pianoCtrl = buildPiano(pianoEl, wrap, {
    showLabels: true, onPress: handleExpNotePress, onRelease: handleExpNoteRelease,
    pannable: true, centerOnNotes: ['C3', 'C4'], // 기본 뷰: 가온다·한옥타브 아래 도가 중앙에 오게, 드래그/스와이프로 좌우 이동 가능
  });

  if (state.expMidiConfirmed) {
    const inputs = midi.listInputs();
    // 실물 건반 입력도 화면 건반과 같은 press/release 경로를 태워서 시각 효과(눌림 표시,
    // 정답/오답 플래시)가 그대로 적용되게 한다 — silent:true라 화면 건반 자체의 신시사이저
    // 소리는 안 나고(실물 피아노가 이미 냄) 시각 피드백만 탄다.
    if (inputs.length) {
      midi.setInput(inputs[0].id, {
        onNoteOn:  note => state.expPerform.pianoCtrl?.press(note, { silent: true }),
        onNoteOff: note => state.expPerform.pianoCtrl?.release(note),
      });
    }
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
  let score = total > 0 ? Math.round((p.correct / total) * p.maxScore * 10) / 10 : 0;
  // 힌트(💡)를 한 번이라도 켰으면 최종 점수에서 10점 감점(0점 미만으로는 안 내려감).
  if (p.hintUsed) score = Math.max(0, Math.round((score - 10) * 10) / 10);
  document.getElementById('exp-perform-score-text').textContent = p.hintUsed
    ? `${p.nickname}님, ${score}점이에요! (${p.maxScore}점 만점, 힌트 사용 -10점)`
    : `${p.nickname}님, ${score}점이에요! (${p.maxScore}점 만점)`;
  document.getElementById('exp-perform-result').classList.remove('hidden');
  // 실패해도(네트워크 등) 연주 결과 화면 자체는 이미 떴으니 조용히 무시 — 순위표 등록
  // 실패가 사용자 체험을 막으면 안 됨. 원인 파악용으로 콘솔에는 남긴다.
  saveLeaderboardEntryCloud(state.expScoreData?.title, p.nickname, score, p.maxScore)
    .catch(e => console.error('[saveLeaderboardEntryCloud] 저장 실패:', e));
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
  // 촬영 버튼도 예시 곡과 같은 칸 모양으로 — 우측 하단 별도 FAB 대신 그리드 안에 나란히.
  // id는 기존 그대로(exp-camera-btn) 유지해서 setupCameraCapture()의 openBtn 참조가 그대로 작동.
  const camCard = document.createElement('button');
  camCard.className = 'exp-sample-card exp-sample-card-camera';
  camCard.id = 'exp-camera-btn';
  camCard.title = '촬영하기';
  camCard.innerHTML = `<span class="exp-sample-emoji">📷</span><span class="exp-sample-title">직접 촬영</span>`;
  grid.appendChild(camCard);

  setupCameraCapture({
    openBtn: 'exp-camera-btn', cancelBtn: 'exp-camera-cancel', shutterBtn: 'exp-camera-shutter',
    captureBox: 'exp-camera-capture', video: 'exp-camera-video', canvas: 'exp-camera-canvas',
    guideCanvas: 'exp-camera-guide-canvas', guideHint: 'exp-camera-guide-hint', error: 'exp-camera-error',
    guideWFrac: 0.99, // 뷰파인더가 화면 전체라 오선 가이드도 프레임 폭 거의 끝까지
    galleryBtn: 'exp-camera-gallery', galleryInput: 'exp-gallery-input',
  }, handleExpCameraCapture);

  document.getElementById('exp-preview-confirm')?.addEventListener('click', () => {
    const p = pendingExpCapture;
    if (!p) return;
    closeExpCapturePreview();
    showExpScore(p.json);
  });
  document.getElementById('exp-preview-retake')?.addEventListener('click', () => {
    closeExpCapturePreview();
    document.getElementById('exp-camera-btn')?.click(); // 카메라 다시 열기
  });

  document.getElementById('exp-handmode-right')?.addEventListener('click', () => {
    if (!state.expScoreData?.onlyBothHands) showExpWait('right');
  });
  document.getElementById('exp-handmode-both')?.addEventListener('click', () => {
    if (state.expScoreData?.staves?.length >= 2) showExpWait('both');
  });

  document.getElementById('exp-play-btn')?.addEventListener('click', playExpScore);
  document.getElementById('exp-perform-btn')?.addEventListener('click', showExpHandMode);

  // 힌트(💡) 토글 — startExpPerform()이 매번 새로 만드는 화면 건반과 달리 이 버튼은
  // 고정 DOM이라 리스너를 한 번만 등록하고, 그때그때 최신 상태(state.expPerform)를
  // 읽고 refreshHighlight로 다시 그리게 한다(리스너 중복 등록 방지).
  document.getElementById('exp-hint-btn')?.addEventListener('click', () => {
    const p = state.expPerform;
    if (!p) return;
    p.hintOn = !p.hintOn;
    if (p.hintOn) p.hintUsed = true;
    document.getElementById('exp-hint-btn')?.classList.toggle('exp-hint-on', p.hintOn);
    p.refreshHighlight?.();
  });

  document.getElementById('exp-start-perform-btn')?.addEventListener('click', async () => {
    const nickname = document.getElementById('exp-nickname-input').value.trim() || '익명';
    const songKey = state.expScoreData?.title;
    const errEl = document.getElementById('exp-nickname-error');
    errEl?.classList.add('hidden');

    // '익명'(닉네임 미입력 시 기본값)은 여러 명이 동시에 쓸 수 있어야 하므로 중복 체크
    // 대상에서 뺀다 -- 곡 안에서 유일해야 하는 건 사용자가 실제로 입력한 닉네임만.
    if (nickname !== '익명') {
      const btn = document.getElementById('exp-start-perform-btn');
      btn.disabled = true;
      let taken = false;
      try {
        taken = await isNicknameTakenInSong(songKey, nickname);
      } catch (e) {
        // 확인 실패(네트워크 등) 시엔 막지 않고 통과시킴 -- 체험이 우선. 원인 파악용으로
        // 콘솔에는 남긴다(Firestore 보안규칙 미설정이 원인인 경우가 많음).
        console.error('[isNicknameTakenInSong] 확인 실패:', e);
      }
      btn.disabled = false;
      if (taken) {
        if (errEl) {
          errEl.textContent = '이미 사용 중인 닉네임이에요 — 다른 닉네임을 입력해주세요';
          errEl.classList.remove('hidden');
        }
        return;
      }
    }
    startExpPerform(nickname);
  });
  document.getElementById('exp-perform-done-btn')?.addEventListener('click', () => {
    setMagnifierVisible(false);
    showExpScore(state.expScoreData); // 순위표 갱신 포함해서 악보 화면으로 복귀
  });

  // 지금 보이는 체험하기 화면(5개 중 하나)을 찾아서 서서히 사라지게 한 뒤 랜딩으로 —
  // 튜토리얼의 fadeExitToLanding()과 같은 페이드 효과. exitFlowToLanding()이 아니라
  // hideExpScreens()를 써야 재생 중인 곡 정지/촬영 미리보기 정리 등 체험하기 전용
  // 뒷정리가 같이 되므로 fadeExitToLanding을 그대로 재사용하지 않고 따로 만든다.
  const goHome = () => {
    setMagnifierVisible(false);
    const visible = ['screen-exp-select', 'screen-exp-score', 'screen-exp-handmode', 'screen-exp-wait', 'screen-exp-perform']
      .find(id => !document.getElementById(id)?.classList.contains('hidden'));
    const el = visible && document.getElementById(visible);
    if (!el) { hideExpScreens(); showLanding(); return; }
    // 랜딩을 먼저 보여준 뒤(.exp-screen이 z-index로 여전히 위에 덮고 있어 화면상 변화는
    // 없음) 지금 화면을 페이드아웃 — 그래야 옅어지는 동안 뒤로 비치는 게 랜딩이지,
    // 뒤에 깔려 있던 다른 화면(튜토리얼 등)이 잠깐 비치는 일이 없다.
    showLanding();
    el.style.transition = `opacity ${FADE_EXIT_MS}ms ease`;
    el.style.opacity = '0';
    setTimeout(() => {
      hideExpScreens();
      el.style.transition = '';
      el.style.opacity = '';
    }, FADE_EXIT_MS);
  };
  document.getElementById('exp-select-home')?.addEventListener('click', goHome);
  document.getElementById('exp-score-home')?.addEventListener('click', goHome);
  document.getElementById('exp-handmode-home')?.addEventListener('click', goHome);
  document.getElementById('exp-wait-home')?.addEventListener('click', goHome);
  document.getElementById('exp-perform-home')?.addEventListener('click', goHome);
  document.getElementById('exp-wait-midi-retry')?.addEventListener('click', runMidiCheck);
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

  const hide = () => popup.classList.remove('visible');

  // 마우스를 올릴 때(hover)가 아니라 눌렀을 때만 뜨고, 다시 누르면 사라지는 토글로 동작.
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
// 공용으로 쓴다. 4옥(오른손 "낮음", C4~B4)만 회색 — 대보표에서 오른손·왼손이 만나는
// 겹치는 구간(가온다 근처, 양쪽 다 덧줄로 넘나들 수 있는 영역)이라 따로 표시. 왼손의
// 진짜 자기 영역은 3옥(높음, C3~B3)=연한 주황부터 시작해서, 2옥(중간)부터는 그 아래
// (낮음·최저 포함) 전부 진한 주황 한 가지로 묶는다.
const HAND_ZONES = [
  { hand: '왼손',  zone: '최저', from: 'A0', to: 'B0', hex: '#E8590C', note: 'A0' },
  { hand: '왼손',  zone: '낮음', from: 'C1', to: 'B1', hex: '#E8590C', note: 'C1' },
  { hand: '왼손',  zone: '중간', from: 'C2', to: 'B2', hex: '#E8590C', note: 'G2' },
  { hand: '왼손',  zone: '높음', from: 'C3', to: 'B3', hex: '#FFC98A', note: 'C3' },
  { hand: '오른손', zone: '낮음', from: 'C4', to: 'B4', hex: '#999999', note: 'C4' },
  { hand: '오른손', zone: '중간', from: 'C5', to: 'B5', hex: '#3A9EE0', note: 'G5' },
  { hand: '오른손', zone: '높음', from: 'C6', to: 'C8', hex: '#0076CE', note: 'C6' },
];

// 커스텀 악보(모든 화면 공통) 존 배경색. 트레블은 HAND_ZONES와 그대로 맞물림(z0=높음
// →z2=낮음). 베이스는 pitchToZone()의 새 경계(z0=4옥 이상/z1=3옥/z2=2옥 이하)에 맞춰
// [회색, 연한 주황, 진한 주황]을 직접 지정 — HAND_ZONES는 건반 위 8개 옥타브 밴드용이라
// (최저/낮음/중간/높음 4단계) 악보의 3단계(z0~z2)와 이름이 1:1로 안 맞는다.
function zoneColorsForHand(hand) {
  return ['높음', '중간', '낮음'].map(zone => HAND_ZONES.find(z => z.hand === hand && z.zone === zone).hex);
}
const TREBLE_ZONE_COLORS = zoneColorsForHand('오른손');
const BASS_ZONE_COLORS   = ['#999999', '#FFC98A', '#E8590C']; // z0=4옥+(회색) z1=3옥(연한주황) z2=2옥 이하(진한주황)

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
    caption: '복잡한 규칙은 버리고, 직관만 남기다',
    subcaption: '조표와 옥타브 연산 없이, 색과 위치만으로 누구나 읽을 수 있는 커스텀 악보를 경험해 보세요.',
    splitDirection: 'row', // 위/아래 대신 좌/우로 두 악보를 나란히 비교
    render(top, bottom) {
      top.innerHTML = `
        <p class="tut-compare-label">전통 오선 악보</p>
        <img class="tut-compare-img" src="assets/tut-intro-original.png" alt="전통 오선 악보 원본">`;
      bottom.innerHTML = `
        <p class="tut-compare-label">커스텀 악보 — 옥타브별 존 색상 참고표</p>
        <img class="tut-compare-img" src="assets/tut-intro-custom.png" alt="커스텀 악보 옥타브 존 참고표 — 높은음자리(오른손)/낮은음자리(왼손) 각 존과 음이름">`;
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
      // 가온다(C4, 오른손 기준점)에 더해 한 옥타브 아래 도(C3, 왼손 기준점)에도 빨간 점.
      ctrl.setDots([
        { note: 'C3', color: '#FF4444' },
        { note: 'C4', color: '#FF4444' },
      ]);
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
  document.getElementById('tut-subcaption').textContent = page.subcaption || '';
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

// 버튼 누를 때마다 마치 책장을 넘기듯, 지금 페이지가 세로축(rotateY)으로 살짝 들리며
// 반대쪽으로 넘어가 사라지고 → 내용 교체 → 다음 페이지가 반대편에서 같은 방식으로
// 넘어와 자리잡는다. dir(1=다음→왼쪽으로 넘어감, -1=이전→오른쪽으로 넘어감)에 따라
// 회전축(transform-origin)과 방향을 반대로 잡는다. perspective는 부모(.screen-content)에서.
const TUT_ANIM_MS = 420;
function goToTutPage(idx) {
  if (idx < 0 || idx >= TUT_PAGES.length || idx === tutPageIdx) return;
  const dir = idx > tutPageIdx ? 1 : -1; // 1=다음, -1=이전
  const frame = document.getElementById('tut-page-frame');
  const EASE = `cubic-bezier(0.45, 0, 0.2, 1)`;

  // 나가는 페이지: dir=1(다음)이면 오른쪽 모서리를 축으로 왼쪽으로 넘어가며 사라짐,
  // dir=-1(이전)이면 왼쪽 모서리를 축으로 오른쪽으로 넘어가며 사라짐.
  frame.style.transformOrigin = dir === 1 ? 'right center' : 'left center';
  frame.style.transition = `transform ${TUT_ANIM_MS}ms ${EASE}, opacity ${TUT_ANIM_MS}ms ease, box-shadow ${TUT_ANIM_MS}ms ease`;
  frame.style.transform  = `rotateY(${dir * -16}deg) translateX(${dir * -5}%)`;
  frame.style.opacity    = '0.25';
  frame.style.boxShadow  = `${dir * -36}px 0 50px -12px rgba(0,0,0,0.55)`;

  setTimeout(() => {
    renderTutPage(idx);
    // 들어오는 페이지: 반대쪽 모서리를 축으로 살짝 들린 채 시작해서 제자리로 접힌다.
    frame.style.transition = 'none';
    frame.style.transformOrigin = dir === 1 ? 'left center' : 'right center';
    frame.style.transform  = `rotateY(${dir * 16}deg) translateX(${dir * 5}%)`;
    frame.style.opacity    = '0.25';
    frame.style.boxShadow  = `${dir * 36}px 0 50px -12px rgba(0,0,0,0.55)`;
    void frame.offsetWidth; // 강제 리플로우 — 위 스타일을 실제로 적용시킨 뒤 아래 트랜지션 시작
    frame.style.transition = `transform ${TUT_ANIM_MS}ms ${EASE}, opacity ${TUT_ANIM_MS}ms ease, box-shadow ${TUT_ANIM_MS}ms ease`;
    frame.style.transform  = 'rotateY(0deg) translateX(0)';
    frame.style.opacity    = '1';
    frame.style.boxShadow  = 'none';
  }, TUT_ANIM_MS);
}

// 랜딩(메인 화면)으로 나갈 때 지금 화면이 서서히 사라지면서 전환 — 뚝 끊기지 않게.
const FADE_EXIT_MS = 320;
function fadeExitToLanding(screenId) {
  const el = document.getElementById(screenId);
  if (!el) { exitFlowToLanding(); return; }
  el.style.transition = `opacity ${FADE_EXIT_MS}ms ease`;
  el.style.opacity = '0';
  setTimeout(() => {
    exitFlowToLanding();
    el.style.transition = '';
    el.style.opacity = '';
  }, FADE_EXIT_MS);
}

function initTutorial() {
  // tut-home-btn은 이제 body 최상위에 있어(위 index.html 주석 참고) #screen-tutorial의
  // active 여부와 더 이상 부모-자식 관계로 자동 연동되지 않는다 -- MutationObserver로
  // #screen-tutorial의 class 속성을 감시해서 active일 때만 보이게 동기화(어느 경로로
  // 화면이 바뀌든 -- navigate()든 fadeExitToLanding()든 -- 다 잡아냄, 호출부마다 따로
  // 안 챙겨도 됨).
  const tutHomeBtn = document.getElementById('tut-home-btn');
  const tutSection = document.getElementById('screen-tutorial');
  if (tutHomeBtn && tutSection) {
    const syncTutHomeBtn = () => tutHomeBtn.classList.toggle('hidden', !tutSection.classList.contains('active'));
    new MutationObserver(syncTutHomeBtn).observe(tutSection, { attributes: true, attributeFilter: ['class'] });
    syncTutHomeBtn();
  }
  tutHomeBtn?.addEventListener('click', () => fadeExitToLanding('screen-tutorial'));
  document.getElementById('tut-prev').addEventListener('click', () => {
    if (tutPageIdx > 0) { goToTutPage(tutPageIdx - 1); return; }
    if (state.flowFromLanding) fadeExitToLanding('screen-tutorial'); // 첫 페이지의 "이전" = 랜딩으로 나가기
  });
  document.getElementById('tut-next').addEventListener('click', () => {
    if (tutPageIdx < TUT_PAGES.length - 1) { goToTutPage(tutPageIdx + 1); return; }
    // 마지막 페이지: 튜토리얼 단독 흐름이면 랜딩으로, 아니면(자유 탐색/구 킨스크) 변환 화면으로
    if (state.flowLock && !state.flowLock.includes('convert')) fadeExitToLanding('screen-tutorial');
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

  // 이어 붙이기 모드 토글 — "새로 저장"이면 제목 입력창, "이어 붙이기"면 기존 악보
  // 드롭다운을 보여준다(모델이 한 줄씩만 인식해서 여러 줄 악보는 나눠 찍어 합쳐야 함).
  document.querySelectorAll('input[name="save-mode"]').forEach(r => {
    r.addEventListener('change', () => {
      const append = document.querySelector('input[name="save-mode"]:checked')?.value === 'append';
      document.getElementById('save-title-input')?.classList.toggle('hidden', append);
      document.getElementById('save-append-target')?.classList.toggle('hidden', !append);
      if (append) refreshAppendTargetOptions();
    });
  });

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
const CAMERA_GUIDE_W_FRAC = 0.88; // 기본값(기존 변환 화면) — 체험하기는 setupCameraCapture(ids)의 guideWFrac로 더 크게 오버라이드
function cameraGuideHFrac(grandStaff) { return grandStaff ? 0.34 : 0.16; }

function cameraGuideRectNative(vw, vh, grandStaff, wFrac = CAMERA_GUIDE_W_FRAC) {
  const w = vw * wFrac;
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

function drawCameraGuideOverlay(canvas, wrap, video, grandStaff, wFrac) {
  const ww = wrap.clientWidth, wh = wrap.clientHeight;
  if (!ww || !wh) return;
  canvas.width = ww; canvas.height = wh;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, ww, wh);

  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  const rect = cameraNativeToDisplay(cameraGuideRectNative(vw, vh, grandStaff, wFrac), vw, vh, ww, wh);

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
  // 갤러리 선택은 옵션 — ids에 galleryBtn/galleryInput을 넘긴 호출부에서만 활성화된다
  // (지금은 체험하기 전용, 스크린샷/기존에 찍어둔 악보 사진을 그대로 인식시킬 때 씀).
  const galleryBtn   = ids.galleryBtn   && document.getElementById(ids.galleryBtn);
  const galleryInput = ids.galleryInput && document.getElementById(ids.galleryInput);
  if (!openBtn || !captureBox) return;
  const guideWrap = captureBox.querySelector('.camera-video-wrap');
  const modeChips = captureBox.querySelectorAll('.camera-mode-chip');

  let grandStaff = true; // flutter 쪽 기본값과 동일 — 피아노 악보는 대보표가 더 흔함
  const guideWFrac = ids.guideWFrac ?? CAMERA_GUIDE_W_FRAC; // 체험하기는 더 큰 값으로 오버라이드

  function updateHint() {
    guideHint.textContent = grandStaff
      ? '대보표(높은음자리+낮은음자리)를 박스 안에 맞춰주세요'
      : '오선 하나를 박스 안에 맞춰주세요';
  }
  function redrawGuide() { drawCameraGuideOverlay(guideCanvas, guideWrap, video, grandStaff, guideWFrac); }

  modeChips.forEach(chip => {
    chip.addEventListener('click', () => {
      grandStaff = chip.dataset.grand === '1';
      modeChips.forEach(c => c.classList.toggle('active', c === chip));
      updateHint();
      redrawGuide();
    });
  });

  video.addEventListener('loadedmetadata', redrawGuide);
  // 화면 레이아웃 리사이즈(ResizeObserver)와 카메라 스트림 자체의 해상도 변경은 서로
  // 다른 타이밍에 일어난다 — 특히 폰을 돌리면 카메라 센서가 videoWidth/videoHeight를
  // 새로 보고하는 게 CSS 레이아웃 리사이즈보다 늦거나 빠를 수 있어서, 화면만 보고
  // 다시 그리면 옛 해상도 기준으로 그려진 오선이 남는 버그가 있었다. video 엘리먼트의
  // 네이티브 resize 이벤트(실제 스트림 해상도 변경 시점)도 별도로 들어서 항상 최신
  // videoWidth/videoHeight로 다시 계산하게 한다.
  video.addEventListener('resize', redrawGuide);
  window.addEventListener('orientationchange', () => setTimeout(redrawGuide, 300));
  new ResizeObserver(redrawGuide).observe(guideWrap);

  function cleanup() {
    state.cameraStream?.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
    video.srcObject = null;
    captureBox.classList.add('hidden');
    state.activeCameraStop = null;
    // 촬영 중에만 세로 모드를 허용해줬던 것도 원상복구(.rotate-prompt 참고).
    document.body.classList.remove('camera-active');
  }

  async function openCamera() {
    errorEl.classList.add('hidden');
    captureBox.classList.remove('hidden');
    state.activeCameraStop = cleanup;
    updateHint();
    // 악보 사진은 세로로 들고 찍는 게 자연스러워서, 촬영 화면이 떠 있는 동안만
    // 가로 고정(.rotate-prompt)을 풀어 세로로도 그대로 찍을 수 있게 한다.
    document.body.classList.add('camera-active');

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

  // 갤러리에서 사진 선택 — 스크린샷했거나 예전에 찍어둔 악보 이미지를 그대로 인식시킨다.
  // 라이브 카메라 가이드로 잘라내는 크롭 단계가 필요 없어(이미 완성된 이미지라) 원본
  // 파일을 그대로 onCaptured에 넘긴다(크롭/풀프레임 둘 다 같은 파일).
  if (galleryBtn && galleryInput) {
    galleryBtn.addEventListener('click', () => galleryInput.click());
    galleryInput.addEventListener('change', e => {
      const file = e.target.files[0];
      galleryInput.value = ''; // 같은 파일을 다시 골라도 change가 또 발생하게
      if (!file) return;
      if (!file.type.startsWith('image/')) { toast('이미지 파일을 선택해주세요'); return; }
      if (file.size > 10 * 1024 * 1024) { toast('파일 크기가 10MB를 초과합니다'); return; }
      state.activeCameraStop?.(); // 라이브 카메라가 켜져 있었으면 정리
      onCaptured(file, file);
    });
  }

  // Vercel 서버리스 함수는 요청 본문 4.5MB 하드 제한이 있어서(설정으로 못 늘림) — 카메라
  // 화면/가이드를 크게 키운 뒤로 원본 해상도 그대로 올리면 넘기기 쉽다. 캡처 시점에
  // 긴 변 기준 CAPTURE_MAX_DIM으로 미리 축소해서 올린다(OMR 인식엔 이 정도 해상도로 충분).
  const CAPTURE_MAX_DIM = 1600;
  shutterBtn.addEventListener('click', () => {
    if (!state.cameraStream || !video.videoWidth) return;
    const rect = cameraGuideRectNative(video.videoWidth, video.videoHeight, grandStaff, guideWFrac);
    const scale = Math.min(1, CAPTURE_MAX_DIM / Math.max(rect.w, rect.h));
    canvas.width  = Math.round(rect.w * scale);
    canvas.height = Math.round(rect.h * scale);
    canvas.getContext('2d').drawImage(video, rect.x, rect.y, rect.w, rect.h, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(cropBlob => {
      if (!cropBlob) return;
      // 인식 서버에는 지금처럼 가이드 영역만 잘라서 보내되(용량 제한/인식 정확도 유지),
      // 미리보기 화면의 "찍은 사진"은 사용자가 실제로 본 프레임 전체를 보여줘야 무엇을
      // 찍었는지 비교하기 쉽다 — 풀프레임을 별도로 한 번 더 캡처해서 같이 넘긴다.
      const fullScale = Math.min(1, CAPTURE_MAX_DIM / Math.max(video.videoWidth, video.videoHeight));
      const fullCanvas = document.createElement('canvas');
      fullCanvas.width  = Math.round(video.videoWidth * fullScale);
      fullCanvas.height = Math.round(video.videoHeight * fullScale);
      fullCanvas.getContext('2d').drawImage(video, 0, 0, fullCanvas.width, fullCanvas.height);
      fullCanvas.toBlob(fullBlob => {
        state.activeCameraStop?.();
        const cropFile = new File([cropBlob], 'capture.jpg', { type: 'image/jpeg' });
        const fullFile = fullBlob ? new File([fullBlob], 'capture-full.jpg', { type: 'image/jpeg' }) : null;
        onCaptured(cropFile, fullFile);
      }, 'image/jpeg', 0.85);
    }, 'image/jpeg', 0.85);
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
  const base = { title: s.title, tempo: s.tempo, timeSignature: s.timeSignature, onlyBothHands: s.onlyBothHands };
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

  const model = document.getElementById('model-select')?.value || 'custom';

  try {
    const json = await recognizeImage(file, { model });
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

  // 새 인식 결과가 나올 때마다 저장 모드를 "새로 저장"으로 되돌림 -- 이어 붙이기는
  // 매번 대상을 직접 고르게 해서(자동 지속 X) 실수로 엉뚱한 악보에 붙는 걸 방지.
  const newModeRadio = document.querySelector('input[name="save-mode"][value="new"]');
  if (newModeRadio) { newModeRadio.checked = true; newModeRadio.dispatchEvent(new Event('change')); }

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

// 저장된 악보 목록으로 "이어 붙이기" 드롭다운을 채운다(모델이 한 줄씩만 인식해서
// 여러 줄짜리 실제 악보는 줄마다 따로 찍어 이어 붙여야 하므로, 2026-08-10 추가).
// title에 특수문자가 있어도 안전하게 옵션을 만들려고 innerHTML 문자열 대신
// createElement + textContent를 씀.
function refreshAppendTargetOptions() {
  const sel = document.getElementById('save-append-target');
  if (!sel) return;
  const all = loadAll();
  sel.innerHTML = '';
  if (!all.length) {
    const opt = document.createElement('option');
    opt.disabled = true; opt.selected = true; opt.textContent = '저장된 악보가 없어요';
    sel.appendChild(opt);
    return;
  }
  all.forEach(n => {
    const opt = document.createElement('option');
    opt.value = n.id; opt.textContent = n.title;
    sel.appendChild(opt);
  });
}

function saveCurrentResult() {
  if (!state.convertResult) return;
  const appendMode = document.querySelector('input[name="save-mode"]:checked')?.value === 'append';

  // Grand staff: flatten to treble notes for playback, keep staves for display
  const captured = { ...state.convertResult };
  if (!captured.notes && captured.staves) {
    captured.notes = captured.staves[0]?.notes ?? [];
  }

  if (appendMode) {
    const targetId = document.getElementById('save-append-target')?.value;
    const target = targetId ? getById(targetId) : null;
    if (!target) { toast('이어 붙일 악보를 선택해주세요'); return; }
    // 다음 줄 인식 결과를 기존 악보 뒤에 그대로 이어 붙인다 -- clef/박자표가 달라도
    // 검사하지 않음(2026-08-10 확정, 같은 곡의 연속된 줄이라는 사용자 의도를 믿고
    // 단순하게 감).
    target.notes = [...(target.notes ?? []), ...(captured.notes ?? [])];
    if (target.staves && captured.staves) {
      target.staves = target.staves.map((s, i) => ({
        ...s, notes: [...(s.notes ?? []), ...(captured.staves[i]?.notes ?? [])],
      }));
    }
    target.updatedAt = Date.now();
    saveNotation(target);
    if (state.user) saveScoreCloud(target).catch(console.error);
    toast(`💾 "${target.title}"에 이어 붙였어요!`);
    return;
  }

  const t = document.getElementById('save-title-input').value.trim();
  if (t) state.convertResult.title = t;
  captured.title = state.convertResult.title;
  saveNotation(captured);
  if (state.user) saveScoreCloud(captured).catch(console.error);
  toast(`💾 "${captured.title}" 저장 완료!`);
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
    if (dot)   { dot.classList.add('connected'); dot.title = '서버 연결됨'; }
    if (badge) badge.textContent = '✅ OMR 서버 연결됨';
    if (msg)   msg.textContent = `커스텀 모델: ${s.custom ? '✓' : '✗'}`;
    // 연결된 모델만 select에 표시
    const sel = document.getElementById('model-select');
    if (sel) {
      [...sel.options].forEach(opt => { opt.disabled = !s[opt.value]; });
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
  audio.preload(); // 그랜드피아노 실샘플 다운로드를 최대한 일찍 시작(사용자 제스처 불필요)

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
  initWalkinBack();

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
