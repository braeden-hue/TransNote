/**
 * firebase.js — Firebase Auth + Firestore 연동
 *
 * ⚠️  사용 전 아래 firebaseConfig를 반드시 교체하세요:
 *   1. https://console.firebase.google.com 접속
 *   2. 프로젝트 만들기 → 앱 추가 → 웹(</>)
 *   3. "Firebase SDK 추가" 단계에서 아래 config 객체 복사
 *   4. Authentication → 시작하기 → Google 로그인 방법 사용 설정
 *   5. Firestore Database → 데이터베이스 만들기 → 테스트 모드 시작
 */

import { initializeApp }
  from 'https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js';
import { getAuth, GoogleAuthProvider, signInWithPopup,
         signOut as fbSignOut, onAuthStateChanged }
  from 'https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js';
import { getFirestore, collection, doc, setDoc, getDocs,
         deleteDoc, addDoc, query, where }
  from 'https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js';

// ── 여기를 Firebase Console에서 복사한 값으로 교체 ──────────────────────────
const firebaseConfig = {
  apiKey: "AIzaSyBrUG9xxCGAf3Qr_FPfL1KKuit1JwsR6AE",
  authDomain: "musicakbo-fac10.firebaseapp.com",
  projectId: "musicakbo-fac10",
  storageBucket: "musicakbo-fac10.firebasestorage.app",
  messagingSenderId: "758449025722",
  appId: "1:758449025722:web:5c0b7972d4eae03f5c8883",
  measurementId: "G-VB9MY0QL30"
};
// ─────────────────────────────────────────────────────────────────────────────

const IS_CONFIGURED = firebaseConfig.apiKey !== 'YOUR_API_KEY';

let _auth = null;
let _db   = null;

if (IS_CONFIGURED) {
  const app = initializeApp(firebaseConfig);
  _auth = getAuth(app);
  _db   = getFirestore(app);
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export function signInWithGoogle() {
  if (!_auth) return Promise.reject(new Error('Firebase 설정 필요'));
  return signInWithPopup(_auth, new GoogleAuthProvider());
}

export function signOutUser() {
  if (!_auth) return Promise.resolve();
  return fbSignOut(_auth);
}

/** callback(user | null) — 반환값을 호출하면 구독 해제 */
export function onAuthChange(callback) {
  if (!_auth) { callback(null); return () => {}; }
  return onAuthStateChanged(_auth, callback);
}

// ── Firestore CRUD ────────────────────────────────────────────────────────────

/** 악보 1개를 클라우드에 저장 (upsert) */
export async function saveScoreCloud(score) {
  if (!_db || !_auth?.currentUser) return;
  await setDoc(doc(_db, 'scores', score.id), {
    ...score,
    userId:    _auth.currentUser.uid,
    updatedAt: Date.now(),
  });
}

/** 현재 로그인 유저의 악보 목록 불러오기 */
export async function loadScoresCloud() {
  if (!_db || !_auth?.currentUser) return [];
  const q    = query(collection(_db, 'scores'), where('userId', '==', _auth.currentUser.uid));
  const snap = await getDocs(q);
  const list = snap.docs.map(d => d.data());
  list.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
  return list;
}

/** 악보 1개를 클라우드에서 삭제 */
export async function deleteScoreCloud(id) {
  if (!_db || !_auth?.currentUser) return;
  await deleteDoc(doc(_db, 'scores', id));
}

// ── 순위표(리더보드) ──────────────────────────────────────────────────────────
// 부스 회전율 때문에 로그인 없이 닉네임만으로 참여(docs/PLAN_booth_companion_page.md
// "키오스크 ↔ 리더보드 연동 메커니즘" 참고) — 위 scores 컬렉션(개인 저장, 로그인 필요)과
// 무관한 별도 컬렉션. 닉네임 유일성은 songKey(곡) 안에서만 검사 — 다른 곡에서는 같은
// 닉네임을 다시 써도 됨(2026-08-10 확정).

/** songKey 안에서 nickname이 이미 쓰였는지 확인. 로그인 불필요(공개 컬렉션). */
export async function isNicknameTakenInSong(songKey, nickname) {
  if (!_db) return false;
  const q = query(collection(_db, 'leaderboard'),
                   where('songKey', '==', songKey), where('nickname', '==', nickname));
  const snap = await getDocs(q);
  return !snap.empty;
}

/** 연주 결과 1건을 리더보드에 저장. 로그인 불필요(공개 쓰기 — Firestore 보안규칙에서
 * leaderboard 컬렉션만 예외적으로 허용해야 함, docs/PLAN_booth_companion_page.md 참고). */
export async function saveLeaderboardEntryCloud(songKey, nickname, score, maxScore) {
  if (!_db) return;
  await addDoc(collection(_db, 'leaderboard'),
               { songKey, nickname, score, maxScore, createdAt: Date.now() });
}

/** songKey 기준 상위 점수 목록 불러오기(1회성). 로그인 불필요.
 * where(songKey) + orderBy(score)를 그대로 Firestore에 보내면 복합 색인(composite index)이
 * 미리 만들어져 있어야 하는데(Firebase 콘솔에서 수동 생성 필요), 안 만들어진 상태면 쿼리
 * 자체가 에러를 던져서 항상 빈 목록으로 보였다(2026-08 확인된 버그) — orderBy를 빼고
 * songKey 단일 조건(색인 불필요)으로만 가져온 뒤 정렬은 여기서 직접 한다. 부스 특성상
 * 곡 하나당 기록 수가 많지 않아 클라이언트 정렬로도 충분하다. */
export async function loadLeaderboardCloud(songKey, top = 3) {
  if (!_db) return [];
  const q = query(collection(_db, 'leaderboard'), where('songKey', '==', songKey));
  const snap = await getDocs(q);
  const list = snap.docs.map(d => d.data());
  list.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  return list.slice(0, top);
}

export { IS_CONFIGURED };
