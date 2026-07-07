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
         deleteDoc, query, where }
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

export { IS_CONFIGURED };
