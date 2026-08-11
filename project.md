# project.md — TransNote: 맞춤형 악보 인식 & 변환 웹 앱

> 세부 문서: 저장소 구조/명령어 [`CLAUDE.md`](CLAUDE.md). 모델 학습 코드·히스토리·정확도
> 근거는 별도 저장소 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote) 참고
> (2026-08-12, 학습 관련 문서/코드를 그쪽으로 이관). 이 문서는 요약/현황판 역할만 한다.

## 목표

악보의 장벽을 허물어 누구나 음악에 "연결"될 수 있도록 한다. 악보 이미지를 촬영·업로드하면 자체
학습한 OMR 모델이 음표를 인식하고 **사용자 정의 표기법**으로 변환한다. 변환된 악보로 화면 위
가상 피아노 또는 연결된 전자 피아노(MIDI)로 직접 연주 연습할 수 있다.

**타겟 유저**: 악보를 처음 접하거나 읽기 어려운 사람 (현재: 피아노 중심)
**입력 범위**: 이미지 한 장 기준 최대 1~4 마디, 초·중급 수준 악보
**형태**: 웹 앱 단일 구조(`server.py` FastAPI + `webpage/` 정적 프론트, Vercel+RunPod 하이브리드
배포) — 전시 부스 태블릿/폰 데모 및 온라인 체험 겸용.
**제출 마감**: 2026-08-17 — 데모 + 포스터 + 이미지 + 영상 제출.

---

## 현재 상태

- **웹앱 전면 구현 완료**: 랜딩 화면(3개 핫스팟) → 튜토리얼(규칙 0~3 + 테스트 5문항, 가로 화면
  auto-fit) / 체험하기(샘플 3곡 + 카메라 촬영, 오른손만·양손 모드, 마디 단위 자동 진행 연주,
  MIDI 연동, 리더보드) 흐름 전부 동작.
- **배포 완료**: [trans-note.vercel.app](https://trans-note.vercel.app) 라이브 — Vercel(정적
  서빙 + 서버리스 프록시) + RunPod Serverless(GPU 추론, Docker) 하이브리드. 로컬/LAN 실행은
  `server/server.py`(FastAPI)로 별도 지원(전시 부스용).
- **OMR 모델**: `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt` 채택 확정, 주 타겟
  장르인 뉴에이지 실사 테스트 기준 약 96% 인식 정확도(상세 근거는 Model_TransNote 참고).
- **HTTPS 제약**: Web MIDI API는 보안 컨텍스트(https:// 또는 localhost) 필요 — 일반 LAN http로는
  MIDI 연동 테스트 불가. 전시 당일 네트워크 구성 시 고려 필요.

---

## 디자인 계획

**웹 UI**: `webpage/` 자체 그라데이션/카드 기반 UI, 앱형 하단 탭바 + 랜딩 핫스팟 네비게이션.

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| 서버 | FastAPI(`server.py`), 정적 파일 서빙 + `/api/recognize`·`/api/status`·`/api/score`·`/api/qr` |
| 프론트엔드 | 바닐라 JS(`webpage/js/`), SVG 커스텀 표기법 렌더링, Web Audio(연주 합성), Web MIDI(전자 피아노 연동) |
| 배포 | Vercel(정적 + 서버리스 프록시) + RunPod Serverless(GPU 추론, Docker — `.github/workflows`가 자동 빌드) |
| OMR 모델 | PyTorch(CNN 인코더 + Transformer 디코더) — 학습 코드/히스토리는 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote) |
| DB | Firebase(닉네임/점수 저장, 무료 티어) — 클라이언트 SDK `webpage/js/firebase.js` |

---

## 과거 결정 (요약)

- **2026-08-05**: r15를 프로덕션 체크포인트로 확정, r16/r17 시도 후 기각(상세는 Model_TransNote).
- **2026-08-09**: Flutter 앱 트랙 전면 폐기, 웹 단일 구조로 전환.
- **2026-08-12**: 저장소를 서버/배포 중심으로 정리 — 학습 코드·문서·평가 스크립트
  (`train/experiments/`, `train/docs/`, `test/`)를
  [Model_TransNote](https://github.com/braeden-hue/Model_TransNote)로 이관하고 이 저장소에서
  제거. Flutter 시절 기획 문서(`docs/archive/`, `docs/PLAN_booth_companion_page.md`)·옛
  서브에이전트 메모리(`.claude/agent-memory/`)도 함께 제거 — 단 `docs/music-notation-rule-designer.md`는
  `webpage/js/notation.js` 등 코드가 실제로 참조하는 활성 문서라 유지. `train/` 최상위
  `.py`/`tokenizer258.json`은 RunPod Docker 빌드가 실제로 복사해가는 런타임 의존 파일이라
  그대로 유지.
