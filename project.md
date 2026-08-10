# project.md — TransNote: 맞춤형 악보 인식 & 변환 웹 앱

> 세부 문서: 학습 히스토리 [`train/docs/TRAINING_REPORT.md`](train/docs/TRAINING_REPORT.md) ·
> RunPod 체크리스트 [`train/docs/POD_TRAINING_CHECKLIST.md`](train/docs/POD_TRAINING_CHECKLIST.md) ·
> claude.ai/code 연동 [`train/docs/CLOUD_SETUP.md`](train/docs/CLOUD_SETUP.md) · 저장소 구조/명령어
> [`CLAUDE.md`](CLAUDE.md) · 과거 기록(참고용, 학습 인계 상태 포함) [`docs/archive/`](docs/archive/).
> 이 문서는 요약/현황판 역할만 한다.

## 목표

악보의 장벽을 허물어 누구나 음악에 "연결"될 수 있도록 한다. 악보 이미지를 촬영·업로드하면 자체
학습한 OMR 모델이 음표를 인식하고 **사용자 정의 표기법**으로 변환한다. 변환된 악보로 화면 위
가상 피아노 또는 연결된 전자 피아노(MIDI)로 직접 연주 연습할 수 있다.

**타겟 유저**: 악보를 처음 접하거나 읽기 어려운 사람 (현재: 피아노 중심)
**입력 범위**: 이미지 한 장 기준 최대 1~4 마디, 초·중급 수준 악보
**형태**: 웹 앱 단일 구조(`server.py` FastAPI + `webpage/` 정적 프론트) — 전시 부스 태블릿/폰
데모 및 온라인 체험 겸용. Flutter 네이티브 앱 트랙은 2026-08-09 전면 폐기(아래 "과거 결정" 참고).
**제출 마감**: 2026-08-17 — 데모 + 포스터 + 이미지 + 영상 제출.

---

## 현재 상태 (2026-08-09 기준)

- **웹앱 전면 구현 완료**: 랜딩 화면(3개 핫스팟) → 튜토리얼(규칙 0~3 + 테스트 5문항, 가로 화면
  auto-fit) / 체험하기(샘플 3곡 + 카메라 촬영, 오른손만·양손 모드, 마디 단위 자동 진행 연주,
  MIDI 연동, 리더보드) 흐름 전부 동작. 상세는 `webpage/` 코드 참고.
- **OMR 모델**: `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt` 채택 확정
  (재확인 완료, 아래 "OMR 모델 현황" 참고).
- **저장소 재구성 완료(로컬)**: `musicscore_flutter` → `TransNote`, `online_webpage` → `webpage`,
  `round3train` → `train`, 대분류 `webpage/`·`train/`·`test/`·`realImage/` 확립, Flutter/구버전
  ML 파이프라인(`android/`, `ios/`, `lib/`, `ml/`, `round1/`, `omr_bridge/`) 삭제. **git
  commit/push는 아직 미실행** — 사용자 확인 대기 중.
- **HTTPS 제약**: Web MIDI API는 보안 컨텍스트(https:// 또는 localhost) 필요 — 일반 LAN http로는
  MIDI 연동 테스트 불가. 전시 당일 네트워크 구성 시 고려 필요.

---

## OMR 모델 현황

**프로덕션 체크포인트**: `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt` +
`train/tokenizer258.json` (segnet 불필요 — 현재 추론 경로는 오선 검출에 학습 모델을 안 씀,
자세한 근거는 `CLAUDE.md`의 "추론 경로" 절).

| 지표 | 값 | 비고 |
|---|---|---|
| 실사 촬영 12곡 정확도(held-out) | **87.2%** | 최신 채택 수치 |
| teacher-forcing 정확도 | 98.0% | 실전(자기회귀) 수치와 격차 큼, 신뢰 안 함 |
| 촬영 노이즈 대응 전 정확도 | 21.8% | 노이즈 증강 도입 전 — 약 4배 개선됨 |

r15 이후 시도한 r16(박자표 미노출 보강)/r17(리듬 분포 보강)은 **둘 다 실사 검증에서 r15를
넘지 못해 기각** — r15가 계속 유일한 프로덕션 체크포인트다. 전체 라운드별 데이터/에폭/정확도와
문제 해결 과정(exposure bias 완화, 노이즈 강건성, 마르코프 체인 기반 피치 가중, r16/r17 기각
근거)은 [`train/docs/TRAINING_REPORT.md`](train/docs/TRAINING_REPORT.md) 참고.

**남은 정확도 이슈**(우선순위 순, `docs/archive/HANDOFF_STATUS.md`에서 이관):
1. newage23 GT 데이터 버그(어휘에 없는 토큰) — 검증 정확도 왜곡, 저비용 수정 가능
2. 3도 오독(단/장3도 음이름 혼동) — 전 검증셋 공통 최다 오류, 근본 원인 미해결
3. 옥타브(8va/8vb)/헤어핀 span 토큰 — 구조적 한계로 recall 20~37%만 지원

**알려진 인식 범위**: 대보표/단일오선, 시스템당 1~4마디, 박자 4종(4/4·3/4·2/4·6/8), 조표 13종,
음표 온음표~16분음표, 화음(2~3음)·다이나믹·페르마타·셋잇단음표·붙임줄·클렙전환. 음높이
C2~B6. **스코프 밖**: 도돌이표, 아티큘레이션/오나먼트/슬러, 5마디 이상 시스템.

---

## RunPod 배포 번들

`train/deploy_bundle/`에 프로덕션 체크포인트(seq2seq_best.pt) + tokenizer258.json 2개 파일만
스테이징해둠(README 포함, git 미추적 — 재생성은 원본 두 경로에서 복사). segnet은 현재 추론
경로에서 안 쓰여서 제외. 상세는 [`train/deploy_bundle/README.md`](train/deploy_bundle/README.md).

---

## 코드 정리 현황 (`.py` orphan 감사)

`train/` 최상위 활성 스크립트(~14개) 기준 아무 코드에서도 참조되지 않는 파일:

| 파일 | 사유 | 조치 |
|---|---|---|
| `train/export_tflite.py` | 삭제된 Flutter/C++ 모바일 엔진 전용 export — 현재 웹서버는 PyTorch 직접 추론이라 불필요 | 유지(향후 네이티브 앱 재추진 시 필요할 수 있어 삭제는 보류, 삭제 원하면 알려주세요) |
| `train/tokenizer1013.json` | vocab 분할(1013→258) 이전 구버전, 참조 0건 | 삭제해도 안전 |
| `train/tokenizer258_pre_tie.json` | tie 토큰 추가 이전 스냅샷, `train/experiments/`의 아카이브된 1회성 스크립트에서만 참조 | 유지(용량 작음, 참고용) |

`train/experiments/`(과거 라운드별 curriculum 셸스크립트 ~100개)와 `test/`(평가 스크립트
~32개)는 이미 "이력 보존용 아카이브"로 격리되어 있어 이번 감사 범위에서 제외했다 — 개별 스크립트
전수 감사가 필요하면 별도로 요청.

---

## 디자인 계획

**웹 UI**: `webpage/` 자체 그라데이션/카드 기반 UI, 앱형 하단 탭바 + 랜딩 핫스팟 네비게이션.
Glory Music/QuickScan UI 킷은 Flutter 시절 참고 자산으로 현재 웹 UI에는 직접 채택하지 않음(웹
전용 톤앤매너로 별도 진행 중) — git 저장소에는 포함하지 않고 로컬에만 보관.

---

## 세부 문서

| 문서 | 내용 |
|---|---|
| [`docs/music-notation-rule-designer.md`](docs/music-notation-rule-designer.md) | DeepScore 토큰 vocabulary, 커스텀 표기법 시각 규칙 — 코드(`webpage/js/notation.js`, `train/generate_scores.py` 등)가 실제로 참조 |
| [`train/docs/TRAINING_REPORT.md`](train/docs/TRAINING_REPORT.md) | 학습 히스토리·라운드별 정확도·최종 결과 통합 리포트 |
| [`docs/PLAN_booth_companion_page.md`](docs/PLAN_booth_companion_page.md) | 부스 컴패니언 페이지(QR) 계획 — 구현 전 |

> `project-orchestrator`/`score-training-agent`/`score-recognition-engine`/`sheet-music-qa`/
> `ui-design-specialist`/`flutter-integration-architect`/`round3-phase2-retrain` 문서는 전부
> Flutter/`ml/`/`round3train/` 시절(삭제된 경로) 기준이라 [`docs/archive/`](docs/archive/)로
> 이동됨 — 현재 구조 파악엔 참고하지 말 것.

---

## 기술 스택 (현재)

| 항목 | 내용 |
|---|---|
| 서버 | FastAPI(`server.py`), 정적 파일 서빙 + `/api/recognize`·`/api/status`·`/api/score`·`/api/qr` |
| 프론트엔드 | 바닐라 JS(`webpage/js/`), SVG 커스텀 표기법 렌더링, Web Audio(연주 합성), Web MIDI(전자 피아노 연동) |
| OMR 모델 | PyTorch(CNN 인코더 + Transformer 디코더), `train/` 자체 학습 |
| 학습 데이터 | `music21` 생성(`train/generate_scores.py`, 마르코프 체인 가중 피치) + 실사 사진(`realImage/`) |
| 라벨 형식 | DeepScore 토큰 시퀀스(`tokenizer258.json`) |
| DB | Firebase(닉네임 저장, 무료 티어) — 클라이언트 SDK `webpage/js/firebase.js` |
| 호스팅 | Vercel(무료 플랜) 준비 중 |

---

## 과거 결정 (요약)

- **2026-07-30**: 학습 계보 전면 재시작(exposure bias·클렙 편향 등 구조적 문제로 처음부터
  Round1~3 재설계). 이전 로드맵(`TrainingStep.md`)은 이 시점에 폐기됨.
- **2026-08-05**: r15를 프로덕션 체크포인트로 확정, r16/r17 시도 후 기각.
- **2026-08-09**: Flutter 앱 트랙 전면 폐기, 웹 단일 구조로 전환. 저장소 재구성
  (`TransNote`/`webpage`/`train`/`test`/`realImage`), 학습 히스토리 문서 통합
  (`train/docs/TRAINING_REPORT.md`), `.py` orphan 감사.
