# RunPod OMR 커리큘럼 학습 현황 (PODPLAN)

마지막 업데이트: 2026-07-21 (3차, 세션 종료)

## 최종 상태 요약 (2026-07-21, 3차) — 이번 세션 최종 확정

**최종 체크포인트: `secrets/checkpoints/seq2seq_p2s4span_w10_best.pt`** (193,593,204 bytes,
sha256 `a97d873f0789ebaeeb5bd4c403bf7df33a24601e5884af3fb22ed3b29d3ad2cc`, pod 원본과 바이트 단위 검증 완료)
계보: `4gm(96.8%) → 4t(95.37%) → 4k(96.72%) → 4dyn(98.45%) → 4sym(97.64%) → 4ott(97.68%) → 4tup(96.38%)
→ 4span_w10(TER 5.3%, span_weight=10 적용)`. Phase 3(전체 언프리즈 파인튜닝)는 **실패로 확정, 사용 안 함**
(아래 참고). segnet은 전 라운드 공통 재사용(`secrets/checkpoints/segnet_best.pt`).

### 커리큘럼 확장 (4t~4tup, 2026-07-21)
기존 확정 스코프(박자 6/8, 조표 13종, 다이나믹, 페르마타+헤어핀)에 이어 **옥타브+셋잇단음표 추가**
(스코프 재조정, tie/붙임줄은 미구현이라 제외):
`4t(95.37%) → 4k(96.72%) → 4dyn(98.45%) → 4sym(97.64%) → 4ott(97.68%) → 4tup(96.38%)` 순서로
전부 게이트(95%) 통과. `curriculum_4t_4sym.sh`의 `STAGE_NAMES` 배열로 관리(6단계).

**scheduled sampling(4dyn부터 적용)**: `train.py`에 word-dropout(`<UNK>` 마스킹) 방식으로 시작했다가,
이후 **모델 자신의 예측으로 대체하는 진짜 self-token 방식**으로 업그레이드(`--tf_ratio`/`--min_tf_ratio`/
`--ss_epochs`, 1차 no-grad forward로 자기예측 획득 후 대체). exposure bias 완화 목적.

### 핵심 발견 — 실사용(자기회귀 free-running) 정확도는 teacher-forcing 게이트와 별개
`error_breakdown.py --beam_width`로 실측한 완전 자기회귀 exact-match는 4t~4tup 전 구간에서
65~77% 수준(게이트 95%+ 와 괴리) — 원래도 있던 gap이며 이번에 처음 정밀 측정. **beam search(이미
구현돼 있었으나 평가 경로에 연결 안 돼 있던 것을 `inference.py`/`error_breakdown.py`에 연결)는
효과가 미미**(45→43건 오류, exact-match 불변).

### 옥타브·헤어핀 span 토큰 인식 문제 (발견 → 부분 완화, 미해결로 세션 종료)
`symbol_recall.py`류 진단 스크립트(pod `/tmp/`, 로컬 미보존 — 재현 필요시 새로 작성)로 정밀 측정한 결과:
- 음표(음이름+옥타브) recall 96.5~97.4%, 길이(duration) 98.1~99.3%, 셋잇단음표 98.5~101%,
  페르마타 90.5%, **다이나믹(셈여림) 95.6~100%** — 전부 양호
- **옥타브(ottava) 시작/종료 recall 13~30%, 헤어핀(hairpin) 시작/종료 recall 14~37%** — 압도적으로 나쁨

원인: 둘 다 여러 음표를 거쳐 1~2마디 뒤에야 닫히는 **긴 구간(span) 표기**라(셋잇단음표·페르마타는
짧거나 즉시 끝남), 시퀀스당 시작+종료 2개 토큰뿐이라 노출을 아무리 늘려도 토큰 단위 loss 신호가
희소함. 시도한 완화책과 결과(4tup 기준 옥타브 20%/헤어핀 14~29% 대비):
| 시도 | 옥타브 | 헤어핀 | 비고 |
|---|---|---|---|
| exposure만 2~3.5배 증가(4span) | 16.7~30% | 17.1~31.4% | 거의 무효과 |
| **+ loss 가중치 10x(4span_w10, 최종 채택)** | **23.3~25%** | **20~37.1%** | 소폭이지만 실제 개선, 음표/길이도 향상 |
| 2마디 고정+90%노출+가중치20x+3000장/30epoch(4span2m) | 13.3~20% | 17.1~28.6% | **오히려 악화** — 좁은 분포(2마디, 고밀도)에 과적합돼 원래 평가 분포(1~4마디)로 일반화 실패 |

**결론(세션 종료 시점 확정): `4span_w10`을 최종으로 채택.** 옥타브·헤어핀은 20~37% 수준의
"부분 지원" 상태로 남겨두고 추가 시도 중단(시간 대비 반복 실패). 재도전 시 참고:
- 2마디로 좁히는 접근은 평가 분포와 괴리를 만들어 실패 — 재시도한다면 학습/평가 분포를 반드시 통일할 것
- loss 가중치는 방향은 맞았음(10x에서 유일하게 개선) — 더 높은 가중치(25~30x) 또는 다른 근본 기법
  (span 상태를 명시적으로 추적하는 보조 loss/토큰 등) 고려 가치 있음

### Phase 3(전체 언프리즈 파인튜닝) — 실패, 사용 안 함
4t~4tup 6단계 각각의 replay 500장씩(총 3000장, `round1_stage{4t,4k,4dyn,4sym,4ott,4tup}_new`에서
병합)으로 freeze 없이 20epoch 파인튜닝 시도. 결과: 같은 평가셋에서 **4tup(39.6%)보다 낮은 35.6%**로
오히려 퇴보, beam search로도 36.2%까지만 회복. 원인 미상(가설: self-token scheduled sampling이
`min_tf_ratio=0.5`로 처음부터 전체 언프리즈 상태와 결합되며 과했을 가능성). **폐기, 재사용 안 함.**

### 인프라 교훈 (이번 세션 신규)
- **중복 스크립트 사본 함정**: `/workspace/<파일명>`과 `/workspace/round3train/<파일명>`이 별도로
  존재하며 어긋날 수 있음(실제로 `validate_stage4_data.py`/`gen_render_local.sh`가 겪음, 4k가 같은
  이유로 두 번 막힘). `gen_render_local.sh`를 `SCRIPT_DIR`(자기 위치 기준 상대경로) 패턴으로 고쳐 근본
  해결. 새 스크립트 작성 시 항상 이 패턴 사용할 것.
- **실행 중인 스크립트 파일은 절대 직접 덮어쓰지 말 것**: `cat > file`(truncate)이 아니라 `.new`로
  써서 `bash -n` 문법 검증 후 `mv`로 원자적 교체.
- **ssh.runpod.io 프록시로 큰 바이너리 pull 시 손상 발생**: `base64 -w0`(긴 한 줄)는 pty 렌더링 중
  깨짐. 표준 줄바꿈 base64 + `sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g'`(ANSI 제거) + 길이 필터
  (`grep -E '^[A-Za-z0-9+/]{20,}=*$'`, 타이핑한 `exit` 같은 명령 에코가 우연히 base64 패턴에 걸리는
  것 방지)로 15MB 단위 청크 분할 + 청크별 sha256 검증이 신뢰 가능한 유일한 방법이었음(193MB 체크포인트
  1개를 13개 청크로 전송, 전부 검증 후 병합). `scp`/`sftp`는 이 프록시에서 subsystem 자체가 거부됨.
- **push(로컬→pod)는 base64 heredoc으로 항상 안정적**(stdin 경유라 pty 렌더링 문제 없음) — 문제는
  pull(pod→로컬, stdout/pty 경유) 방향에서만 발생.
- pod stop은 여전히 API 자동화 없이 수동으로만(RunPod 대시보드) — 이번 세션도 동일하게 처리.

## 최종 상태 요약 (2026-07-20, 2차)

**4g 이후 "화음+조표+마디확장(1~4, 밀도 기반 줄바꿈)" 체인(4gchord~4gm) 완료.** 최종(현재 최신) 체크포인트:
- `round3train/models/round1_curriculum_p2s4gm/seq2seq_best.pt` (Acc 96.8%)
- `secrets/checkpoints/seq2seq_p2s4gm_best.pt` (동일 파일 백업)
- `secrets/checkpoints/segnet_best.pt` (Phase1 오선 탐지, 변경 없음)
- (참고) `round3train/models/round1_curriculum_p2s4g/seq2seq_best.pt` (Acc 99.22%, 화음/조표/마디확장 이전 체크포인트, 4gm의 상위 호환이라 보통 4gm만 쓰면 됨)

아래 "4g 이후: 화음/조표/마디확장 (4gchord~4gm)" 절 참고. Pod는 작업 종료 후 정지시켜둠.
그 이전 duration 세부 커리큘럼(4a~4g)은 "Stage4 duration 세부 커리큘럼 (4a~4g)" 절 참고.

## 개요

`round3train/` 신규 OMR seq2seq 모델(CNN 인코더 + Transformer 디코더, 객체 탐지 아님 — 좌→우 autoregressive 토큰 생성)을 커리큘럼 러닝으로 학습 중. 난이도를 4단계로 나눠 각 단계마다 이전 단계 데이터 30% replay를 섞어 재앙적 망각을 방지하며 진행.

- 데이터: `music21` + MuseScore4 CLI 렌더링(`xvfb-run`)으로 합성 악보(MusicXML→PNG) 생성
- SegNet(오선/픽셀 분할)은 Phase1에만 쓰이고, Phase2(seq2seq) 학습에는 `detect_staffs()`(OpenCV 고전 알고리즘)만 사용 — SegNet은 Stage1~4에서 재학습 없이 그대로 재사용됨
- 각 단계: 25epoch, epoch7 조기경보(loss가 epoch1→epoch7에 개선 안 되면 발산으로 간주해 중단)
- 각 단계 학습 후, **학습에 전혀 쓰이지 않은 새 seed의 같은 난이도 테스트셋(300장)**을 별도 생성해 평가 — validation split 재사용/replay 오염 없는 "클린" 정확도 확보용

## 단계별 현황

| Stage | 내용 | 신규/replay | 결과 (val, 오염됨) | 클린 테스트셋 결과 |
|---|---|---|---|---|
| Stage1 | 고립된 음표/쉼표 | 3000 / - | TER 0%, Acc 100% (자명한 과제, 정상) | - |
| Stage2 | 온전한 마디+세로줄 | 3000 / 900(S1) | TER 2.3%, Acc ~97.7% | TER ~2.6%, Acc ~97.4% |
| Stage3 | 임시표 도입 | 3000 / 900(S1+S2) | TER 1.5%, Acc 98.5% | **TER 2.1%, Acc 97.9%** (seed 12345) |
| Stage4 | 대보표(오선 2개, treble+bass) | 3000 / 900(S1+S2+S3) | TER 20.5%, Acc 79.5% (25epoch 완료) | 재학습 예정으로 스킵 |

**현재 가장 신뢰할 수 있는 체크포인트: Stage3 (`seq2seq_curriculum_stage3_best.pt`, Acc 97.9%)**
Stage4는 아래 이유로 재학습 예정이라 79.5% 체크포인트는 임시 상태.

## Stage4 원인 분석 (Acc가 79.5%에 그친 이유)

1. **시퀀스 길이 1.7배 증가**: Stage3 평균 토큰 13.8개 → Stage4 평균 23.8개(최대43). 토큰 구조가 `[treble마디] staff-bass clef-F [bass마디] barline` 형태로 마디당 treble+bass를 순차 생성해야 함.
2. **Teacher forcing이 항상 1.0(exposure bias)**: 학습은 항상 정답 토큰을 다음 입력으로 사용하지만, 검증은 `greedy_decode`로 완전 자기회귀 생성. 시퀀스가 짧았던 Stage1~3은 이 문제가 덜 드러났지만, Stage4는 길어진 만큼 한 번의 실수가 전체로 전파되기 쉬움 (특히 treble→bass 전환 지점에서 틀리면 뒤쪽 전체가 무너짐).
3. **학습 곡선이 이미 플래토**: epoch20~25 loss가 1.27대에서 정체, TER도 20~22% 사이에서 진동만 함 — 발산이 아니라 현재 레시피(3900장/25epoch)로는 이 새 구조를 다 배우기엔 데이터/epoch가 부족한 정상적 한계.

## Stage4 재학습 계획 (진행 예정)

| 항목 | 기존 | 변경 |
|---|---|---|
| 신규 데이터 | 3000장 | 6000장 |
| replay | 900장(30%) | 1800장(30%, S1/S2/S3 비례 증가) |
| 총 학습 데이터 | 3900 | 7800 |
| epoch | 25 | 40 |
| freeze(encoder 동결) | 5 (epochs÷5, 고정) | 12 (`--freeze_epochs` 독립 인자 추가 필요) |
| scheduled sampling | 없음 | freeze 종료(epoch12) 시점부터 시작, 20epoch에 걸쳐 tf 1.0→0.5로 감소 후 유지 (min_tf_ratio 기본 0.1은 과격해서 0.5로 완화) |
| 조기경보 probe epoch | 7 | 14 (freeze 종료+2epoch) |

### 코드 수정 필요 사항 (`round3train/train.py`)
- `--freeze_epochs` 인자 추가 (없으면 기존처럼 `epochs//5` fallback)
- scheduled sampling(tf 감소)이 **freeze 종료 시점부터** 시작하도록 수정 (현재는 epoch1부터 감소 시작 — freeze 연장과 충돌 가능)

### 주의사항
- **key는 C장조 고정 유지, 마디 수도 1마디 고정 유지** — round1_easy에서 조표/여러 마디를 처음 도입하는 커리큘럼 설계를 지키기 위해, 이번엔 "오선 개수" 외 다른 변수는 건드리지 않음. (실제 확인: 기존 Stage4 3000장 전부 key-C 확인됨)
- **디스크 여유 재확인 필요**: `_pre.npy` 캐시(샘플당 ~925KB, 1920px 폭 전처리 grayscale)가 학습 1epoch차에 새로 쌓이는 게 진짜 디스크 병목. 7800샘플 기준 캐시만 ~7.6GB. 기존 실패한 Stage4 학습 데이터(3.8GB) 삭제 후 시작해도 피크 사용량이 이전 사고 지점(~40G)과 비슷 — 재발 방지로 학습 중 디스크 사용량 모니터링 강화 필요.
- 재학습 전 현재 Stage3 체크포인트(Acc 97.9%, 가장 신뢰 가능한 상태)를 백업해뒀음.

## 인프라 이슈 및 교훈 (재발 시 참고)

- **Pod 재시작마다 컨테이너 디스크 초기화됨** — `cv2`(opencv-python-headless), `music21`, `xvfb` 등은 `/workspace`(네트워크 볼륨)가 아니라 컨테이너 로컬에 설치되므로 재시작 후 `ModuleNotFoundError: No module named 'cv2'` 등으로 학습이 즉시 죽음. 재시작 후엔 항상 `pip install --break-system-packages opencv-python-headless music21` + `apt-get install -y xvfb` 먼저 확인.
- **네트워크 볼륨(50GB, MooseFS) 디스크 할당량 초과 시 조용히 실패**(0바이트 파일, exit code는 정상으로 보일 수 있음) — `du`/`df` 표시치와 실제 할당량 강제 시점이 안 맞을 수 있어 `dd if=/dev/zero of=test bs=1M count=1` 로 실제 쓰기 테스트하는 게 가장 확실.
- **pgrep -f로 중복 프로세스 탐지 시 DataLoader worker(부모 argv 그대로 가짐)를 오탐 안 하도록 wrapper 프로세스(`bash -c mkdir -p <out_dir> ...`) 패턴으로만 판단**해야 함.
- **stop_pod() API 호출은 원인 불명의 자동 종료 문제로 전부 주석 처리됨** — 수동으로 꺼야 함.
- SSH 연결이 순간적으로 끊기는 경우가 잦아, 감시 스크립트는 반드시 재시도 로직(연속 실패 N회 이상일 때만 진짜 종료로 판단)을 넣어야 오탐을 피할 수 있음.

## Stage4 duration 세부 커리큘럼 (4a~4g, 2026-07-19~20)

Stage4(대보표) 자체는 4a(4분음표만)에서 99.3%까지 올라가 구조 자체는 학습 가능함을 확인했으나,
이후 duration 종류를 하나씩 늘리는 세부 커리큘럼(4b~4g)에서 8분음표 도입 시점(4c)부터
정확도가 83~86%에서 막히고 90% 게이트를 통과하지 못하는 문제가 반복됨. 아래 순서로 원인을 진단.

### 기각된 가설 (순서대로 테스트)
1. **비밍(beaming)**: 연속된 8분음표가 이어진 모양으로 렌더링되어 모양이 달라지는 것 아닌가 —
   실제 렌더 이미지 직접 확인 결과 music21→MuseScore export는 항상 개별 깃발(flag)만 쓰고
   비밍 없음. 기각.
2. **기둥(stem) 방향/음높이 범위 다양성**: `--narrow-pitch` 플래그로 음높이를 오선 중간선
   아래로 좁혀 기둥 방향을 항상 위쪽으로 고정해 재테스트 → 74.7% vs 기존 75.0%, 차이 없음. 기각.
3. **렌더링 해상도(업스케일 블러)**: `dataset.py`의 `preprocess()`가 `TARGET_W=1920`px로
   강제 리사이즈하는데 기존 150dpi 렌더는 1240px라 매번 업스케일(블러)됨을 확인, `RENDER_DPI=300`
   (2481px, 다운스케일)으로 재테스트 → 74.8% vs 75.0%, 차이 없음. 기각.

### 진짜 원인: `dataset.py`의 `extract_system_canvas()` 캔버스 잘림 버그 (발견 및 수정)

GT-예측 토큰 정렬 분석(`round3train/error_breakdown.py`)으로 오류를 종류별(음이름/길이/note-rest
혼동/누락·과잉)로 집계한 결과, duration 자체보다 **음이름 오류·음표 개수 불일치**가 압도적으로
많다는 게 드러남 → "한 마디에 음표가 많을수록(8분음표는 4분음표의 2배) 밀도가 높아져 문제가
생기는 것 아닌가"라는 가설로 이어짐.

직접 확인 결과, `extract_system_canvas()`가 오선 내용을 높이 기준으로만 스케일링한 뒤
`CANVAS_W`(1280px)에 맞춰 **패딩하거나(내용이 좁으면) 그냥 잘라내는(내용이 넓으면) `_pad()`**
함수를 쓰고 있었는데, 후자가 진짜 리사이즈가 아니라 **단순 슬라이싱(truncate)**이었음:

```python
def _pad(strip):
    tile = np.full((half_h, CANVAS_W), 255, dtype=np.uint8)
    cw = min(CANVAS_W, strip.shape[1])
    tile[:, :cw] = strip[:, :cw]   # 내용이 CANVAS_W보다 넓으면 나머지가 통째로 잘려나감
    return tile
```

측정 결과: 4a(4분음표 4개/마디)는 높이 정규화 후 폭 ~560px로 CANVAS_W 안에 들어오지만,
4b_1(8분음표 8개/마디, 4/4 한 마디 꽉 채움)은 폭 ~2384px로 계산되어 **약 46%가 잘려나감** —
즉 GT 토큰은 8개 음표를 요구하는데 이미지에는 마지막 3~4개 음표가 아예 존재하지 않는
상태로 학습되고 있었음. 단일 오선 경로(`extract_staff_canvas`)는 애초에 `cv2.resize`로
제대로 리사이즈하고 있어서 이 버그가 없었고, 대보표 경로(`extract_system_canvas`, Stage4에서
실제 사용하는 경로)만 영향을 받음.

**수정**: treble/bass 각각의 높이 기준 스케일을 유지하되, 스케일된 폭이 `CANVAS_W`를 넘으면
두 오선에 동일한 비율(시간축 정렬 유지 위해 공유)로 추가 축소해서 항상 캔버스 안에 들어오도록
변경 (`round3train/dataset.py`의 `extract_system_canvas()`).

### 버그 수정 전/후 비교 (동일 조건 재학습)

| 대상 | 수정 전 | 수정 후 |
|---|---|---|
| 4b_1 (1/8 단독 진단, 4/4 한 마디 8개) | 75.0% | **99.8%** |
| 4c (신규 duration=1/8, 쉼표 없음) | 83~86% (게이트 미달, 중단) | **99.5%** |
| 4b (온음표/2분음표/4분음표, 밀도 낮아 원래도 잘림 적음) | 91.98% | 93.27% (소폭 개선만 — 밀도 낮은 단계는 버그 영향도 작다는 걸 방증) |

### 4b~4g 전체 재검증 결과 (수정된 코드, 게이트는 단계가 누적될수록 90%→78%까지 완만히 낮춤)

| 단계 | 신규 내용 | 최고 Acc | 게이트 |
|---|---|---|---|
| 4a | 4분음표만 | 99.32% | - |
| 4b | +2분/온음표 | 93.27% | 90% |
| 4c | +8분음표 (쉼표 없이) | 99.51% | 87% |
| 4c2 | 8분음표 + 쉼표 복원 | 98.43% | 87% |
| 4d | +16분음표 (쉼표 없이) | 100.00% | 84% |
| 4d2 | 16분음표 + 쉼표 복원 | 98.77% | 84% |
| 4e | +점8분음표(3/8) (쉼표 없이) | 99.02% | 81% |
| 4e2 | 3/8 + 쉼표 복원 | 99.75% | 81% |
| 4f | +점16분음표(3/16) (쉼표 없이) | 100.00% | 78% |
| 4f2 | 3/16 + 쉼표 복원 | 99.91% | 78% |
| 4g | 최종(전체 duration+쉼표, 박자 3종 혼합, 4000장/40epoch) | **99.22%** | 78% |

전 단계 통과, `curriculum_4b_to_4g.sh`가 자동으로 4g까지 완주함(중간 게이트 미달 시 자동 중단
하도록 설계되어 있었으나 이번엔 발동 안 함).

**[2026-07-20 정정] 4g 실제 학습 범위 재확인**: `curriculum_4b_to_4g.sh`의 `GEN_ARGS`가
`--min-measures 1 --max-measures 1 --force-c-major`를 **4b~4g 전 단계에 예외 없이** 적용하고
있음을 뒤늦게 확인함. 즉 4g도 **항상 1마디, 항상 다장조(C major)**로만 학습됐고, 애초 설계
주석(`STAGE_NAMES` 위 주석 등)에 있던 "마디 2~6개, 조표 다양화"는 실제로 구현되지 않았음(설계
의도와 실제 실행 스크립트가 처음부터 어긋나 있었음, 이번 세션에서 실행할 때도 못 걸러냄).
다만 **박자(time signature)는 4g만 다양함** — 4g는 `--duration-subset`을 안 써서(빈 문자열)
4/4 고정 로직이 안 걸리고 기본값(4/4 50%·3/4 30%·2/4 20%)이 그대로 적용됨. 정리하면 4g는
"다장조·1마디·박자 3종 혼합·전체 duration+쉼표" 인식기이며, **여러 마디/다른 조표는 전혀
검증되지 않은 영역**. 아래 "다음 인식 대상" 절 참고.

### 파이프라인/스크립트 관련 참고
- `round3train/gen_render_local.sh`에 `RENDER_DPI` 환경변수 추가(기본 150, 필요시 `RENDER_DPI=300`
  등으로 오버라이드) — 해상도 가설 테스트용으로 추가했으나 가설 자체는 기각됨. 다만 옵션은 남겨둠.
- `round3train/generate_scores.py`에 `--time-sig` 옵션 추가 — `--duration-subset`이 강제하는 4/4
  대신 다른 박자(예: 2/4)를 쓸 수 있게 함. 밀도 가설 검증용으로 추가.
- `round3train/error_breakdown.py` 신규 스크립트 — 검증셋 전체에 대해 GT-예측 토큰을 정렬해서
  오류를 종류별(음이름/길이/note-rest 혼동/누락/과잉)로 집계. 향후 새 정체 구간이 생기면 먼저
  이걸로 원인 카테고리부터 좁힐 것.
- `round3train/curriculum_4b_to_4g.sh`에 `GATE_THRESHOLDS` 배열 추가 — 단계가 누적될수록
  90→87→84→81→78(바닥)로 완만히 낮아지는 게이트 적용.

## 로컬 저장소 정리 (2026-07-20)

다음 항목들을 삭제(모두 4g 결과로 superseded되었거나 재생성 가능한 산출물):
- `build/` (1.6G, Flutter 빌드 산출물, 재생성 가능)
- `round3train/__pycache__/`, 오래된 진단 로그(`analyze_errors_*.log`), Word 잠금파일, 오래된 flutter 로그
- `round3train/checkpoints_by_epoch/` (1.1G, 구버전 Round1 phase1/2/3 파이프라인의 10에폭 단위 스냅샷)
- `pod_bundle_stage3/` (215M, stage3까지만 학습된 구버전 번들)
- `omr_latest_weights/` (1.7G, 이미 `.gitignore`에 "프로토타입 leftover"로 표시되어 있던 구버전 가중치)
- 루트의 `seq2seq_68.pt` (185M, 출처 불명 루트 파일, checkpoints_by_epoch와 같은 계열로 추정)
- `round3train/HANDOFF_STATUS.md`, `round3train/RETRAIN_GUIDE.md` (구버전 Phase1/2/3 인계 절차 —
  이제 `curriculum_4b_to_4g.sh` + 이 문서로 대체됨)
- `secrets/checkpoints/seq2seq_p2s4b_best.pt` → `seq2seq_p2s4g_best.pt`로 교체(최종 체크포인트만 보관)

## 4g 이후: 화음/조표/마디확장 (4gchord~4gm, 2026-07-20)

Stage4(4a~4g) 완료 후, "대보표 하나 + 최대 4마디" 최종 스코프를 목표로 남은 축(화음, 조표,
마디 수, 악보 기호)을 하나씩 추가하는 2차 커리큘럼 진행. 전부 4g(1마디, 다장조 고정,
99.22%)에서 시작해 이어감.

### 진행 결과

| 단계 | 내용 | resume | data | epoch | replay | 결과 |
|---|---|---|---|---|---|---|
| 4gchord | 화음 2음, 다양한 길이 | 4g | 3000 | 30 | 없음 | 99.0% |
| 4gchord2 | 화음 2~3음(임시표 가능) | 4gchord | 3000 | 25 | 40% | 98.3% |
| 4gkey | 조표 다양화(C/G/F/D/Bb) | 4gchord2 | 3000 | 25 | 40% | 99.0% |
| **4gm** | **마디 1~4(밀도 기반 줄바꿈) + 화음 + 조표** | 4gkey | 6000 | 40 | 40%(1200) | **96.8%** |

### 핵심 발견 1 — 화음 임시표 도입 시 조표 위험 요소 (해결됨)

조표를 C장조 외로 넓히면, 같은 피치 토큰(예: `note-F#4`)이 **조표에 따라 임시표 표시 여부가
달라지는**(G장조는 F#이 조표에 이미 있어 임시표 없음, C장조는 임시표 필요) 새로운 시각적
모호성이 생김. `generate_scores.py`에 `_pick_pitch()`/`DIATONIC_BIAS`(기본 0.75) 추가 —
조표 음계에 속한 음(철자까지 정확히 일치, pitchClass만으론 부족 -- D장조에서 Db 대신 C#을
쓰는지까지 확인)을 우선 선택해서 낯선 조합의 빈도를 실제 음악처럼 낮춤. `--diatonic-bias`로
조정 가능.

### 핵심 발견 2 — 마디가 여러 개일 때 "줄바꿈" 방식 (중요, 근본적 수정)

**처음 시도(고정 마디수 기준 줄바꿈, `--auto-measures-per-system`)는 실패.** 2~3마디를 화음+
조표와 함께 단일 시스템(한 줄)에 그대로 렌더링하는 사전테스트 결과 **66.8%로 실패** — 캔버스
폭(1280px) 대비 내용이 0.537배로 압축되는 게 원인(구버전 4h2/88% 정체와 동일 수준의 압축률
인데, 화음+조표까지 겹쳐 더 나빠짐). 추가로 **홀수 마디를 여러 줄로 나눌 때 `generate_scores.py`
(실제 렌더링 줄바꿈, 예: 3마디→2+1)와 `dataset.py`(균등분배 가정, 3마디→1+2)의 분배 방향이
정반대인 버그**도 발견.

**해결책 — 밀도 기반 줄바꿈으로 근본 재설계** (`--density-break` 플래그):
- 마디 개수가 아니라 **실제 내용 밀도**(음표/쉼표 이벤트=1, 화음 추가음=0.5, `MAX_SYSTEM_WEIGHT`
  기본 8.0)를 누적하다가 한계를 넘기 직전에 새 시스템 시작 — 실제 조판자가 내용 많으면 한
  줄에 마디를 적게 넣는 것과 같은 원리. 마디 수가 몇 개든, 얼마나 밀도가 다르든 항상 안전.
- 치/베이스 양쪽을 다 봐야 정확한 분배를 알 수 있어서 `_build_part()`가 더 이상 자체적으로
  줄바꿈을 결정하지 않고, `build_score_r3()`가 양쪽 마디 객체를 받은 뒤 `_decide_system_breaks()`로
  한 번에 결정.
- **실제 줄바꿈 지점을 JSON에 `system_breaks`로 저장**하고, `dataset.py`의
  `_split_grand_staff_interleaved()`가 이 정보가 있으면 정확히 그 경계로 분할(없으면 예전처럼
  균등분배로 fallback, 구버전 데이터 호환).
- **검증**: 실패했던 2~3마디+화음+조표 조합을 `--density-break`로 재테스트 → **66.8% → 93.2%**로
  완전히 회복. 이후 4gm(1~4마디, 6000장/40epoch)까지 확대해도 96.8%로 안전하게 통과.
- 이 설계 덕분에 "마디 2~3"과 "마디 2~4"를 별도 단계로 나눌 필요가 없어져서 한 단계(4gm)로 통합.

### 로컬 저장소 반영 (2026-07-20, 2차)
- `round3train/models/round1_curriculum_p2s4gm/seq2seq_best.pt` 다운로드 완료(최신 체크포인트)
- `secrets/checkpoints/seq2seq_p2s4g_best.pt` → `seq2seq_p2s4gm_best.pt`로 교체
- pod의 `round1_stage4gm_new` 학습 데이터는 30%(1800/6000장)만 남기고 정리(향후 replay용 최소 보존)

## 4t 이후: 박자/조표/다이나믹/기호 (4t~4sym, 2026-07-21) — 반복기호 제외 결정

4t(박자 6/8 도입, 다장조 고정) 95.37%로 게이트 통과. `error_breakdown.py`로 잔여 오류(66건) 분석
결과 barline 관련이 27건(41%)으로 압도적 1위, 오류 샘플 8개 중 6개가 `barline-start-repeat`/
`barline-end-repeat`(반복기호)를 일반 `barline`으로 오인식하는 동일 패턴이었음(체감상 오류의
~70%). `--repeat-prob`를 0.12→0.6으로 인위적으로 올려 노출 부족 가설을 격리 진단하려 했으나,
같은 4t 체크포인트에서 이어 학습했음에도 epoch1부터 Acc=0%/TER=100%대로 완전히 붕괴 —
단순 노출 부족이 아니라 데이터 생성/렌더링 쪽에 더 깊은 문제가 있는 것으로 판단.

**결정: 반복기호를 프로젝트 범위에서 제외.** `curriculum_4t_4sym.sh`의 `COMMON_ARGS`에서
`--repeat-prob`를 0.12 → 0으로 변경(4k/4dyn/4sym부터 반복기호 신규 생성 안 함). 진단용 데이터/
체크포인트(`round1_stage4t_repeat_diag_new`, `round1_curriculum_p2s4t_repeat_diag`)는 삭제.
tokenizer 자체는 그대로 두되(vocab 변경 없음, 재학습 불필요), 향후 생성 데이터에 해당 토큰이
더 이상 등장하지 않음.

## 다음 단계
1. **악보 기호 추가** (다음 2차 커리큘럼 단계) — 지금까지 전부 `easy` 난이도로 화음/조표/마디만
   다뤘고, 셈여림(dynamic)/아티큘레이션(artic)/슬러(slur)/셋잇단음표(tuplet)/페르마타(fermata)/
   크레센도·디크레센도(hairpin)/옥타브(ottava)/장식음(ornament)은 아직 전혀 학습 안 됨(전부
   확률 0으로 꺼둔 상태). `--dynamic-prob` 등 개별 확률 오버라이드 플래그는 이미 준비되어 있음.
   4gm에서 resume, 데이터/epoch는 4000/30 정도 권장(6000/40은 사전테스트 대비 다소 과했음).
2. **다중 시스템 재조립 파이프라인** — `inference.py`/C++ 엔진에 "페이지에서 여러 시스템(줄)
   감지 → 시스템별로 독립 인식 → 읽는 순서대로 토큰 이어붙이기" 오케스트레이션 로직 구현 필요
   (모델 재학습이 아니라 추론 파이프라인 코드 작업). 지금까지 학습은 전부 "시스템 하나 인식"
   단위로 진행됐고, 실제 사진 속 여러 줄짜리 페이지를 처리하려면 이 조립 단계가 필수.
3. TFLite export(`ml/omr/training/export_tflite.py`)로 최신 체크포인트를 실제 앱에 연결하는 작업
   착수 여부 검토 (CLAUDE.md "Known Gaps" 참고). `round3train/export_tflite.py`(이번 세션에
   round3train 전용으로 새로 작성, vocab 258/그랜드스태프 캔버스 기준)로 FP32 export까지는
   검증 완료 — INT8 양자화는 실제 대보표 캘리브레이션 이미지가 필요해서 보류 중.
