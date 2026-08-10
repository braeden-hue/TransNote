# 학습/검증 인계 상태 (2026-08-05 기준)

> 2026-08-09 저장소 재구성으로 로컬 `round3train/` → `train/`으로 폴더명이 바뀌었다. 아래
> 로컬 경로는 전부 새 이름으로 갱신함(내용/결론은 변경 없음, pod 원격 경로는 실제 원격 상태
> 그대로 `round3train/`로 남겨둠 — 재접속 시 리싱크 필요). 정확도 수치 근거는
> [`TRAINING_REPORT.md`](TRAINING_REPORT.md)에도 반영되어 있음.

## 결론부터: 프로덕션 체크포인트는 `r15_cropfix_coordconv`

오늘 시도한 r16(박자표 미노출)/r17(duration 분포 보강) 둘 다 실사 검증에서 r15를
못 넘었다. **r15를 계속 최선/데모용 체크포인트로 유지할 것.**

- 로컬: `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt`
- pod에는 오늘 세션 정리 과정에서 체크포인트 3개(r15/r16/r17) 전부 삭제함(로컬 백업
  확인 후 삭제) — pod에서 다시 쓰려면 로컬에서 재업로드 필요.

## pod 접속 정보 (오늘 세션 기준, 만료/재시작 시 갱신 필요)

- Host: `157.157.221.29`, Port: `24873`, User: `root`
- SSH 키: `~/.ssh/runpod_auto`
- 접속: `ssh -i ~/.ssh/runpod_auto -p 24873 root@157.157.221.29`
- 볼륨 id `8r58mbi66s`(과거 세션에서 쓰던 S3 API 접근용 old volume)는 **사용자가
  2026-08-05 직접 삭제함** — 더 이상 존재하지 않음, S3 API 키도 자연히 무효.

## 오늘 있었던 일 (요약)

1. **새 pod가 `/workspace` 완전히 빈 상태로 시작** — 이전 세션 산출물(r12_all120_realphotos,
   r12_replay_merged, r15 체크포인트 등)이 전부 없어진 것처럼 보임.
2. 실사 사진 원본(`data/local_pools/exactPicture`)과 r15 체크포인트는 로컬에 있어서 재구성 가능했음.
   `r12_replay_merged`(8486장, 순수 합성)는 로컬에도 없어서 "동일 파라미터로 재생성"
   (`gen_replay_pool_reconstruction.sh`)으로 근사 대체본을 만듦.
3. 이 근사 replay로 r16/r17을 처음 학습 → **둘 다 r15보다 하락**(84.2%→82.7%/81.3%, 6곡 기준).
4. **알고 보니 이전 볼륨이 유실된 게 아니라 S3 API로 접근 가능한 상태로 그대로 살아있었음**
   (`https://s3api-eur-is-1.runpod.io`, 볼륨 `8r58mbi66s`). 원본 `r12_replay_merged`(10.72GB)를
   그대로 복구해서 r16/r17을 **원본 replay로 재학습** → 이번엔 격차가 줄었지만(87.2%→86.1%/85.5%,
   12곡 기준) **여전히 r15 못 넘음**. replay 진위가 유일한 원인은 아니었다는 뜻.
5. 검증용 신규곡을 6곡(sonatine_22_30/23_38/23_42/32_38/36_60/81_92) → **12곡**으로 확장,
   newage21~26 추가(`.mscz`에서 GT 생성, `mscz_to_tokens.py`).
   - **newage23에 실제 GT 버그 있음**: 2번째 마디 베이스에 `note-A1`이 들어있는데 이 토큰이
     `tokenizer258.json` 어휘에 없음(`'note-A1' in vocab` → `False`) — 이 마디는 구조적으로
     100% 인식 불가능. 아직 미수정(악보에서 옥타브 올리거나 검증에서 제외 필요).
6. **사후처리(재학습 불필요) 실험**: 치/베이스 마디 박자 합 대조로 음표 과잉/누락 마디를
   찾아내는 진단(`diag_beatsum_flag.py`)은 precision 94.7%/recall 37.5%로 유망했으나,
   실제 교정(제약 디코딩으로 그 마디만 재디코딩, `inference.py`의
   `correct_treble_note_counts`)을 붙여보니 **오히려 87.2%→85.7%로 하락** — 진단
   정확도와 교정 품질은 별개 문제였음. `run_image()` 호출은 되돌렸고 함수만 코드에 남김
   (재학습 없이 재시도 가능하니 참고용).
7. 볼륨 삭제 사건으로 S3 API 접근이 막힘(`AccessDenied`) → 사용자가 이미 볼륨 자체를
   지운 것으로 확인, 정상.
8. pod 스토리지 정리: 체크포인트 3개(로컬 백업 확인 후) 삭제, r16/r17용 replay 병합본
   (각 12GB, r12_replay_merged 중복 포함) 삭제, **`r12_replay_merged`(10GB, 8486장)는 보존**.

## 로컬에 있는 것 (재현/재개에 필요)

- `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt` — 프로덕션 체크포인트
- `train/checkpoints/r16_hide_timesig_v2/seq2seq_best.pt`,
  `train/checkpoints/r17_dense_rhythm_v2/seq2seq_best.pt` — 실패한 시도(참고용, 원본 replay 버전)
- `train/data/local_pools/exactPicture/` — 실사 원본(126곡, newage21~26 포함)
- `train/data/local_pools/exactpicture_test_full/` — GT(126곡분, 4마디 트리밍)
- `train/diag_new6_errors.py` — 12곡 오류 세분화 진단(`--ckpt`/`--device`/`--songs` 옵션)
- `train/diag_beatsum_flag.py` — 마디 박자 합 불일치 진단(1단계, 교정 없음)
- `train/gen_replay_pool_reconstruction.sh`, `gen_r16_hide_timesig.sh`,
  `gen_r17_dense_rhythm.sh`, `curriculum_r16_hide_timesig.sh`, `curriculum_r17_dense_rhythm.sh`
  — 오늘 작성한 신규 스크립트(전부 원본 `r12_replay_merged` 경로 기준으로 수정됨)

## pod에 남아있는 것 (재접속 시)

- `/workspace/data/r12_replay_merged/`(10GB, 8486장) — **유일하게 로컬에 없는 자산**,
  다음에 pod 쓸 일 있으면 여기부터 확인
- `/workspace/round3train/data/local_pools/`(3.3GB) — 실사 사진+GT (pod 원격 경로, 로컬 이름과 다름)
- `/workspace/musescore/`(1.1GB) — MuseScore 4.7.4 설치본(버전 고정 안 됨, 매번 "최신" 릴리스 받아옴)
- `/workspace/models/`는 비어있음(체크포인트 전부 삭제됨)

## 다음에 시도할 만한 것 (우선순위 순)

1. **newage23 GT 버그 수정**(A1 옥타브 조정) — 가장 저렴, 검증 정확도 왜곡 제거
2. **3도 오독(단/장3도 음이름 혼동)** — 모든 검증(신규 6곡/12곡, 장르 불문)에서 최다
   오류 카테고리로 반복 확인됐지만 아직 근본 원인 조사(`diagnose_third_confusion.py`)
   이후 실제 수정 시도는 없었음
- r16/r17을 다시 시도한다면: "왜 원본 replay로도 여전히 r15보다 낮은가"부터 먼저
  규명 필요 (LR 재가열/과적합/새 축 자체의 부작용인지 등, 오늘 세션에서 결론 못 냄)
