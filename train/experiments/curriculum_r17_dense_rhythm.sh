#!/bin/bash
# round3train/curriculum_r17_dense_rhythm.sh
#
# duration(음가) 예측 축 보강 라운드(2026-08-04). r15_cropfix_coordconv 신규6곡 재실측에서
# dur-1/16 missing/extra가 duration 오류 중 압도적 1위로 확인됐고, 실사 GT 126곡 실측 결과
# 합성 데이터의 기본 duration 분포(1/4 위주)가 실제 촬영 대상(1/8 위주)과 크게 어긋나 있었음.
# gen_r17_dense_rhythm.sh로 실사 실측 분포에 맞춰 재보정한 합성 2000장 생성.
#
# r11_rhythm_focus(같은 축을 좁은 합성 MAIN 데이터로 시도했다가 -9~13pp 실패)의 전례에 따라,
# 이번엔 r12~r15가 검증한 "실사 120곡 MAIN 유지 + 합성은 replay 축으로만 추가" 패턴을 그대로
# 따름 -- MAIN 데이터(r12_all120_realphotos)는 손대지 않고, r17 신규 배치를 replay 풀에 병합해서만 사용.
#
# 2026-08-04 정정: 1차 시도에서 replay_pool_reconstructed(파라미터 근사 재구성)로 학습했다가
# r16/r17 둘 다 신규6곡 정확도가 r15보다 하락(84.2%->82.7%/81.3%)하는 걸 확인 -- 원인으로
# "재구성 replay가 원본과 다른 콘텐츠였던 것"이 유력해서, S3(8r58mbi66s)에서 원본
# r12_replay_merged(8486장, 실물 그대로)를 복구해 이걸로 재학습함. 원인 검증을 위한 재시도.
# r16_hide_timesig과 마찬가지로 r15에서 독립 분기(병렬 라운드).
#
# 사전 조건: gen_r17_dense_rhythm.sh가 GEN_COMPLETE로 끝나있어야 함,
# /workspace/data/r12_replay_merged(S3에서 복구한 원본)가 존재해야 함.

set -uo pipefail
LOG=/workspace/curriculum_r17_dense_rhythm.log
STATUS=/workspace/curriculum_r17_dense_rhythm_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 17(duration 분포 보강) 시작 ==="
: > "$STATUS"

if ! grep -q "^GEN_COMPLETE" /workspace/gen_r17_dense_rhythm_status.txt 2>/dev/null; then
  echo "[r17_dense_rhythm] duration 보강 데이터 생성이 GEN_COMPLETE 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_GEN_NOT_COMPLETE" >> "$STATUS"
  exit 1
fi
if [ ! -d /workspace/data/r12_replay_merged ]; then
  echo "[r17_dense_rhythm] 원본 replay 풀(r12_replay_merged) 없음 -- 중단"
  echo "PIPELINE_STOPPED_REPLAY_NOT_COMPLETE" >> "$STATUS"
  exit 1
fi

REAL_POOL_DIR=/workspace/round3train/data/local_pools/r12_all120_realphotos   # r12~r15와 동일, 손대지 않음
BASE_REPLAY_DIR=/workspace/data/r12_replay_merged                            # 원본(8486장, S3에서 복구)
NEW_POOL_DIR=/workspace/round3train/data/r17_dense_rhythm_pool               # 이번 라운드 신규(목표 2000장)
REPLAY_DIR=/workspace/data/r17_replay_merged                                 # 병합 결과(신규 생성)

if [ ! -d "$REAL_POOL_DIR" ] || [ ! -d "$BASE_REPLAY_DIR" ] || [ ! -d "$NEW_POOL_DIR" ]; then
  echo "[r17_dense_rhythm] 데이터/replay 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_DATA" >> "$STATUS"
  exit 1
fi

# 병합은 최초 1회만(이미 병합돼 있으면 재실행해도 건너뜀 -- 카운트로 판단)
BASE_N=$(find "$BASE_REPLAY_DIR" -maxdepth 1 -name '*.png' | wc -l)
NEW_N=$(find "$NEW_POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
EXPECT_MERGED_N=$(( BASE_N + NEW_N ))
MERGED_N=$(find "$REPLAY_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
if [ "$MERGED_N" -lt "$EXPECT_MERGED_N" ]; then
  echo "[r17_dense_rhythm] replay 병합 시작: $BASE_REPLAY_DIR($BASE_N장) + $NEW_POOL_DIR($NEW_N장) -> $REPLAY_DIR"
  mkdir -p "$REPLAY_DIR"
  cp -n "$BASE_REPLAY_DIR"/*.png "$BASE_REPLAY_DIR"/*.json "$REPLAY_DIR"/
  cp -n "$NEW_POOL_DIR"/*.png "$NEW_POOL_DIR"/*.json "$REPLAY_DIR"/
  MERGED_N=$(find "$REPLAY_DIR" -maxdepth 1 -name '*.png' | wc -l)
fi
echo "[r17_dense_rhythm] replay 병합 완료: ${MERGED_N}장(기대 ${EXPECT_MERGED_N}장)"
REPLAY_COUNT=$MERGED_N

RESUME_CKPT=/workspace/models/r15_cropfix_coordconv/seq2seq_best.pt   # 현재 최선 체크포인트에서 재개
IN_CH=2   # r15와 동일(CoordConv 유지, 구조 변경 없음)
EXTRA_HEIGHT_STAGES=4
POOL_H=1
STAGE_NAME=r17_dense_rhythm
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15
FREEZE_EPOCHS=0
GATE=70

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi

REAL_N=$(find "$REAL_POOL_DIR" -maxdepth 1 \( -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, in_ch=${IN_CH}, 메인=실사 ${REAL_N}장, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$REAL_POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --page_level_noise --noise_level 3 --noise_level_max 4 --p_level_max 0.5 \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS"
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[$STAGE_NAME] 학습 프로세스 비정상 종료(exit=$TRAIN_RC) -- 파이프라인 중단"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi

BEST_ACC=$(python3 -c "
import csv
best = 0.0
with open('$TRAIN_OUT/seq2seq_phase2_log.csv') as f:
    for row in csv.DictReader(f):
        a = float(row['val_acc'])
        if a > best:
            best = a
print(f'{best:.2f}')
")
echo "[$STAGE_NAME] 학습 완료 -- best val_acc(teacher-forcing) = ${BEST_ACC}% (gate=${GATE}%)"
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- 신규 6곡(diag_new6_errors.py)으로 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 17(duration 분포 보강) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
