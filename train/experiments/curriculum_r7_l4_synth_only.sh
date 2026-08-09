#!/bin/bash
# round3train/curriculum_r7_l4_synth_only.sh
#
# Round7 재정의(2026-08-02) -- 실사 사진 없이 L4 위주 노이즈만 먼저 안정화. 이전 시도
# (curriculum_r7_l4_major_realphotos.sh)가 "노이즈 강화"와 "실사 도입"을 동시에 해서
# 불안정했던 것으로 판단, 이번 세션 내내 확인된 원칙(대보표 도입도 다른 축과 동시에
# 넣었을 때만 붕괴했음)에 따라 분리. 실사 사진은 Round8에서 별도로 도입.
#
# r6_l4_noise(L3위주+L4소량)에서 이어받아, 신규 합성(이미 생성된 r7_l4_major_synth 재사용,
# 합성 데이터 신규 생성은 안 함) + replay로 L3/L4 노이즈를 안정적으로 소화시킨다.

set -uo pipefail
LOG=/workspace/curriculum_r7b.log
STATUS=/workspace/curriculum_r7b_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 7(재정의: L4 위주, 실사 없음) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r6_l4_noise" /workspace/curriculum_r6_status.txt 2>/dev/null; then
  echo "[r7b] Round6이 STAGE_PASSED 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_R6_NOT_PASSED" >> "$STATUS"
  exit 1
fi

DATA_DIR=/workspace/data/r7_l4_major_synth   # 이미 생성 완료(1999장), 재사용
REPLAY_DIR=/workspace/data/replay_pool_v3    # 이전 라운드(1200)+직전 r6(800), 이미 구성됨
REPLAY_COUNT=2000

RESUME_CKPT=/workspace/models/r6_l4_noise/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r7b_l4_synth_only
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=20
FREEZE_EPOCHS=0
GATE=85

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$DATA_DIR" ] || [ ! -d "$REPLAY_DIR" ]; then
  echo "[$STAGE_NAME] 데이터/replay 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_DATA" >> "$STATUS"
  exit 1
fi

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, replay=${REPLAY_COUNT}, noise L3/L4 p=0.7) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --page_level_noise --noise_level 3 --noise_level_max 4 --p_level_max 0.7 \
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

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 7(재정의) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
