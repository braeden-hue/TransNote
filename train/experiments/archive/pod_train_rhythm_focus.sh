#!/bin/bash
# 2단계(밀집 리듬 집중) 학습: 1단계(화음 집중) 체크포인트에서 이어서, 8분/16분 런 밀집
# 구간의 음표 과잉/누락 약점 보강. 밀집 리듬 집중 데이터 1500장 + replay 1000장
# (step1_pool).
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/rhythm_focus_train.log
STATUS=/workspace/rhythm_focus_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 2단계(밀집 리듬 집중) 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/rhythm_focus_pool
REPLAY_DIR=/workspace/round3train/data/step1_pool
REPLAY_COUNT=1000
OUT_DIR=/workspace/models/rhythm_focus
RESUME=/workspace/models/chord_focus/seq2seq_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=10
FREEZE_EPOCHS=0

mkdir -p "$OUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --resume "$RESUME" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --batch 24 --epochs "$EPOCHS" --workers 8 \
  --freeze_epochs "$FREEZE_EPOCHS" --no_augment
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[rhythm_focus] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:rhythm_focus" >> "$STATUS"
echo "=== $(date) 2단계(밀집 리듬 집중) 학습 종료 ==="
