#!/bin/bash
# markov-bias 0.3 데이터 재학습: 2단계(화음+리듬) 체크포인트에서 이어서, 새 4000장
# (markov-bias 0.3) + replay 1000장(step1_pool).
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/lowmarkov_train.log
STATUS=/workspace/lowmarkov_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) markov-bias 0.3 재학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/lowmarkov_pool
REPLAY_DIR=/workspace/round3train/data/step1_pool
REPLAY_COUNT=1000
OUT_DIR=/workspace/models/lowmarkov
RESUME=/workspace/models/rhythm_focus/seq2seq_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=20
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
  echo "[lowmarkov] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:lowmarkov" >> "$STATUS"
echo "=== $(date) markov-bias 0.3 재학습 종료 ==="
