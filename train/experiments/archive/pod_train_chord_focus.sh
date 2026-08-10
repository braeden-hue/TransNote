#!/bin/bash
# 1단계(화음 집중) 학습: 극단음역 재학습 체크포인트에서 이어서, 화음 개수 세기 약점
# (Step1 100장 재검증: extra/missing note+dur가 오류의 58%) 보강. 화음 집중 데이터
# 1520장 + replay 1000장(step1_pool).
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/chord_focus_train.log
STATUS=/workspace/chord_focus_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 1단계(화음 집중) 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/chord_focus_pool
REPLAY_DIR=/workspace/round3train/data/step1_pool
REPLAY_COUNT=1000
OUT_DIR=/workspace/models/chord_focus
RESUME=/workspace/models/extreme_register/seq2seq_best.pt
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
  echo "[chord_focus] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:chord_focus" >> "$STATUS"
echo "=== $(date) 1단계(화음 집중) 학습 종료 ==="
