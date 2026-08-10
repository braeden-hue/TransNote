#!/bin/bash
# Step1 재학습(replay 포함): Round3 체크포인트에서 다시 resume(Step1 최종 체크포인트가
# 아니라 원본 Round3 베이스에서 시작 -- Step1은 이미 실측 기준 퇴보가 확인돼, 그 위에
# replay를 얹으면 이미 손상된 지점에서 시작하는 셈이라 깨끗한 Round3 베이스에서 재시작).
# 2026-07-31: Step1(replay 없음) 결과가 TF Acc 98.29%로 좋아 보였지만 exactPicture 89곡
# 실측(82.6%->79.6%)과 신규 합성 10곡 실측(85.3%->81.3%) 둘 다 Round3 대비 퇴보 확인 --
# catastrophic forgetting 의심. Round3 학습 데이터(r3_density_register_clef, 15000장)를
# replay_dir로 섞어 재시도.
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/step1_replay_train.log
STATUS=/workspace/step1_replay_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) Step1(replay) 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/step1_pool
REPLAY_DIR=/workspace/round3train/data/local_pools/r3_density_register_clef
REPLAY_COUNT=2500
OUT_DIR=/workspace/models/step1_replay
RESUME=/workspace/checkpoints/seq2seq_r3_density_register_clef_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=30
FREEZE_EPOCHS=8
GATE=90

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
  echo "[step1_replay] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi

BEST_ACC=$(python3 -c "
import csv
best = 0.0
with open('$OUT_DIR/seq2seq_phase2_log.csv') as f:
    for row in csv.DictReader(f):
        a = float(row['val_acc'])
        if a > best:
            best = a
print(f'{best:.2f}')
")
echo "[step1_replay] 학습 완료 -- best val_acc(teacher-forcing) = ${BEST_ACC}% (참고용, gate=${GATE}%)"
echo "[step1_replay] 주의: TF 기준이라 신뢰 불가 -- 실측(exactPicture 등)으로 재검증할 것"
echo "STAGE_DONE:step1_replay:${BEST_ACC}" >> "$STATUS"
echo "=== $(date) Step1(replay) 학습 종료 ==="
