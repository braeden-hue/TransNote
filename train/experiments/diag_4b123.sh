#!/bin/bash
# Phase 0 진단: 4b_1(1/8만), 4b_2(1/16만), 4b_3(1/8+1/16만), 4b_key(4b 안전 duration+조표만 다양화)
# 4b_1~4b_3은 4a에서, 4b_key는 4b에서 resume. replay 없음, 대보표1/마디1 고정, 2000장/25epoch.
set -uo pipefail
LOG=/workspace/diag_4b123.log
STATUS=/workspace/diag_4b123_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Phase 0 진단 시작 (4b_1/4b_2/4b_3/4b_key) ==="
: > "$STATUS"

NAMES=(4b_1 4b_2 4b_3 4b_key)
START_IDX=(1700001 1750001 1800001 1850001)
SEEDS=(170001 175001 180001 185001)

CKPT_4A=/workspace/models/round1_curriculum_p2s4_quarteronly/seq2seq_best.pt
CKPT_4B=/workspace/models/round1_curriculum_p2s4b/seq2seq_best.pt

RESUME_CKPTS=("$CKPT_4A" "$CKPT_4A" "$CKPT_4A" "$CKPT_4B")
# 각 단계별 generate_scores.py 추가 인자 (배열 각 원소는 공백으로 구분된 하나의 인자 묶음)
GEN_EXTRA=(
  "--force-c-major --duration-subset 1/8"
  "--force-c-major --duration-subset 1/16"
  "--force-c-major --duration-subset 1/8,1/16"
  "--duration-subset 1/4,1/2,1/1"
)

for i in "${!NAMES[@]}"; do
  NAME=${NAMES[$i]}
  SIDX=${START_IDX[$i]}
  SEED=${SEEDS[$i]}
  CKPT=${RESUME_CKPTS[$i]}
  EXTRA=${GEN_EXTRA[$i]}
  NEW_DIR=/workspace/data/round1_stage${NAME}_new
  TRAIN_OUT=/workspace/models/round1_curriculum_p2s${NAME}

  echo ""
  echo "=== $(date) [$NAME] 시작 (gen_extra='${EXTRA}', 2000장, 25epoch, resume=${CKPT}, replay 없음) ==="

  dd if=/dev/zero of=/workspace/_qcheck bs=1M count=100 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "[$NAME] quota 위험 -- 중단"
    rm -f /workspace/_qcheck
    echo "DIAG_STOPPED_QUOTA:$NAME" >> "$STATUS"
    exit 1
  fi
  rm -f /workspace/_qcheck

  EXISTING_N=$(find "$NEW_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "$EXISTING_N" -eq 2000 ]; then
    echo "[$NAME] 데이터 이미 존재(${EXISTING_N}장) -- 재생성 스킵"
  else
    rm -rf "$NEW_DIR"
    bash /workspace/round3train/gen_render_local.sh "$NEW_DIR" 2000 \
      --start-idx "$SIDX" --difficulty easy --min-measures 1 --max-measures 1 \
      --seed "$SEED" $EXTRA
    if [ $? -ne 0 ]; then
      echo "[$NAME] 데이터 생성/검증 실패 -- 중단"
      echo "DIAG_STOPPED_GENFAIL:$NAME" >> "$STATUS"
      exit 1
    fi
  fi

  mkdir -p "$TRAIN_OUT"
  cd /workspace/round3train
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 -u train.py --phase 2 \
    --data_dir "$NEW_DIR" --out_dir "$TRAIN_OUT" \
    --resume "$CKPT" \
    --batch 24 --epochs 25 --workers 16 \
    --freeze_epochs 8
  TRAIN_RC=$?
  if [ $TRAIN_RC -ne 0 ]; then
    echo "[$NAME] 학습 비정상 종료(exit=$TRAIN_RC)"
    echo "DIAG_TRAINFAIL:$NAME" >> "$STATUS"
    continue
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
  echo "[$NAME] 학습 완료 -- best val_acc = ${BEST_ACC}%"
  echo "DIAG_RESULT:$NAME:$BEST_ACC" >> "$STATUS"
done

echo ""
echo "=== $(date) Phase 0 진단 전체 완료 ==="
echo "DIAG_ALL_DONE" >> "$STATUS"
