#!/bin/bash
# 4t 잔여 오류 원인 진단: barline-start-repeat/barline-end-repeat 오인식이 학습 노출 부족
# 때문인지(격리 시 개선) 아니면 렌더링/모델 구조 문제인지(격리해도 안 나아짐) 확인.
# 4t와 동일 설정에서 --repeat-prob만 0.12->0.6으로 인위적으로 올린 소량 데이터로
# 4t 체크포인트에서 짧게 이어 학습 후, 4t 때와 같은 방식(error_breakdown.py)으로 재평가.
set -uo pipefail
LOG=/workspace/diag_4t_repeat.log
STATUS=/workspace/diag_4t_repeat_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 4t 반복기호 격리 진단 시작 ==="
: > "$STATUS"

NAME=4t_repeat_diag
COUNT=1500
EPOCHS=15
FREEZE_EPOCHS=3
SIDX=3900001
SEED=390001

CKPT=/workspace/models/round1_curriculum_p2s4t/seq2seq_best.pt
REPLAY_DIR=/workspace/data/round1_stage4t_new
NEW_DIR=/workspace/data/round1_stage${NAME}_new
TRAIN_OUT=/workspace/models/round1_curriculum_p2s${NAME}

GEN_ARGS=(--start-idx "$SIDX" --seed "$SEED"
          --min-measures 1 --max-measures 4 --density-break --chord-prob 0.15
          --repeat-prob 0.6 --artic-prob 0 --ornament-prob 0 --slur-prob 0
          --tuplet-prob 0 --ottava-prob 0 --force-c-major --dynamic-prob 0
          --hairpin-prob 0 --fermata-prob 0)

echo "[$NAME] quota 확인 중..."
if ! dd if=/dev/zero of=/workspace/_qcheck_diag bs=1M count=100 2>/dev/null; then
  echo "[$NAME] quota 초과 위험 -- 중단"
  rm -f /workspace/_qcheck_diag
  echo "DIAG_STOPPED_QUOTA:$NAME" >> "$STATUS"
  exit 1
fi
rm -f /workspace/_qcheck_diag

EXISTING_N=$(find "$NEW_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
if [ "$EXISTING_N" -eq "$COUNT" ]; then
  echo "[$NAME] 데이터 이미 존재(${EXISTING_N}장) -- 재생성 스킵"
else
  rm -rf "$NEW_DIR"
  echo "[$NAME] 데이터 생성: ${GEN_ARGS[*]}"
  bash /workspace/round3train/gen_render_local.sh "$NEW_DIR" "$COUNT" "${GEN_ARGS[@]}"
  if [ $? -ne 0 ]; then
    echo "[$NAME] 데이터 생성/검증 실패 -- 중단"
    echo "DIAG_STOPPED_GENFAIL:$NAME" >> "$STATUS"
    exit 1
  fi
fi

REPLAY_COUNT=$(find "$REPLAY_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
echo "[$NAME] replay: ${REPLAY_COUNT}장 (${REPLAY_DIR}, 복사 없음)"

mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$NEW_DIR" --out_dir "$TRAIN_OUT" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --resume "$CKPT" \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS" --tf_ratio 1.0 --min_tf_ratio 1.0
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[$NAME] 학습 비정상 종료(exit=$TRAIN_RC) -- 중단"
  echo "DIAG_TRAINFAIL:$NAME" >> "$STATUS"
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
echo "[$NAME] 학습 완료 -- best val_acc = ${BEST_ACC}%"
echo "DIAG_RESULT:$NAME:$BEST_ACC" >> "$STATUS"

echo "[$NAME] 오류 분석 (repeat-prob 높인 held-out 데이터 기준)..."
python3 /workspace/round3train/error_breakdown.py \
  --seq2seq "$TRAIN_OUT/seq2seq_best.pt" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --data_dir "$NEW_DIR" \
  > "$TRAIN_OUT/error_breakdown.log" 2>&1
echo "[$NAME] 오류 분석 완료: $TRAIN_OUT/error_breakdown.log"

echo ""
echo "=== $(date) 4t 반복기호 격리 진단 완료 ==="
echo "DIAG_ALL_DONE" >> "$STATUS"
