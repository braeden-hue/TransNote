#!/bin/bash
# round3train/curriculum_r2_v2_grandstaff.sh
#
# Round1 v2(단일오선 재학습) 위에 대보표를 추가. 구조는 기존 curriculum_r2_grandstaff.sh와
# 동일(대보표 단독 도입이 가장 위험한 구조적 도약이라 다른 축과 동시에 넣지 않음), 파라미터만
# 2026-08-01 계획 기준 보정값으로 교체.
#
# 사전 조건: curriculum_r1_v2_foundation.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r2_v2.log
STATUS=/workspace/curriculum_r2_v2_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 2 v2(대보표 재학습, 보정 파라미터) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r1_v2_foundation" /workspace/curriculum_r1_v2_status.txt 2>/dev/null; then
  echo "[r2_v2] Round1 v2가 STAGE_PASSED 상태가 아님 -- 중단(먼저 확인 필요)"
  echo "PIPELINE_STOPPED_R1_NOT_PASSED" >> "$STATUS"
  exit 1
fi

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
POOL_DIR=/workspace/data/r2_v2_grandstaff
R1_POOL_DIR=/workspace/data/r1_v2_foundation
COUNT=5000
REPLAY_COUNT=1500
START_IDX=6000001
SEED=600001
N_SHARDS=6

RESUME_CKPT=/workspace/models/r1_v2_foundation/seq2seq_best.pt
IN_CH=${IN_CH:-2}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r2_v2_grandstaff
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=12
FREEZE_EPOCHS=0
GATE=90
MARKOV_TABLE=/workspace/round3train/markov_transitions.json
MARKOV_BIAS=0.3

COMMON_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.22
             --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0
             --cross-register-prob 0.20 --tie-prob 0.25
             --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
             --preferred-register-prob 0.3
             --markov-bias "$MARKOV_BIAS" --markov-table "$MARKOV_TABLE")
             # cross-register-prob 0.15->0.20 상향(2026-08-01) -- 클렙 레지스터 편향이
             # 이미 알려진 취약점이라 v1보다 처음부터 더 강하게 노출.

echo "[$STAGE_NAME] 데이터 풀 확인 중..."
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
MIN_OK_N=$(( COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[$STAGE_NAME] 데이터 풀 이미 존재(${EXISTING_N}/${COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"
  mkdir -p "$POOL_DIR"
  echo "[$STAGE_NAME] 대보표 데이터 생성 중 (${COUNT}장, ${N_SHARDS}개 병렬, markov-bias=${MARKOV_BIAS})..."
  SHARD_BASE=$(( COUNT / N_SHARDS ))
  PIDS=()
  for i in $(seq 0 $((N_SHARDS - 1))); do
    SHARD_START=$(( START_IDX + i * 100000 ))
    SHARD_SEED=$(( SEED + i ))
    THIS_COUNT=$SHARD_BASE
    if [ "$i" -eq $((N_SHARDS - 1)) ]; then
      THIS_COUNT=$(( COUNT - SHARD_BASE * (N_SHARDS - 1) ))
    fi
    xvfb-run -a python3 /workspace/round3train/generate_scores.py \
      --count "$THIS_COUNT" --output "$POOL_DIR" --musescore "$MUSESCORE" \
      --start-idx "$SHARD_START" --seed "$SHARD_SEED" "${COMMON_ARGS[@]}" \
      > "/workspace/gen_shard_${STAGE_NAME}_${i}.log" 2>&1 &
    PIDS+=($!)
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
  if [ "$FAIL" -ne 0 ]; then
    echo "[$STAGE_NAME] 일부 샤드 생성 실패 -- /workspace/gen_shard_${STAGE_NAME}_*.log 확인 필요"
    echo "PIPELINE_STOPPED_GENFAIL" >> "$STATUS"
    exit 1
  fi
  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[$STAGE_NAME] 데이터 풀 완성: ${FINAL_N}/${COUNT}장"
fi
echo "STAGE_DATA_DONE:$STAGE_NAME" >> "$STATUS"

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$R1_POOL_DIR" --replay_count "$REPLAY_COUNT" \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS" --no_augment
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- register_accuracy_r89.py/diagnose_third_confusion.py로 실측 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) Round 2 v2 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
