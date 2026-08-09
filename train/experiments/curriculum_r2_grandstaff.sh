#!/bin/bash
# round3train/curriculum_r2_grandstaff.sh
#
# Round 1(단일오선 기초) 위에 대보표(치+베이스 동시 생성)를 별도 라운드로 추가한다.
# PODPLAN.md 실측 기록: 대보표를 다른 축과 동시에 처음부터 넣었던 원래 Stage4는
# 97.9%->79.5%로 붕괴했고(시퀀스 길이 1.7배, teacher forcing exposure bias), 복구하려면
# 데이터 2배(3000->6000)+freeze_epochs 3배(5->12)+epoch 1.6배(25->40)가 필요했음 --
# 이번에도 그 검증된 레시피를 그대로 스케일업해서 씀(8000장 기초 기준으로 비례 확대).
#
# replay(단일오선 30%)는 Round1 풀을 복사 없이 --replay_dir/--replay_count로 직접 재사용.
#
# 사전 조건: curriculum_r1_foundation.sh가 STAGE_PASSED로 끝나있어야 함
# (/workspace/curriculum_r1_status.txt 확인). 통과 못 했으면 원인 먼저 파악.

set -uo pipefail
LOG=/workspace/curriculum_r2.log
STATUS=/workspace/curriculum_r2_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 2(대보표 추가) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r1_foundation" /workspace/curriculum_r1_status.txt 2>/dev/null; then
  echo "[r2_grandstaff] Round1이 STAGE_PASSED 상태가 아님 -- 중단(먼저 확인 필요)"
  echo "PIPELINE_STOPPED_R1_NOT_PASSED" >> "$STATUS"
  exit 1
fi

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
POOL_DIR=/workspace/data/r2_grandstaff
R1_POOL_DIR=/workspace/data/r1_foundation
COUNT=5000
REPLAY_COUNT=1500
START_IDX=2000001
SEED=200001
N_SHARDS=6

RESUME_OUT=/workspace/models/r1_foundation
STAGE_NAME=r2_grandstaff
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=25
FREEZE_EPOCHS=10
GATE=90
MARKOV_TABLE=/workspace/round3train/markov_transitions.json
MARKOV_BIAS=0.5

COMMON_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.08
             --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0
             --cross-register-prob 0.15 --tie-prob 0.15
             --dotted8-bias 10.0 --short-note-bias 2.0 --preferred-register-prob 0.3
             --markov-bias "$MARKOV_BIAS" --markov-table "$MARKOV_TABLE")
             # 2026-07-30 추가 -- local_pools(6reg1/tie1/7den1)에서 이미 검증된 축들을
             # 누적 반영(carry-forward, 원래 각 축의 마지막 단계 값 그대로): 교차음역 편향
             # 복구, 붙임줄 연쇄붕괴 복구, 리듬밀도 화음환각 복구. 이걸 빠뜨리면 예전에
             # 고친 문제를 새 체인에서 재현할 위험이 있음(사용자 지적으로 발견).

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
  for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
  done
  if [ "$FAIL" -ne 0 ]; then
    echo "[$STAGE_NAME] 일부 샤드 생성 실패 -- /workspace/gen_shard_${STAGE_NAME}_*.log 확인 필요"
    echo "PIPELINE_STOPPED_GENFAIL" >> "$STATUS"
    exit 1
  fi
  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[$STAGE_NAME] 데이터 풀 완성: ${FINAL_N}/${COUNT}장"
fi

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=r1_foundation, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_OUT/seq2seq_best.pt" \
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
echo "[$STAGE_NAME] 주의: 이 숫자는 TF 기준이라 신뢰 불가 -- 반드시 eval_page_noise.py/eval_token_acc.py로 실측 재검증할 것"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) Round 2 완료 -- 실사 촬영 전 기초 학습 종료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
