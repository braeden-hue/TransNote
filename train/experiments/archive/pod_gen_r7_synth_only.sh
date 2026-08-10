#!/bin/bash
# Round7의 신규 합성 데이터만 미리 생성(GPU 학습 중인 Round6와 병렬, CPU/xvfb 전용이라
# 안전). curriculum_r7_l4_major_realphotos.sh와 완전히 동일한 POOL_DIR/COUNT/파라미터를
# 써서, 나중에 그 스크립트가 실행되면 EXISTING_N 체크로 재생성 없이 바로 다음 단계로 감.
set -uo pipefail
LOG=/workspace/gen_r7_synth_only.log
exec >> "$LOG" 2>&1
echo "=== $(date) Round7 신규 합성 데이터 사전 생성 시작 ==="

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
SYN_POOL_DIR=/workspace/data/r7_l4_major_synth
SINGLE_COUNT=300
GRAND_COUNT=1700
START_IDX_SINGLE=11000001
START_IDX_GRAND=11500001
SEED_SINGLE=1100001
SEED_GRAND=1150001
N_SHARDS=6

BASE_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.22
           --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
           --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
           --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
           --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
           --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
           --preferred-register-prob 0.3
           --markov-bias 0.3 --markov-table /workspace/round3train/markov_transitions.json)
GRAND_ONLY_ARGS=(--cross-register-prob 0.20)

rm -rf "$SYN_POOL_DIR"
mkdir -p "$SYN_POOL_DIR"

echo "단일오선(${SINGLE_COUNT}장) 생성 중..."
SHARD_BASE=$(( SINGLE_COUNT / N_SHARDS ))
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
  SHARD_START=$(( START_IDX_SINGLE + i * 10000 ))
  SHARD_SEED=$(( SEED_SINGLE + i ))
  THIS_COUNT=$SHARD_BASE
  if [ "$i" -eq $((N_SHARDS - 1)) ]; then
    THIS_COUNT=$(( SINGLE_COUNT - SHARD_BASE * (N_SHARDS - 1) ))
  fi
  xvfb-run -a python3 /workspace/round3train/generate_scores.py \
    --count "$THIS_COUNT" --output "$SYN_POOL_DIR" --musescore "$MUSESCORE" \
    --single-staff --start-idx "$SHARD_START" --seed "$SHARD_SEED" \
    "${BASE_ARGS[@]}" > "/workspace/gen_shard_r7pre_single_${i}.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then
  echo "단일오선 생성 일부 실패 -- 로그 확인 필요"
fi

echo "대보표(${GRAND_COUNT}장) 생성 중..."
SHARD_BASE=$(( GRAND_COUNT / N_SHARDS ))
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
  SHARD_START=$(( START_IDX_GRAND + i * 10000 ))
  SHARD_SEED=$(( SEED_GRAND + i ))
  THIS_COUNT=$SHARD_BASE
  if [ "$i" -eq $((N_SHARDS - 1)) ]; then
    THIS_COUNT=$(( GRAND_COUNT - SHARD_BASE * (N_SHARDS - 1) ))
  fi
  xvfb-run -a python3 /workspace/round3train/generate_scores.py \
    --count "$THIS_COUNT" --output "$SYN_POOL_DIR" --musescore "$MUSESCORE" \
    --start-idx "$SHARD_START" --seed "$SHARD_SEED" \
    "${BASE_ARGS[@]}" "${GRAND_ONLY_ARGS[@]}" > "/workspace/gen_shard_r7pre_grand_${i}.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then
  echo "대보표 생성 일부 실패 -- 로그 확인 필요"
fi

FINAL_N=$(find "$SYN_POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "=== $(date) Round7 신규 합성 데이터 생성 완료: ${FINAL_N}/$((SINGLE_COUNT+GRAND_COUNT))장 ==="
