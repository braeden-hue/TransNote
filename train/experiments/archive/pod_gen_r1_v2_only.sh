#!/bin/bash
# 데이터 생성만 먼저(CPU/xvfb-run 전용, GPU 학습과 병렬 안전) -- curriculum_r1_v2_foundation.sh와
# 완전히 동일한 POOL_DIR/COUNT/파라미터를 써서, 나중에 그 스크립트를 실행하면 EXISTING_N 체크로
# 재생성 없이 바로 학습 단계로 넘어가도록 함. combined_coordconv GPU 학습이 끝나기 전에
# Round1 학습까지 같이 돌리면 VRAM 경합(OOM) 위험이 있어 학습 단계는 분리한다.
set -uo pipefail
LOG=/workspace/gen_r1_v2_only.log
exec >> "$LOG" 2>&1
echo "=== $(date) Round1 v2 데이터 생성만 시작 ==="

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
POOL_DIR=/workspace/data/r1_v2_foundation
COUNT=3000
START_IDX=5000001
SEED=500001
N_SHARDS=6

COMMON_ARGS=(--single-staff --min-measures 1 --max-measures 4 --chord-prob 0.22
             --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
             --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
             --preferred-register-prob 0.3
             --markov-bias 0.3 --markov-table /workspace/round3train/markov_transitions.json)

rm -rf "$POOL_DIR"
mkdir -p "$POOL_DIR"
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
    > "/workspace/gen_shard_r1_v2_${i}.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
if [ "$FAIL" -ne 0 ]; then
  echo "일부 샤드 실패 -- ${FINAL_N}/${COUNT}장만 생성됨. /workspace/gen_shard_r1_v2_*.log 확인"
  exit 1
fi
echo "=== $(date) Round1 v2 데이터 생성 완료: ${FINAL_N}/${COUNT}장 ==="
