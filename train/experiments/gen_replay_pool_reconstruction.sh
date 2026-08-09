#!/bin/bash
# round3train/gen_replay_pool_reconstruction.sh
#
# r12_replay_merged(8486장)를 대체하는 replay 풀 재구성(2026-08-04). 이전 pod/볼륨이
# 사라지면서 r12_replay_merged 자체(및 그 구성요소인 replay_pool_v2/v3)의 원본 파일이
# 전부 유실됨을 확인 -- replay_pool_v2/v3는 r1_v2/r2_v2/r3_v2/r5 산출물에서 "수동으로
# 골라 복사"해서 만든 것이라, 그 선별 기준 자체가 이미 그때(r5 시점)도 재현 불가능한
# 상태였음(각 스크립트 주석에 "원본 풀은 이미 삭제됨"이라고 명시돼 있었음).
#
# 사용자 확인(2026-08-04): "동일 파라미터로 새로 생성" 방식으로 진행 -- 정확히 같은
# 악보(바이트 단위)는 아니지만, replay의 실제 기능(이전 라운드들이 다뤘던 콘텐츠 분포에
# 대한 노출을 유지해서 파인튜닝 중 망각을 막는 것)은 "같은 생성 파라미터로 만든 콘텐츠"로도
# 동일하게 달성됨. 4개 구성 요소:
#   1) general_diversity(2500장) -- r3_v2_mixed.sh의 BASE_ARGS(가장 성숙한 "일반" 다양성
#      파라미터, 단일:대보표=15:85) 재사용. replay_pool_v2를 대체.
#   2) rare_durations(800장) -- r5_rare_durations.sh의 BASE_ARGS(dotted8-bias 12,
#      rare-long-bias 3.0) 재사용, 단일:대보표=15:85. replay_pool_v3에 섞였던 r5분(800장) 대체.
#   3) l4_major_synth(2000장) -- pod_gen_r7_synth_only.sh와 완전히 동일한 파라미터+시드로
#      생성 -- 이건 원본 생성 스크립트 자체가 결정론적(--seed 고정)이라 근사가 아니라 사실상
#      동일 재현.
#   4) chord_accidental_synth(2000장) -- curriculum_r9_chord_accidental.sh와 완전히 동일한
#      파라미터+시드 -- 마찬가지로 사실상 동일 재현.
# 합계 7300장(원본 8486장보다 적음 -- v2/v3의 "1200장 출처 불명" 부분은 근사 없이 생략,
# 아래 3)/4)는 결정론적 재현이라 정밀함).
set -uo pipefail
cd /workspace/round3train
MUSESCORE=/workspace/musescore/squashfs-root/AppRun
LOG=/workspace/gen_replay_reconstruction.log
STATUS=/workspace/gen_replay_reconstruction_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) replay 풀 재구성 시작 ==="
: > "$STATUS"

N_SHARDS=7   # 이 pod 실측 cpu.max=15.3코어(2026-08-04) 기준 절반

OUT_DIR=/workspace/data/replay_pool_reconstructed
mkdir -p "$OUT_DIR"

count_existing () {
  local idx_base=$1
  python3 -c "
import glob, re
n = 0
for p in glob.glob('$OUT_DIR/num*.png'):
    m = re.search(r'num(\d+)\.png$', p)
    if m and $idx_base <= int(m.group(1)) < $idx_base + 1000000:
        n += 1
print(n)
"
}

gen_batch () {
  local name=$1 target=$2 idx_base=$3 seed_base=$4 is_single=$5
  shift 5
  local extra_args=("$@")
  local existing=$(count_existing "$idx_base")
  echo "[$name] 기존 ${existing}장(목표 ${target}장)"
  if [ "$existing" -ge "$target" ]; then
    echo "[$name] 이미 충분 -- 스킵"
    echo "STAGE_DONE:${name}:${existing}" >> "$STATUS"
    return
  fi
  local need=$(( target - existing ))
  local n_shards=$N_SHARDS
  if [ "$need" -lt "$n_shards" ]; then n_shards=$need; fi
  local shard_base=$(( need / n_shards ))
  local pids=()
  for i in $(seq 0 $((n_shards - 1))); do
    local shard_start=$(( idx_base + i * 10000 ))
    local shard_seed=$(( seed_base + i ))
    local this_count=$shard_base
    if [ "$i" -eq $((n_shards - 1)) ]; then
      this_count=$(( need - shard_base * (n_shards - 1) ))
    fi
    local staff_flag=""
    if [ "$is_single" = "1" ]; then staff_flag="--single-staff"; fi
    xvfb-run -a python3 generate_scores.py \
      --count "$this_count" --output "$OUT_DIR" --musescore "$MUSESCORE" \
      $staff_flag --start-idx "$shard_start" --seed "$shard_seed" \
      "${extra_args[@]}" > "/workspace/gen_replay_${name}_${i}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  local final=$(count_existing "$idx_base")
  if [ "$fail" -ne 0 ] || [ "$final" -lt "$target" ]; then
    echo "[$name] 경고: 목표 미달(${final}/${target}) -- 계속 진행"
    echo "STAGE_SHORT:${name}:${final}:${target}" >> "$STATUS"
  else
    echo "[$name] 완료: ${final}/${target}"
    echo "STAGE_DONE:${name}:${final}" >> "$STATUS"
  fi
}

# ── 1) general_diversity(2500) -- r3_v2_mixed.sh BASE_ARGS, 단일:대보표=15:85 ──
GD_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.22
         --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
         --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
         --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
         --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
         --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
         --clef-change-prob 0.15
         --markov-bias 0.3 --markov-table markov_transitions.json)
gen_batch "general_single" 375 21000001 2100001 1 "${GD_ARGS[@]}" --preferred-register-prob 0.3
gen_batch "general_grand"  2125 21500001 2150001 0 "${GD_ARGS[@]}" --cross-register-prob 0.30 --preferred-register-prob 0.6

# ── 2) rare_durations(800) -- r5_rare_durations.sh BASE_ARGS, 단일:대보표=15:85 ──
RD_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.22
         --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
         --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
         --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
         --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
         --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
         --rare-long-bias 3.0 --preferred-register-prob 0.3
         --markov-bias 0.3 --markov-table markov_transitions.json)
gen_batch "rare_single" 120 22000001 2200001 1 "${RD_ARGS[@]}"
gen_batch "rare_grand"  680 22500001 2250001 0 "${RD_ARGS[@]}" --cross-register-prob 0.20

# ── 3) l4_major_synth(2000) -- pod_gen_r7_synth_only.sh와 완전히 동일 파라미터+시드 ──
R7_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.22
         --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
         --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
         --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
         --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
         --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
         --preferred-register-prob 0.3
         --markov-bias 0.3 --markov-table markov_transitions.json)
gen_batch "r7_single" 300  11000001 1100001 1 "${R7_ARGS[@]}"
gen_batch "r7_grand"  1700 11500001 1150001 0 "${R7_ARGS[@]}" --cross-register-prob 0.20

# ── 4) chord_accidental_synth(2000) -- curriculum_r9_chord_accidental.sh와 완전히 동일 ──
R9_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.28
         --chord-min-notes 2 --chord-max-notes 4 --chord-size-weights "2:35,3:35,4:30"
         --repeat-prob 0
         --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
         --diatonic-bias 0.55 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
         --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
         --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
         --preferred-register-prob 0.3
         --markov-bias 0.3 --markov-table markov_transitions.json)
gen_batch "r9_single" 300  12000001 1200001 1 "${R9_ARGS[@]}"
gen_batch "r9_grand"  1700 12500001 1250001 0 "${R9_ARGS[@]}" --cross-register-prob 0.20

TOTAL_N=$(find "$OUT_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "=== $(date) replay 풀 재구성 종료: 총 ${TOTAL_N}/7300장 -> $OUT_DIR ==="
echo "GEN_COMPLETE:${TOTAL_N}" >> "$STATUS"
