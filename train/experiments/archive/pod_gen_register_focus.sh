#!/bin/bash
# 레지스터 편향(clef register bias) 집중 데이터 1500장 생성 (2026-08-03).
# 배경: newage07/09 held-out 오류 분석에서, 베이스보표가 레저선(덧줄)이 필요한 음역
# (아래로 많이 벗어나거나 위로 치보표에 근접)을 읽을 때 실제 위치 대신 "베이스클렙의
# 전형적 음역"으로 회귀하는 패턴 확인(newage09: G4를 B2로, 2옥타브 이상 오독). 이건
# project_register_bias_failure.md에 2026-07-28부터 이미 기록된 미해결 이슈 --
# generate_scores.py에 CROSS_REGISTER_PROB 인프라(TREBLE_LOW_PITCHES/BASS_HIGH_PITCHES)
# 가 이미 있었지만 기본값 0.0이라 어느 커리큘럼 라운드에서도 실제로 켠 적이 없었음.
# 대보표(치+베이스)에만 적용되는 축이라 대보표 전용으로 생성.
set -uo pipefail
cd /workspace/round3train
MUSESCORE=/workspace/musescore/squashfs-root/AppRun
LOG=/workspace/gen_register_focus.log
STATUS=/workspace/gen_register_focus_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 레지스터 편향 집중 데이터 생성 시작 ==="
: > "$STATUS"

N_SHARDS=20

# cross-register-prob를 0.45로 높여 절반 가까이는 교차음역(치=낮은음역/베이스=높은음역
# swap 또는 양쪽 다 확장) 노출. 다른 축(화음/셋잇단/장식)은 낮춰서 레지스터 신호만
# 깨끗하게 노출("한 번에 축 하나만" 원칙).
COMMON_ARGS=(--min-measures 2 --max-measures 6 --chord-prob 0.15
             --chord-min-notes 2 --chord-max-notes 3
             --chord-progression-bias 0.3
             --repeat-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0
             --hairpin-prob 0 --ottava-prob 0 --tuplet-prob 0.03
             --diatonic-bias 0.75 --dynamic-prob 0.2 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.0 --tie-prob 0.15 --clef-change-prob 0.05
             --courtesy-accidental-prob 0.1
             --markov-bias 0.5 --markov-table markov_transitions.json
             --dotted8-bias 4.0 --eighth-run-prob 0.1 --sixteenth-run-prob 0.05
             --cross-register-prob 0.45
             --accompaniment-prob 0.4)

count_existing () {
  local out_dir=$1 idx_base=$2
  python3 -c "
import glob, re
n = 0
for p in glob.glob('$out_dir/num*.png'):
    m = re.search(r'num(\d+)\.png$', p)
    if m and $idx_base <= int(m.group(1)) < $idx_base + 1000000:
        n += 1
print(n)
"
}

gen_stage () {
  local stage_name=$1 out_dir=$2 target=$3 idx_base=$4 seed_base=$5
  shift 5
  local extra_args=("$@")

  local existing=$(count_existing "$out_dir" "$idx_base")
  echo "[$stage_name] 기존 ${existing}장 확인됨 (목표 ${target}장)"

  local round_offsets=(0 400000 700000)
  local round=0
  while [ "$existing" -lt "$target" ] && [ "$round" -lt 3 ]; do
    local need=$(( target - existing ))
    local offset=${round_offsets[$round]}
    local n_shards=$N_SHARDS
    if [ "$need" -lt "$n_shards" ]; then n_shards=$need; fi
    echo "[$stage_name] 라운드 $((round+1)): ${need}장 부족 -> ${n_shards}샤드로 생성 시도..."

    local shard_base=$(( need / n_shards ))
    local pids=()
    for i in $(seq 0 $((n_shards - 1))); do
      local shard_start=$(( idx_base + offset + i * 5000 ))
      local shard_seed=$(( seed_base + round * 1000 + i ))
      local this_count=$shard_base
      if [ "$i" -eq $((n_shards - 1)) ]; then
        this_count=$(( need - shard_base * (n_shards - 1) ))
      fi
      if [ "$this_count" -le 0 ]; then continue; fi
      xvfb-run -a python3 generate_scores.py \
        --count "$this_count" --output "$out_dir" --musescore "$MUSESCORE" \
        --start-idx "$shard_start" --seed "$shard_seed" \
        "${COMMON_ARGS[@]}" "${extra_args[@]}" \
        > "/workspace/gen_${stage_name}_r${round}_${i}.log" 2>&1 &
      pids+=($!)
    done
    local fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    if [ "$fail" -ne 0 ]; then
      echo "[$stage_name] 라운드 $((round+1))에서 일부 샤드 실패(무시하고 재확인)"
    fi

    existing=$(count_existing "$out_dir" "$idx_base")
    round=$((round + 1))
  done

  if [ "$existing" -lt "$target" ]; then
    echo "[$stage_name] 경고: 3회 시도 후에도 목표 미달 (${existing}/${target}) -- 계속 진행"
    echo "STAGE_SHORT:${stage_name}:${existing}:${target}" >> "$STATUS"
  else
    echo "[$stage_name] 완료: ${existing}/${target}"
    echo "STAGE_DONE:${stage_name}:${existing}" >> "$STATUS"
  fi
}

POOL_DIR=data/register_focus_pool
mkdir -p "$POOL_DIR"

# 전량 대보표(CROSS_REGISTER_PROB는 대보표 전용 축)
gen_stage "register_grand" "$POOL_DIR" 1500 20000001 2010001

TOTAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "=== $(date) 레지스터 편향 집중 데이터 생성 종료: 총 ${TOTAL_N}/1500장 ==="
echo "GEN_COMPLETE:${TOTAL_N}" >> "$STATUS"
