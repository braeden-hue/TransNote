#!/bin/bash
# 1단계(화음 집중) 데이터 1500장 생성 (2026-07-31).
# 배경: 극단음역 재학습 체크포인트로 Step1 100장 재검증 결과, 오류 496건 중 288건(58%)이
# "음표/길이 과잉(extra)·누락(missing)" -- 계이름을 틀린 게 아니라 화음 구성음 개수를
# 세는 것 자체가 헷갈리는 문제로 보임. 화음 크기(2/3/4음)를 고르게, 다른 복잡도(셋잇단/
# 화성진행 편향)는 낮춰서 "화음 개수 세기" 신호만 깨끗하게 노출.
set -uo pipefail
cd /workspace/round3train
MUSESCORE=/workspace/musescore/squashfs-root/AppRun
LOG=/workspace/gen_chord_focus.log
STATUS=/workspace/gen_chord_focus_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 화음 집중 데이터 생성 시작 ==="
: > "$STATUS"

N_SHARDS=20

# 화음 확률을 매우 높이고(0.5) 2/3/4음 균등 비중(기존 60:30:10 -> 33:33:34),
# 화성진행 편향은 낮춰서(0.3) 다양한 화음 조합이 골고루 나오게 함. 리듬 밀도는
# Step1 POOL_B(희소) 수준으로 낮춰서 리듬 복잡도가 화음 신호를 가리지 않게 함.
COMMON_ARGS=(--min-measures 2 --max-measures 6 --chord-prob 0.5
             --chord-min-notes 2 --chord-max-notes 4
             --chord-size-weights "2:34,3:33,4:33" --chord-interval-weights "2:34,3:33,4:33"
             --chord-progression-bias 0.3
             --repeat-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0
             --hairpin-prob 0 --ottava-prob 0 --tuplet-prob 0.03
             --tuplet-ledger-prob 0.3 --tuplet-rest-prob 0.15
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.0 --tie-prob 0.25 --clef-change-prob 0.1
             --courtesy-accidental-prob 0.1
             --markov-bias 0.5 --markov-table markov_transitions.json
             --dotted8-bias 8.0 --eighth-run-prob 0.05 --sixteenth-run-prob 0.05
             --long-note-bias 1.2
             --preferred-register-prob 0.3)
GRAND_ONLY_ARGS=(--accompaniment-prob 0.5)

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
  local stage_name=$1 out_dir=$2 target=$3 idx_base=$4 seed_base=$5 is_single=$6
  shift 6
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
      local staff_flag=""
      if [ "$is_single" = "1" ]; then staff_flag="--single-staff"; fi
      xvfb-run -a python3 generate_scores.py \
        --count "$this_count" --output "$out_dir" --musescore "$MUSESCORE" \
        $staff_flag --start-idx "$shard_start" --seed "$shard_seed" \
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

POOL_DIR=data/chord_focus_pool
mkdir -p "$POOL_DIR"

# 375 단일오선 + 1125 대보표 (기존 25/75 비율 유지)
gen_stage "chord_single" "$POOL_DIR" 375  10000001 1000001 1
gen_stage "chord_grand"  "$POOL_DIR" 1125 11000001 1010001 0 "${GRAND_ONLY_ARGS[@]}"

TOTAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "=== $(date) 화음 집중 데이터 생성 종료: 총 ${TOTAL_N}/1500장 ==="
echo "GEN_COMPLETE:${TOTAL_N}" >> "$STATUS"
