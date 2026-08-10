#!/bin/bash
# 극단 음역(덧줄 많은 고음/저음) 집중 데이터 3000장 생성 (2026-07-31).
# 배경: Step1 학습 데이터(6157장) 옥타브 분포 집계 결과 옥타브 2~5는 21.7~27.5%씩
# 고르게 분포했지만 옥타브 6은 22건(0.0%), 옥타브 1 이하는 사실상 0건이었음. 89곡
# 실측에서 나온 체계적 오독(D6->B5, C6->A5, C6->E4, Bb1->Db2)이 전부 GT가 이
# 노출 거의 없는 옥타브였던 것과 일치 -- 이 노출 격차 자체가 원인일 가능성 검증.
# generate_scores.py에 이미 있던 --preferred-register-prob(치: D3~A3 U A5~A6,
# 베이스: C2~B2 U F4~B4)를 Step1의 0.15보다 훨씬 세게(0.7) 켬.
set -uo pipefail
cd /workspace/round3train
MUSESCORE=/workspace/musescore/squashfs-root/AppRun
LOG=/workspace/gen_extreme.log
STATUS=/workspace/gen_extreme_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 극단 음역 데이터 생성 시작 ==="
: > "$STATUS"

N_SHARDS=20

# Step1의 POOL_B(희소) 계열 리듬 밀도를 그대로 써서 "음역 노출"만 새 변수로 남긴다
# (리듬 복잡도까지 같이 바뀌면 나중에 무엇 때문에 좋아졌는지 구분이 안 됨).
COMMON_ARGS=(--min-measures 2 --max-measures 6 --chord-prob 0.22
             --chord-min-notes 2 --chord-max-notes 4
             --chord-size-weights "2:60,3:30,4:10" --chord-interval-weights "2:60,3:30,4:10"
             --chord-progression-bias 0.6
             --repeat-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0
             --hairpin-prob 0 --ottava-prob 0 --tuplet-prob 0.05
             --tuplet-ledger-prob 0.3 --tuplet-rest-prob 0.15
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.0 --tie-prob 0.25 --clef-change-prob 0.15
             --courtesy-accidental-prob 0.1
             --markov-bias 0.5 --markov-table markov_transitions.json
             --dotted8-bias 6.0 --eighth-run-prob 0.05 --sixteenth-run-prob 0.08
             --long-note-bias 1.5
             --preferred-register-prob 0.7)
GRAND_ONLY_ARGS=(--accompaniment-prob 0.6)

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

POOL_DIR=data/extreme_register_pool
mkdir -p "$POOL_DIR"

# 750 단일오선 + 2250 대보표 (Step1의 25/75 비율과 동일하게 맞춤)
gen_stage "extreme_single" "$POOL_DIR" 750  8000001 900001 1
gen_stage "extreme_grand"  "$POOL_DIR" 2250 9000001 910001 0 "${GRAND_ONLY_ARGS[@]}"

TOTAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "=== $(date) 극단 음역 데이터 생성 종료: 총 ${TOTAL_N}/3000장 ==="
echo "GEN_COMPLETE:${TOTAL_N}" >> "$STATUS"
