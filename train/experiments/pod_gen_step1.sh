#!/bin/bash
# Step1 데이터 생성: Pool A(밀집/오선검출 강화, 2500장) + Pool B(희소, 3500장) = 6000장.
# 2026-07-31 세션에서 로컬 검증 완료된 파라미터 그대로 사용.
#
# 2026-07-31 개정: N_SHARDS=32에서 xvfb-run 동시 실행 경쟁으로 일부 샤드가 렌더링에
# 실패하면서 파이프라인 전체가 한 번에 죽는 문제 발생(사용자 제보) -- N_SHARDS를 20으로
# 낮추고, 부분 실패해도 전체를 죽이지 않고 부족분만 다른 인덱스 구간에서 보충 생성하는
# 방식으로 재작성. 기존에 생성된 파일은 그대로 재사용(삭제 안 함).
set -uo pipefail
cd /workspace/round3train
MUSESCORE=/workspace/musescore/squashfs-root/AppRun
LOG=/workspace/step1_gen.log
STATUS=/workspace/step1_gen_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) Step1 데이터 생성 시작 ==="
: > "$STATUS"

N_SHARDS=20

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
             --preferred-register-prob 0.15)

POOL_A_ARGS=(--dotted8-bias 18.0 --eighth-run-prob 0.20 --sixteenth-run-prob 0.25
             --eighth-run-prob-2-4 0.10 --sixteenth-run-prob-2-4 0.35)
POOL_B_ARGS=(--dotted8-bias 6.0 --eighth-run-prob 0.05 --sixteenth-run-prob 0.08
             --long-note-bias 1.5)
GRAND_ONLY_ARGS=(--accompaniment-prob 0.6)

# idx_base들은 서로 100만(1,000,000)씩 떨어뜨려서 절대 겹치지 않게 함. 한 블록(1,000,000)
# 안에서 round_offset(0/600000/850000)으로 재시도마다 다른 하위 구간을 써서, 이전에
# 이미 성공한 파일과도 충돌하지 않게 한다.
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

# 단일오선 또는 대보표 한 단계를 목표치까지 채운다. 부족하면 최대 2회 추가 보충
# 시도(round_offset 0 -> 600000 -> 850000), 그래도 부족하면 경고만 남기고 계속 진행
# (전체 파이프라인을 죽이지 않음 -- 2026-07-31 사용자 지시).
gen_stage () {
  local stage_name=$1 out_dir=$2 target=$3 idx_base=$4 seed_base=$5 is_single=$6
  shift 6
  local extra_args=("$@")

  local existing=$(count_existing "$out_dir" "$idx_base")
  echo "[$stage_name] 기존 ${existing}장 확인됨 (목표 ${target}장)"

  # 400000부터 시작 -- 이전(실패했던 32샤드 x stride 10000) 실행이 이미 0~310000
  # 구간을 썼을 수 있어(예: poolA_single 기존 530장), 그 구간과 겹치지 않게 함
  # (2026-07-31, 기존 파일 재사용 시 충돌 방지).
  local round_offsets=(400000 600000 850000)
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
      echo "[$stage_name] 라운드 $((round+1))에서 일부 샤드 실패(무시하고 재확인) -- 로그: /workspace/gen_${stage_name}_r${round}_*.log"
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

POOL_DIR=data/step1_pool
mkdir -p "$POOL_DIR"

gen_stage "poolA_single" "$POOL_DIR" 625  3000001 700001 1 "${POOL_A_ARGS[@]}"
gen_stage "poolA_grand"  "$POOL_DIR" 1875 4000001 710001 0 "${POOL_A_ARGS[@]}" "${GRAND_ONLY_ARGS[@]}"
gen_stage "poolB_single" "$POOL_DIR" 875  5000001 800001 1 "${POOL_B_ARGS[@]}"
gen_stage "poolB_grand"  "$POOL_DIR" 2625 6000001 810001 0 "${POOL_B_ARGS[@]}" "${GRAND_ONLY_ARGS[@]}"

TOTAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "=== $(date) Step1 데이터 생성 종료: 총 ${TOTAL_N}/6000장 (한 폴더 $POOL_DIR 로 통합) ==="
echo "GEN_COMPLETE:${TOTAL_N}" >> "$STATUS"
