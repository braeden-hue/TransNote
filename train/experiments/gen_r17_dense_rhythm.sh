#!/bin/bash
# round3train/gen_r17_dense_rhythm.sh
#
# duration(음가) 예측 축 보강용 합성 데이터 2000장 생성(2026-08-04).
#
# 배경: r15_cropfix_coordconv를 신규 6곡(19장)에 로컬 재실측한 결과(diag_new6_errors.py),
# duration 오류 중 dur-1/16 missing(19건)+extra(18건)가 압도적 1위 -- project_dense_rhythm_failure
# 메모리에서 이미 지적된 문제가 r15에서도 여전함. 실사 GT 126곡의 duration 토큰 분포를 실측하니
# (2026-08-04) 합성 기본 DURATIONS(1/4 55%, 1/8 15%, 1/16 8%)와 실사 실측(1/4 22%, 1/8 52%,
# 1/16 19%)이 크게 어긋나 있었음 -- 실사는 8분음표가 압도적인데 합성은 4분음표 위주로 생성돼왔음.
#
# generate_scores.py에 신규 추가한 --eighth-bias/--sixteenth-bias(EIGHTH_BIAS/SIXTEENTH_BIAS,
# SHORT_NOTE_BIAS와 달리 1/8·1/16을 개별 조정 가능)를 실사 실측치에 맞춰 역산(로컬에서
# _choose_dur() 20만회 몬테카를로 검증 완료, generate_scores.py 실제 렌더링은 미검증 --
# MuseScore/xvfb 없는 로컬 환경 한계):
#   --eighth-bias 7.65 --sixteenth-bias 4.85 --dotted8-bias 5.0
#   -> 1/8 50.0%, 1/4 21.4%, 1/16 18.3%, 3/16 1.17% (실사 실측과 거의 일치)
#
# 중요: r11_rhythm_focus(2026-08-03, --dotted8-bias 20 --eighth/sixteenth-run-prob 0.30/0.35)가
# "정확히 이 축"을 이미 시도했다가 held-out 정확도 -9~13pp 급락으로 실패했음
# (project_r8_2_diversity_is_best_checkpoint 메모리). 그러나 실패 원인으로 지목된 건 콘텐츠
# 타깃팅 자체가 아니라 "r8_2_diversity에서 재개 + 좁은 합성 데이터를 MAIN으로 학습 + 고정 replay"
# 라는 레시피였고, 이후 r12~r15는 전부 "실사 120곡을 MAIN으로 유지 + 합성은 replay 축으로만
# 추가"하는 방식으로만 성공했음. 이번 라운드는 반드시 그 성공 패턴을 따라 이 스크립트로 생성한
# 배치를 MAIN이 아니라 replay 풀에 병합해서 쓸 것(curriculum_r17_dense_rhythm.sh 참고).
#
# eighth-run-prob/sixteenth-run-prob(연속 비트 그룹 강제)는 r11의 공격적인 값(0.30/0.35)을
# 그대로 재사용하지 않고 절반 수준(0.15/0.20)의 보수적인 값으로 둠 -- 이번 라운드는 분포 자체를
# 실측치에 맞추는 게 핵심이고, run-prob는 검증 안 된 판단값이므로 pod 재개 전에 필요시 조정할 것.
set -uo pipefail
cd /workspace/round3train
MUSESCORE=/workspace/musescore/squashfs-root/AppRun
LOG=/workspace/gen_r17_dense_rhythm.log
STATUS=/workspace/gen_r17_dense_rhythm_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) duration 분포 보강 데이터 생성 시작(r17) ==="
: > "$STATUS"

# POD_TRAINING_CHECKLIST.md 5번: nproc이 아니라 cpu.max 코어 수의 절반 기준으로 조정할 것
N_SHARDS=6

COMMON_ARGS=(--min-measures 2 --max-measures 6 --chord-prob 0.10
             --chord-min-notes 2 --chord-max-notes 3
             --chord-size-weights "2:70,3:30" --chord-interval-weights "2:70,3:30"
             --chord-progression-bias 0.3
             --repeat-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0
             --hairpin-prob 0 --ottava-prob 0 --tuplet-prob 0.03
             --tuplet-ledger-prob 0.3 --tuplet-rest-prob 0.15
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.0 --tie-prob 0.25 --clef-change-prob 0.1
             --courtesy-accidental-prob 0.1
             --markov-bias 0.5 --markov-table markov_transitions.json
             --eighth-bias 7.65 --sixteenth-bias 4.85 --dotted8-bias 5.0
             --eighth-run-prob 0.15 --sixteenth-run-prob 0.20
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
        > "/workspace/gen_r17_${stage_name}_r${round}_${i}.log" 2>&1 &
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

POOL_DIR=data/r17_dense_rhythm_pool
mkdir -p "$POOL_DIR"

# r11과 동일 비율(단일오선 25% / 대보표 75%)
gen_stage "dense_single" "$POOL_DIR" 500  17000001 1700001 1
gen_stage "dense_grand"  "$POOL_DIR" 1500 18000001 1710001 0 "${GRAND_ONLY_ARGS[@]}"

TOTAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "=== $(date) duration 분포 보강 데이터 생성 종료: 총 ${TOTAL_N}/2000장 ==="
echo "GEN_COMPLETE:${TOTAL_N}" >> "$STATUS"
