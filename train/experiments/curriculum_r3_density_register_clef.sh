#!/bin/bash
# round3train/curriculum_r3_density_register_clef.sh
#
# Round2(대보표 기초) 위에 세 축을 동시에 강화: 리듬 조밀성(이미 Round2에도 있지만 재확인),
# 넓은 음역대(Round2의 희석된 값에서 원래 강도 근처로 재상승), 마디 중간 클렙전환(신규,
# 단일오선+대보표 둘 다 -- 2026-07-30에 대보표 지원 추가함).
#
# Round1/Round2와 달리 이 라운드부터 단일오선+대보표를 한 라운드 안에서 섞는다 -- 둘 다
# 이미 각각 학습됐으니 이제 같이 노출시켜도 안전하다고 판단(대보표 단독 도입이라는
# 가장 위험한 구조적 도약은 이미 Round2에서 지나감).
#
# 사전 조건: curriculum_r2_grandstaff.sh가 STAGE_PASSED로 끝나있어야 함.
# 게이트 통과와 별개로 실측(eval_page_noise.py 등)으로 Round2 품질 확인 먼저 할 것 --
# 실측이 안 좋으면 이 라운드로 넘어가기 전에 Round2 데이터/epoch부터 보강.

set -uo pipefail
LOG=/workspace/curriculum_r3.log
STATUS=/workspace/curriculum_r3_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 3(조밀성+넓은음역+클렙전환, 단일오선+대보표 혼합) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r2_grandstaff" /workspace/curriculum_r2_status.txt 2>/dev/null; then
  echo "[r3] Round2가 STAGE_PASSED 상태가 아님 -- 중단(먼저 확인 필요)"
  echo "PIPELINE_STOPPED_R2_NOT_PASSED" >> "$STATUS"
  exit 1
fi

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
POOL_DIR=/workspace/data/r3_density_register_clef
R1_POOL_DIR=/workspace/data/r1_foundation
R2_POOL_DIR=/workspace/data/r2_grandstaff
R1R2_MERGED_DIR=/workspace/data/r1_r2_merged   # 아래에서 R1/R2 각각 지정 개수만 심볼릭 링크
SINGLE_COUNT=2500
GRAND_COUNT=2500
R1_REPLAY_N=1000   # R1(3000장)에서 고정 선별
R2_REPLAY_N=2000   # R2(5000장)에서 고정 선별(대보표라 더 비중 둠)
REPLAY_COUNT=3000   # R1_REPLAY_N + R2_REPLAY_N -- train.py가 이 개수만큼 통합 풀에서 뽑는데
                    # 통합 풀 크기 자체를 정확히 3000으로 맞춰뒀으니 전부 다 쓰이게 됨
START_IDX_SINGLE=3000001
START_IDX_GRAND=4000001
SEED_SINGLE=300001
SEED_GRAND=400001
N_SHARDS=6

RESUME_OUT=/workspace/models/r2_grandstaff
STAGE_NAME=r3_density_register_clef
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=30
FREEZE_EPOCHS=8
GATE=90
MARKOV_TABLE=/workspace/round3train/markov_transitions.json
MARKOV_BIAS=0.5

# 공통(치/베이스 공통 축) -- 교차음역/선호음역은 대보표에만 의미 있어서 대보표 호출에만 추가.
BASE_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.08
           --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
           --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
           --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
           --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.15
           --dotted8-bias 10.0 --short-note-bias 2.0 --clef-change-prob 0.15
           --markov-bias "$MARKOV_BIAS" --markov-table "$MARKOV_TABLE")
GRAND_ONLY_ARGS=(--cross-register-prob 0.30 --preferred-register-prob 0.6)

echo "[$STAGE_NAME] 데이터 풀 확인 중..."
TOTAL_COUNT=$((SINGLE_COUNT + GRAND_COUNT))
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
MIN_OK_N=$(( TOTAL_COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[$STAGE_NAME] 데이터 풀 이미 존재(${EXISTING_N}/${TOTAL_COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"
  mkdir -p "$POOL_DIR"

  echo "[$STAGE_NAME] 단일오선(${SINGLE_COUNT}장, ${N_SHARDS}개 병렬) 생성 중..."
  SHARD_BASE=$(( SINGLE_COUNT / N_SHARDS ))
  PIDS=()
  for i in $(seq 0 $((N_SHARDS - 1))); do
    SHARD_START=$(( START_IDX_SINGLE + i * 100000 ))
    SHARD_SEED=$(( SEED_SINGLE + i ))
    THIS_COUNT=$SHARD_BASE
    if [ "$i" -eq $((N_SHARDS - 1)) ]; then
      THIS_COUNT=$(( SINGLE_COUNT - SHARD_BASE * (N_SHARDS - 1) ))
    fi
    xvfb-run -a python3 /workspace/round3train/generate_scores.py \
      --count "$THIS_COUNT" --output "$POOL_DIR" --musescore "$MUSESCORE" \
      --single-staff --start-idx "$SHARD_START" --seed "$SHARD_SEED" \
      "${BASE_ARGS[@]}" > "/workspace/gen_shard_${STAGE_NAME}_single_${i}.log" 2>&1 &
    PIDS+=($!)
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
  if [ "$FAIL" -ne 0 ]; then
    echo "[$STAGE_NAME] 단일오선 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:single" >> "$STATUS"
    exit 1
  fi

  echo "[$STAGE_NAME] 대보표(${GRAND_COUNT}장, ${N_SHARDS}개 병렬) 생성 중..."
  SHARD_BASE=$(( GRAND_COUNT / N_SHARDS ))
  PIDS=()
  for i in $(seq 0 $((N_SHARDS - 1))); do
    SHARD_START=$(( START_IDX_GRAND + i * 100000 ))
    SHARD_SEED=$(( SEED_GRAND + i ))
    THIS_COUNT=$SHARD_BASE
    if [ "$i" -eq $((N_SHARDS - 1)) ]; then
      THIS_COUNT=$(( GRAND_COUNT - SHARD_BASE * (N_SHARDS - 1) ))
    fi
    xvfb-run -a python3 /workspace/round3train/generate_scores.py \
      --count "$THIS_COUNT" --output "$POOL_DIR" --musescore "$MUSESCORE" \
      --start-idx "$SHARD_START" --seed "$SHARD_SEED" \
      "${BASE_ARGS[@]}" "${GRAND_ONLY_ARGS[@]}" > "/workspace/gen_shard_${STAGE_NAME}_grand_${i}.log" 2>&1 &
    PIDS+=($!)
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
  if [ "$FAIL" -ne 0 ]; then
    echo "[$STAGE_NAME] 대보표 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:grand" >> "$STATUS"
    exit 1
  fi

  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[$STAGE_NAME] 데이터 풀 완성: ${FINAL_N}/${TOTAL_COUNT}장"
fi

echo "[$STAGE_NAME] R1(${R1_REPLAY_N})+R2(${R2_REPLAY_N}) 고정 선별 replay 풀 구성 중 (심볼릭 링크, 복사 없음)..."
rm -rf "$R1R2_MERGED_DIR"
mkdir -p "$R1R2_MERGED_DIR"
for stem in $(find "$R1_POOL_DIR" -maxdepth 1 -name '*.png' -printf '%f\n' | sed 's/\.png$//' | shuf -n "$R1_REPLAY_N"); do
  ln -s "$R1_POOL_DIR/$stem.png"  "$R1R2_MERGED_DIR/$stem.png"
  ln -s "$R1_POOL_DIR/$stem.json" "$R1R2_MERGED_DIR/$stem.json"
done
for stem in $(find "$R2_POOL_DIR" -maxdepth 1 -name '*.png' -printf '%f\n' | sed 's/\.png$//' | shuf -n "$R2_REPLAY_N"); do
  ln -s "$R2_POOL_DIR/$stem.png"  "$R1R2_MERGED_DIR/$stem.png"
  ln -s "$R2_POOL_DIR/$stem.json" "$R1R2_MERGED_DIR/$stem.json"
done
MERGED_N=$(find "$R1R2_MERGED_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "[$STAGE_NAME] 통합 풀: ${MERGED_N}장 (replay_count=${REPLAY_COUNT})"

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=r2_grandstaff, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# train.py --replay_dir는 디렉토리 하나만 받으므로, 위에서 심볼릭 링크로 미리 합쳐둔
# R1+R2 통합 풀($R1R2_MERGED_DIR)을 가리킨다 -- 실제 파일 복사 없이 두 풀 다 replay 가능.
python3 -u train.py --phase 2 \
  --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_OUT/seq2seq_best.pt" \
  --replay_dir "$R1R2_MERGED_DIR" --replay_count "$REPLAY_COUNT" \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- eval_page_noise.py/eval_token_acc.py로 실측 재검증할 것"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) Round 3 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
