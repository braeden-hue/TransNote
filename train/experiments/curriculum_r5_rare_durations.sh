#!/bin/bash
# round3train/curriculum_r5_rare_durations.sh
#
# 보강 라운드 -- 온음표(dur-1/1)/점2분음표(dur-3/4) 노출 부족 보정. 신규 --rare-long-bias
# 옵션(2026-08-02 추가, generate_scores.py)으로 이 두 duration만 콕 집어 부스트한다.
# newage 신규 라벨(14곡) 확인 결과 dur-3/4 3.4%, dur-1/1은 여전히 0건 -- 실사 근거로 보정.
#
# r4_noise(노이즈 학습 완료 체크포인트)에서 이어받되, 이번 라운드는 노이즈를 완전히 끄지
# 않고 가볍게(L1)만 유지 -- Round4에서 얻은 노이즈 강건성이 옅어지는 걸 막기 위함
# (Round1 v2가 replay 없이 학습해서 레지스터를 잠깐 잃었던 것과 같은 종류의 리스크 방지).
#
# replay는 이번에 새로 구성한 replay_pool_v2(2500장, 실제 파일 복사본 -- 심볼릭 링크 아님,
# 원본 r1v2/r2v2/r3v2 풀은 디스크 정리로 이미 삭제됨)에서 2000장 샘플링.

set -uo pipefail
LOG=/workspace/curriculum_r5.log
STATUS=/workspace/curriculum_r5_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 5(온음표/점2분음표 보강) 시작 ==="
: > "$STATUS"

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
POOL_DIR=/workspace/data/r5_rare_durations
SINGLE_COUNT=225
GRAND_COUNT=1275
START_IDX_SINGLE=9000001
START_IDX_GRAND=9500001
SEED_SINGLE=900001
SEED_GRAND=950001
N_SHARDS=6

REPLAY_DIR=/workspace/data/replay_pool_v2
REPLAY_COUNT=2000
RESUME_CKPT=/workspace/models/r4_noise/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r5_rare_durations
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=12
FREEZE_EPOCHS=0
GATE=90
MARKOV_TABLE=/workspace/round3train/markov_transitions.json
MARKOV_BIAS=0.3

BASE_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.22
           --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
           --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
           --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
           --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
           --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
           --rare-long-bias 3.0
           --preferred-register-prob 0.3
           --markov-bias "$MARKOV_BIAS" --markov-table "$MARKOV_TABLE")
GRAND_ONLY_ARGS=(--cross-register-prob 0.20)

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] 데이터 풀 확인 중..."
TOTAL_COUNT=$((SINGLE_COUNT + GRAND_COUNT))
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
MIN_OK_N=$(( TOTAL_COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[$STAGE_NAME] 데이터 풀 이미 존재(${EXISTING_N}/${TOTAL_COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"
  mkdir -p "$POOL_DIR"

  echo "[$STAGE_NAME] 단일오선(${SINGLE_COUNT}장) 생성 중..."
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

  echo "[$STAGE_NAME] 대보표(${GRAND_COUNT}장) 생성 중..."
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
echo "STAGE_DATA_DONE:$STAGE_NAME" >> "$STATUS"

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, replay=${REPLAY_COUNT}, noise L1 light) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --page_level_noise --noise_level 1 \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS"
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- register_accuracy_r89.py/eval_exactpicture_realphotos.py로 실측 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) Round 5 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
