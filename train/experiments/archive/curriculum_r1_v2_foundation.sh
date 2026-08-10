#!/bin/bash
# round3train/curriculum_r1_v2_foundation.sh
#
# 2026-08-01 재학습 계획: 기존 curriculum_r1_foundation.sh(7/30) 구조는 유지하되,
# 그 뒤 세션에서 검증된 보정값(chord-prob 0.22, tuplet-prob 0.05, markov-bias 0.3,
# dotted8/eighth-run/sixteenth-run 강화, preferred-register-prob 신규)을 반영한 v2.
# 기존 v1(random init, 20에폭)과 달리 이번엔 처음부터가 아니라 현재 최선 체크포인트
# (RESUME_CKPT)에서 이어받는다 -- v1 실측상 random init은 20에폭에도 Acc 17.5%뿐이었고
# 이전 체크포인트에서 resume하면 5에폭에 92.2%였음("seq2seq는 반드시 resume").
#
# 사전 조건: pod_bootstrap.sh 완료(cv2/music21/xvfb/MuseScore), RESUME_CKPT 파일 존재 확인.

set -uo pipefail
LOG=/workspace/curriculum_r1_v2.log
STATUS=/workspace/curriculum_r1_v2_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 1 v2(기초 재학습: 단일오선, 보정 파라미터) 시작 ==="
: > "$STATUS"

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
POOL_DIR=/workspace/data/r1_v2_foundation
COUNT=3000
START_IDX=5000001
SEED=500001
N_SHARDS=6

STAGE_NAME=r1_v2_foundation
TRAIN_OUT=/workspace/models/$STAGE_NAME
RESUME_CKPT=${RESUME_CKPT:-/workspace/models/combined_coordconv/seq2seq_best.pt}
IN_CH=${IN_CH:-2}   # RESUME_CKPT의 실제 in_ch와 반드시 일치해야 함(CoordConv=2, 기존=1)
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}   # RESUME_CKPT 구조와 반드시 일치(기존=4, heightpreserve=2)
POOL_H=${POOL_H:-1}                             # RESUME_CKPT 구조와 반드시 일치(기존=1, heightpreserve=4)
EPOCHS=10
FREEZE_EPOCHS=0
GATE=90
REPLAY_DIR=${REPLAY_DIR:-}     # 2026-08-01: 1차 시도(replay 없음)에서 대보표 능력 망각 확인돼
                                # 추가됨. RESUME_CKPT가 학습됐던 원본 풀(예: lowmarkov_pool)을
                                # 지정하면 단일오선 재보정 중에도 대보표 능력을 유지시켜줌.
REPLAY_COUNT=${REPLAY_COUNT:-0}
MARKOV_TABLE=/workspace/round3train/markov_transitions.json
MARKOV_BIAS=0.3

COMMON_ARGS=(--single-staff --min-measures 1 --max-measures 4 --chord-prob 0.22
             --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
             --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
             --preferred-register-prob 0.3
             --markov-bias "$MARKOV_BIAS" --markov-table "$MARKOV_TABLE")
             # cross-register-prob은 단일오선엔 개념 없음(v1과 동일 이유로 제외).
             # preferred-register-prob 0.3은 v1엔 없었음 -- 단일오선도 덧줄 극단음역
             # 노출이 필요하다는 판단(2026-08-01 계획).

echo "[$STAGE_NAME] RESUME_CKPT=$RESUME_CKPT"
if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 파일 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] 데이터 풀 확인 중..."
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
MIN_OK_N=$(( COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[$STAGE_NAME] 데이터 풀 이미 존재(${EXISTING_N}/${COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"
  mkdir -p "$POOL_DIR"
  echo "[$STAGE_NAME] 데이터 생성 중 (${COUNT}장, ${N_SHARDS}개 병렬, markov-bias=${MARKOV_BIAS})..."
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
      > "/workspace/gen_shard_${STAGE_NAME}_${i}.log" 2>&1 &
    PIDS+=($!)
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
  done
  if [ "$FAIL" -ne 0 ]; then
    echo "[$STAGE_NAME] 일부 샤드 생성 실패 -- /workspace/gen_shard_${STAGE_NAME}_*.log 확인 필요"
    echo "PIPELINE_STOPPED_GENFAIL" >> "$STATUS"
    exit 1
  fi
  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[$STAGE_NAME] 데이터 풀 완성: ${FINAL_N}/${COUNT}장"
fi
echo "STAGE_DATA_DONE:$STAGE_NAME" >> "$STATUS"

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPLAY_ARGS=()
if [ -n "$REPLAY_DIR" ] && [ "$REPLAY_COUNT" -gt 0 ]; then
  REPLAY_ARGS=(--replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT")
  echo "[$STAGE_NAME] Replay 사용: $REPLAY_DIR ($REPLAY_COUNT장)"
fi
python3 -u train.py --phase 2 \
  --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  "${REPLAY_ARGS[@]}" \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- register_accuracy_r89.py/diagnose_third_confusion.py로 실측 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) Round 1 v2 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
