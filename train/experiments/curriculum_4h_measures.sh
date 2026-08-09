#!/bin/bash
# 4g(1마디 고정) 다음 단계: 여러 마디 인식 커리큘럼.
# 4g까지는 --min-measures 1 --max-measures 1이 전 단계에 강제 적용되어 있었음(설계 의도와
# 실제 스크립트가 어긋나 있었음, PODPLAN.md 2026-07-20 정정 참고) -- 이번엔 실제로 마디 수를
# 늘려서 학습한다. duration-subset은 안 씀(4g와 동일하게 전체 duration+쉼표+박자 3종 유지),
# --force-c-major는 유지(조표 다양화는 다음 단계 과제).
#
# 4h  : min=2 max=2 (1마디->2마디로 늘어나는 것 자체를 순수 격리해서 학습)
# 4h2 : min=2 max=4
# 4i  : min=2 max=6 (최종 목표 범위, 원래 4g 설계 의도였던 범위)
#
# RESUME_FROM 환경변수로 중간 단계부터 재개 가능 (0=4h부터, 1=4h2부터, 2=4i부터).

set -uo pipefail
LOG=/workspace/curriculum_4h_measures.log
STATUS=/workspace/curriculum_4h_measures_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 여러 마디 커리큘럼 시작 (4h~4i) ==="
: > "$STATUS"

STAGE_NAMES=(4h 4h2 4i)
MIN_MEASURES=(2 2 2)
MAX_MEASURES=(2 4 6)
COUNTS=(2000 3000 5000)
EPOCHS=(25 30 35)
FREEZE_EPOCHS=(8 8 10)
REPLAY_PCT=(40 40 40)
START_IDX_BASE=(2200001 2250001 2300001)
SEEDS=(220001 225001 230001)
GATE_THRESHOLDS=(78 78 78)

RESUME_FROM=${RESUME_FROM:-0}

if [ "$RESUME_FROM" -eq 0 ]; then
  PREV_NAME=4g
  PREV_OUT=/workspace/models/round1_curriculum_p2s4g
  PREV_DATA=/workspace/data/round1_stage4g_new
  PREV_COUNT=4000
  TWO_AGO_DATA=""
else
  PN=${STAGE_NAMES[$((RESUME_FROM-1))]}
  PREV_NAME="$PN"
  PREV_OUT="/workspace/models/round1_curriculum_p2s${PN}"
  PREV_DATA="/workspace/data/round1_stage${PN}_new"
  PREV_COUNT=${COUNTS[$((RESUME_FROM-1))]}
  TWO_AGO_DATA=""
  echo "=== $(date) RESUME_FROM=${RESUME_FROM} (${STAGE_NAMES[$RESUME_FROM]}부터 재개, 직전=${PREV_NAME}) ==="
fi

# 4g 데이터가 이미 정리(삭제)됐을 수 있음 -- replay 소스가 없으면 재생성해서 준비
if [ "$RESUME_FROM" -eq 0 ] && [ ! -d "$PREV_DATA" ]; then
  echo "[준비] $PREV_DATA 없음 -- 4g replay용 데이터 재생성 (4000장, 4g와 동일 스펙)"
  bash /workspace/round3train/gen_render_local.sh "$PREV_DATA" 4000 \
    --start-idx 2000001 --difficulty easy --min-measures 1 --max-measures 1 \
    --force-c-major --seed 110001
  if [ $? -ne 0 ]; then
    echo "[준비] 4g replay 데이터 재생성 실패 -- 중단"
    echo "PIPELINE_STOPPED_GENFAIL:4g_replay_prep" >> "$STATUS"
    exit 1
  fi
fi

for i in "${!STAGE_NAMES[@]}"; do
  if [ "$i" -lt "$RESUME_FROM" ]; then
    continue
  fi
  NAME=${STAGE_NAMES[$i]}
  MINM=${MIN_MEASURES[$i]}
  MAXM=${MAX_MEASURES[$i]}
  COUNT=${COUNTS[$i]}
  EP=${EPOCHS[$i]}
  FRZ=${FREEZE_EPOCHS[$i]}
  SIDX=${START_IDX_BASE[$i]}
  SEED=${SEEDS[$i]}

  NEW_DIR=/workspace/data/round1_stage${NAME}_new
  TRAIN_OUT=/workspace/models/round1_curriculum_p2s${NAME}

  echo ""
  echo "=== $(date) [$NAME] 시작 (마디 ${MINM}~${MAXM}, 신규 ${COUNT}장, epoch ${EP}, resume=${PREV_NAME}, replay=${PREV_DATA}) ==="

  if [ -n "$TWO_AGO_DATA" ] && [ -d "$TWO_AGO_DATA" ]; then
    echo "[$NAME] 디스크 정리: $TWO_AGO_DATA 삭제"
    rm -rf "$TWO_AGO_DATA"
  fi

  echo "[$NAME] quota 확인 중..."
  if ! dd if=/dev/zero of=/workspace/_qcheck bs=1M count=100 2>/dev/null; then
    echo "[$NAME] 경고: quota 초과 위험 -- 파이프라인 중단"
    rm -f /workspace/_qcheck
    echo "PIPELINE_STOPPED_QUOTA:$NAME" >> "$STATUS"
    exit 1
  fi
  rm -f /workspace/_qcheck

  EXISTING_N=$(find "$NEW_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "$EXISTING_N" -eq "$COUNT" ]; then
    echo "[$NAME] 데이터 이미 존재(${EXISTING_N}장) -- 재생성 스킵"
  else
    rm -rf "$NEW_DIR"
    GEN_ARGS=(--start-idx "$SIDX" --difficulty easy --min-measures "$MINM" --max-measures "$MAXM" --force-c-major --seed "$SEED")
    echo "[$NAME] 데이터 생성: ${GEN_ARGS[*]}"
    bash /workspace/round3train/gen_render_local.sh "$NEW_DIR" "$COUNT" "${GEN_ARGS[@]}"
    if [ $? -ne 0 ]; then
      echo "[$NAME] 데이터 생성/검증 실패 -- 파이프라인 중단"
      echo "PIPELINE_STOPPED_GENFAIL:$NAME" >> "$STATUS"
      exit 1
    fi
  fi

  RPCT=${REPLAY_PCT[$i]}
  REPLAY_COUNT=$(( PREV_COUNT * RPCT / 100 ))
  echo "[$NAME] replay: ${REPLAY_COUNT}장 (${RPCT}%, ${PREV_DATA}에서 직접, 복사 없음)"

  mkdir -p "$TRAIN_OUT"
  cd /workspace/round3train
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 -u train.py --phase 2 \
    --data_dir "$NEW_DIR" --out_dir "$TRAIN_OUT" \
    --replay_dir "$PREV_DATA" --replay_count "$REPLAY_COUNT" \
    --resume "$PREV_OUT/seq2seq_best.pt" \
    --batch 24 --epochs "$EP" --workers 16 \
    --freeze_epochs "$FRZ" --tf_ratio 1.0 --min_tf_ratio 1.0
  TRAIN_RC=$?
  if [ $TRAIN_RC -ne 0 ]; then
    echo "[$NAME] 학습 프로세스 비정상 종료(exit=$TRAIN_RC) -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_TRAINFAIL:$NAME" >> "$STATUS"
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
  echo "[$NAME] 학습 완료 -- best val_acc = ${BEST_ACC}%"

  GATE=${GATE_THRESHOLDS[$i]}
  PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
  if [ "$PASS" != "1" ]; then
    echo "[$NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%) -- 파이프라인 중단, 다음 단계 진행 안 함"
    echo "PIPELINE_STOPPED_LOW_ACC:$NAME:$BEST_ACC" >> "$STATUS"
    exit 2
  fi

  echo "[$NAME] 통과(${BEST_ACC}% >= ${GATE}%) -- 다음 단계로 진행"
  echo "STAGE_PASSED:$NAME:$BEST_ACC" >> "$STATUS"

  TWO_AGO_DATA="$PREV_DATA"
  PREV_NAME="$NAME"
  PREV_OUT="$TRAIN_OUT"
  PREV_DATA="$NEW_DIR"
  PREV_COUNT="$COUNT"
done

echo ""
echo "=== $(date) 4i까지 전체 통과 -- 여러 마디 커리큘럼 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
