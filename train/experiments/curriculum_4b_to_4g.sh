#!/bin/bash
# 4b~4g duration 커리큘럼 자동 파이프라인.
# 각 단계: quota 확인 -> 데이터 생성/렌더링/검증(gen_render_local.sh) -> 학습(직전 단계 REPLAY_PCT% replay,
# 복사 없이 --replay_dir로 직접 읽음) -> 정확도 게이트(best val_acc >= 90%) -> 통과 시 다음 단계.
# 게이트 실패 시 그 자리에서 멈추고 상태만 로그에 남김(pod 종료는 오케스트레이터가 로그 보고 수행).
# RESUME_FROM 환경변수로 중간 단계부터 재개 가능 (예: RESUME_FROM=1 bash curriculum_4b_to_4g.sh 로 4c부터 시작).
#
# 사전 조건(직접 확인 후 실행할 것):
#   - cv2/music21/xvfb/libEGL 등 설치 확인 (pod 재시작 시 초기화됨)
#   - round3train/{dataset.py,train.py,generate_scores.py,validate_stage4_data.py,gen_render_local.sh} 최신본 배포됨
#   - round1_curriculum_p2s4_quarteronly(4a) 체크포인트 + round1_stage4_quarteronly(4a 데이터) 존재

set -uo pipefail
LOG=/workspace/curriculum_pipeline.log
STATUS=/workspace/curriculum_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 커리큘럼 파이프라인 시작 (4b~4g) ==="
: > "$STATUS"

# 각 신규 duration을 "쉼표 배제 -> 길이만 순수 학습" 후 "쉼표 복원 -> note/rest 판별" 2단계로 분리.
# (4c는 이미 이 원칙 적용됨; 4d/4e/4f도 동일 패턴, N2 단계가 쉼표 복원)
STAGE_NAMES=(4b 4c 4c2 4d 4d2 4e 4e2 4f 4f2 4g)
DURATION_SUBSETS=(
  "1/4,1/2,1/1"
  "1/4,1/2,1/1,1/8"
  "1/4,1/2,1/1,1/8"
  "1/4,1/2,1/1,1/8,1/16"
  "1/4,1/2,1/1,1/8,1/16"
  "1/4,1/2,1/1,1/8,1/16,3/8"
  "1/4,1/2,1/1,1/8,1/16,3/8"
  "1/4,1/2,1/1,1/8,1/16,3/8,3/16"
  "1/4,1/2,1/1,1/8,1/16,3/8,3/16"
  ""
)
COUNTS=(2000 2000 2000 2000 2000 2000 2000 2000 2000 4000)
EPOCHS=(25 25 25 25 25 25 25 25 25 40)
FREEZE_EPOCHS=(5 7 7 7 7 8 8 8 8 12)
SS_EPOCHS=(0 0 0 0 0 0 0 0 0 20)
MIN_TF=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.5)
REPLAY_PCT=(30 50 40 40 40 40 40 40 40 40)
NO_RESTS=(0 1 0 1 0 1 0 1 0 0)
START_IDX_BASE=(600001 800002 850001 1000001 1050001 1200001 1250001 1600001 1650001 2000001)
SEEDS=(60001 70002 70502 80001 80501 90001 90501 100001 100501 110001)
# 단계가 누적될수록 요구 정확도를 90%->78%(바닥)까지 완만히 낮춤
# (새 duration 도입 단계와 그 뒤 쉼표 복원 단계는 같은 난이도 레벨로 취급해 동일 임계값 적용)
GATE_THRESHOLDS=(90 87 87 84 84 81 81 78 78 78)

# 재개 지점: 0=4b부터, 1=4c부터(4b는 이미 통과함) 등. 기본 0(처음부터).
RESUME_FROM=${RESUME_FROM:-0}

if [ "$RESUME_FROM" -eq 0 ]; then
  PREV_NAME=4a
  PREV_OUT=/workspace/models/round1_curriculum_p2s4_quarteronly
  PREV_DATA=/workspace/data/round1_stage4_quarteronly
  PREV_COUNT=3000
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

for i in "${!STAGE_NAMES[@]}"; do
  if [ "$i" -lt "$RESUME_FROM" ]; then
    continue
  fi
  NAME=${STAGE_NAMES[$i]}
  DSUB=${DURATION_SUBSETS[$i]}
  COUNT=${COUNTS[$i]}
  EP=${EPOCHS[$i]}
  FRZ=${FREEZE_EPOCHS[$i]}
  SS=${SS_EPOCHS[$i]}
  MTF=${MIN_TF[$i]}
  SIDX=${START_IDX_BASE[$i]}
  SEED=${SEEDS[$i]}
  NOREST=${NO_RESTS[$i]}

  NEW_DIR=/workspace/data/round1_stage${NAME}_new
  TRAIN_OUT=/workspace/models/round1_curriculum_p2s${NAME}

  echo ""
  echo "=== $(date) [$NAME] 시작 (신규 ${COUNT}장, epoch ${EP}, resume=${PREV_NAME}, replay=${PREV_DATA}) ==="

  # --- 디스크 정리: 더 이상 아무도 참조하지 않는 2단계 전 데이터 삭제 ---
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

  # --- 데이터 생성+렌더링+검증 (로컬 파이프라인, 이미 존재하면 재사용) ---
  EXISTING_N=$(find "$NEW_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "$EXISTING_N" -eq "$COUNT" ]; then
    echo "[$NAME] 데이터 이미 존재(${EXISTING_N}장) -- 재생성 스킵"
  else
    rm -rf "$NEW_DIR"
    GEN_ARGS=(--start-idx "$SIDX" --difficulty easy --min-measures 1 --max-measures 1 --force-c-major --seed "$SEED")
    if [ "$NOREST" -eq 1 ]; then
      GEN_ARGS+=(--no-rests)
    fi
    if [ -n "$DSUB" ]; then
      GEN_ARGS+=(--duration-subset "$DSUB")
    fi
    echo "[$NAME] 데이터 생성: ${GEN_ARGS[*]}"
    bash /workspace/round3train/gen_render_local.sh "$NEW_DIR" "$COUNT" "${GEN_ARGS[@]}"
    if [ $? -ne 0 ]; then
      echo "[$NAME] 데이터 생성/검증 실패 -- 파이프라인 중단"
      echo "PIPELINE_STOPPED_GENFAIL:$NAME" >> "$STATUS"
      exit 1
    fi
  fi

  # --- replay 수 계산 (직전 단계 신규 개수의 REPLAY_PCT%) ---
  RPCT=${REPLAY_PCT[$i]}
  REPLAY_COUNT=$(( PREV_COUNT * RPCT / 100 ))
  echo "[$NAME] replay: ${REPLAY_COUNT}장 (${RPCT}%, ${PREV_DATA}에서 직접, 복사 없음)"

  # --- 학습 ---
  mkdir -p "$TRAIN_OUT"
  cd /workspace/round3train
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 -u train.py --phase 2 \
    --data_dir "$NEW_DIR" --out_dir "$TRAIN_OUT" \
    --replay_dir "$PREV_DATA" --replay_count "$REPLAY_COUNT" \
    --resume "$PREV_OUT/seq2seq_best.pt" \
    --batch 24 --epochs "$EP" --workers 16 \
    --freeze_epochs "$FRZ" --tf_ratio 1.0 --min_tf_ratio "$MTF" --ss_epochs "$SS"
  TRAIN_RC=$?
  if [ $TRAIN_RC -ne 0 ]; then
    echo "[$NAME] 학습 프로세스 비정상 종료(exit=$TRAIN_RC) -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_TRAINFAIL:$NAME" >> "$STATUS"
    exit 1
  fi

  # --- 정확도 게이트 ---
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
echo "=== $(date) 4g까지 전체 통과 -- 파이프라인 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
