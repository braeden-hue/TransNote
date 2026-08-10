#!/bin/bash
# 단일 오선 인식 복원 커리큘럼 (5ss1): Stage1~3(초반)에는 단일 오선만 학습해 Acc 97~100%를
# 찍었으나, Stage4부터 5n5까지 전부 대보표 전용으로만 이어지며(4gchord~4gm~4t~4tup~4span_w10~
# 5n4~5n5 어떤 커리큘럼 셸 스크립트에도 --single-staff가 쓰인 적 없음, grep으로 확인) 잊어버림.
# 2026-07-26 로컬 재확인: 5n4/5n5 둘 다 단일 오선 20장 exact_match 0/20.
#
# dataset.py의 omr_collate()가 애초에 "단일 오선과 대보표가 섞일 수 있다"는 전제로 설계돼
# 있어(서로 다른 캔버스 높이를 max_H로 white-pad) 코드 수정 없이 데이터 풀 구성비만으로 해결
# 가능 -- OMRDataset.__init__도 n_staffs 홀짝으로 이미 단일/대보표 분기 처리함.
#
# 대보표(40%)는 REPLAY 목적(망각 방지) -- 현재 확정 데모 스코프(2026-07-22, artic/ornament/
# slur/ottava/hairpin 제외)를 그대로 유지해 이번 단계가 대보표 인식률에 영향 주지 않게 함.
# 단일 오선(60%)이 이번 단계의 실제 학습 목표.
#
# resume은 5n5(5n6과 달리 5n4 아님) -- 이건 노이즈 강건성과 직교하는 축이라 최신 지점에서
# 이어감. 5n6(curl 노이즈 지속)과는 독립적으로 진행 가능(순서 무관, 서로 다른 축).

set -uo pipefail
LOG=/workspace/curriculum_5ss.log
STATUS=/workspace/curriculum_5ss_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 단일 오선 복원 커리큘럼 시작 (5ss1) ==="
: > "$STATUS"

POOL_DIR=/workspace/data/round1_stage5ss1_pool
GRAND_COUNT=1600     # 40% -- 대보표 replay
SINGLE_COUNT=2400    # 60% -- 단일 오선 (이번 단계 목표)
GRAND_START_IDX=3700001
SINGLE_START_IDX=3900001
SEED=306001

RESUME_NAME=5n5
RESUME_OUT=/workspace/models/round1_curriculum_p2s5n5
STAGE_NAME=5ss1
TRAIN_OUT=/workspace/models/round1_curriculum_p2s${STAGE_NAME}
EPOCHS=15
FREEZE_EPOCHS=2
NOISE_LEVEL=2       # page_level_noise 없이 캔버스 레벨만 (이번 단계는 오선 개수 혼합에만 집중)
GATE=90             # 5n5/5n6의 65%(노이즈 구조적 하한)와 달리 표준 심볼 커리큘럼 기준 적용
SPAN_WEIGHT=1       # 옥타브/헤어핀 생성 안 하므로 무의미(5n5/5n6와 동일)

# 공통 스코프 인자(현재 확정 데모 스코프, 2026-07-22) -- 대보표/단일오선 둘 다 동일 적용
COMMON_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.08 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04)

echo "[5ss1] 데이터 풀 확인 중..."
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
TOTAL_COUNT=$((GRAND_COUNT + SINGLE_COUNT))
MIN_OK_N=$(( TOTAL_COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[5ss1] 데이터 풀 이미 존재(${EXISTING_N}/${TOTAL_COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"

  echo "[5ss1] 대보표(40%, ${GRAND_COUNT}장) 생성 중..."
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$GRAND_COUNT" \
    "${COMMON_ARGS[@]}" --density-break --ottava-prob 0 --hairpin-prob 0 \
    --start-idx "$GRAND_START_IDX" --seed "$SEED"
  if [ $? -ne 0 ]; then
    echo "[5ss1] 대보표 데이터 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:grand" >> "$STATUS"
    exit 1
  fi

  echo "[5ss1] 단일 오선(60%, ${SINGLE_COUNT}장) 생성 중..."
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$SINGLE_COUNT" \
    "${COMMON_ARGS[@]}" --single-staff \
    --start-idx "$SINGLE_START_IDX" --seed "$((SEED + 1))"
  if [ $? -ne 0 ]; then
    echo "[5ss1] 단일 오선 데이터 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:single" >> "$STATUS"
    exit 1
  fi

  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[5ss1] 데이터 풀 완성: ${FINAL_N}/${TOTAL_COUNT}장 (대보표 ${GRAND_COUNT} + 단일오선 ${SINGLE_COUNT})"
fi

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=${RESUME_NAME}, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}) ==="

echo "[$STAGE_NAME] quota 확인 중..."
if ! dd if=/dev/zero of=/workspace/_qcheck bs=1M count=100 2>/dev/null; then
  echo "[$STAGE_NAME] 경고: quota 초과 위험 -- 파이프라인 중단"
  rm -f /workspace/_qcheck
  echo "PIPELINE_STOPPED_QUOTA:$STAGE_NAME" >> "$STATUS"
  exit 1
fi
rm -f /workspace/_qcheck

mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
  --resume "$RESUME_OUT/seq2seq_best.pt" \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS" --noise_level "$NOISE_LEVEL" --span_weight "$SPAN_WEIGHT"
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[$STAGE_NAME] 학습 프로세스 비정상 종료(exit=$TRAIN_RC) -- 파이프라인 중단"
  echo "PIPELINE_STOPPED_TRAINFAIL:$STAGE_NAME" >> "$STATUS"
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
echo "[$STAGE_NAME] 학습 완료 -- best val_acc = ${BEST_ACC}% (gate=${GATE}%)"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%) -- 오류 분석 실행"
  echo "PIPELINE_STOPPED_LOW_ACC:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
  python3 /workspace/round3train/error_breakdown.py \
    --seq2seq "$TRAIN_OUT/seq2seq_best.pt" \
    --tokenizer /workspace/round3train/tokenizer258.json \
    --data_dir "$POOL_DIR" \
    > "$TRAIN_OUT/error_breakdown.log" 2>&1
  echo "[$STAGE_NAME] 오류 분석 결과: $TRAIN_OUT/error_breakdown.log 확인"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) 5ss1 완료 -- 단일 오선 복원 커리큘럼 종료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
