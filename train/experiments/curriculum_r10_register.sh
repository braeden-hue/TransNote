#!/bin/bash
# round3train/curriculum_r10_register.sh
#
# 레지스터 편향(clef register bias) 교정 라운드(2026-08-03). newage07/09 held-out
# 오류 분석에서, 베이스보표가 레저선이 필요한 음역(아래로 많이 벗어나거나 위로
# 치보표에 근접)을 읽을 때 실제 위치 대신 "베이스클렙의 전형적 음역"으로 회귀하는
# 패턴 확인(newage09: G4를 B2로, 2옥타브 이상 오독) -- project_register_bias_failure.md
# 에 2026-07-28부터 기록된 미해결 이슈. generate_scores.py의 CROSS_REGISTER_PROB
# 인프라는 이미 있었지만 어느 라운드도 실제로 켠 적이 없었음(기본값 0.0).
#
# pod_gen_register_focus.sh로 cross-register-prob=0.45로 새로 생성한 1500장(순수
# 합성, 대보표 전용)을 메인으로, 기존 replay_pool_v2로 일반화 유지. r8_2_diversity
# (현재까지 최고 체크포인트, 오늘 오전 파이프라인 버그 6건 수정 후 재검증 시
# held-out 74.6%, 조표 후처리 적용 시 83.7%)에서 재개.
#
# 사전 조건: pod_gen_register_focus.sh가 GEN_COMPLETE로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r10_register.log
STATUS=/workspace/curriculum_r10_register_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 10(레지스터 편향 교정) 시작 ==="
: > "$STATUS"

if ! grep -q "^GEN_COMPLETE" /workspace/gen_register_focus_status.txt 2>/dev/null; then
  echo "[r10_register] 레지스터 집중 데이터 생성이 GEN_COMPLETE 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_GEN_NOT_COMPLETE" >> "$STATUS"
  exit 1
fi

DATA_DIR=/workspace/round3train/data/register_focus_pool   # 1500장, 순수 합성, 대보표 전용
REPLAY_DIR=/workspace/data/replay_pool_v2
REPLAY_COUNT=2000

RESUME_CKPT=/workspace/models/r8_2_diversity/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r10_register
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15
FREEZE_EPOCHS=0
GATE=80

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$DATA_DIR" ] || [ ! -d "$REPLAY_DIR" ]; then
  echo "[$STAGE_NAME] 데이터/replay 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_DATA" >> "$STATUS"
  exit 1
fi

DATA_N=$(find "$DATA_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, 메인=합성 ${DATA_N}장, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --page_level_noise --noise_level 3 --noise_level_max 4 --p_level_max 0.5 \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- newage held-out으로 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 10(레지스터 편향 교정) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
