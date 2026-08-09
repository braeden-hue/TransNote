#!/bin/bash
# round3train/curriculum_r15_cropfix_coordconv.sh
#
# 크롭 로직 수정(r14_cropfix) 위에 CoordConv(r13_coordconv, 다른 축)를 결합(2026-08-03).
# 신규 6곡 검증 결과 두 축이 서로 다른 곡에서 개선을 보임:
#   r12(기준) 76.9% -> r13(CoordConv, r12에서 분기) 79.1% -> r14(크롭수정, r12에서 분기) 78.0%
#   r14는 크롭 위치 편차가 컸던 sonatine_22_30(+25.5pp)/23_42(+7.2pp)에서 크게 개선됐지만
#   sonatine_23_38에서 -22.0pp 급락(r14 스크립트 주석에 이미 적어둔 우려, "입력표현이 바뀌므로
#   가벼운 파인튜닝만으론 부족할 수 있음"이 실현된 것으로 추정) -- 순효과는 +1.1pp에 그침.
# "한 번에 축 하나만" 원칙에 따라 두 축을 순차 결합해서(r13+r14를 동시에 새로 만드는 게 아니라
# r14 위에 CoordConv만 추가) 상쇄 없이 두 개선이 겹치는지 확인.
#
# r13_coordconv와 동일하게 in_ch만 1->2로 바꾸는 최소 변경(첫 conv 레이어 재초기화, 나머지
# 유지) -- dataset.py의 make_model_input(in_ch=2)/extract_staff_canvas 크롭 수정이 이미
# 적용된 상태이므로 데이터 재생성 불필요.

set -uo pipefail
LOG=/workspace/curriculum_r15_cropfix_coordconv.log
STATUS=/workspace/curriculum_r15_cropfix_coordconv_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 15(크롭수정+CoordConv 결합) 시작 ==="
: > "$STATUS"

REAL_POOL_DIR=/workspace/round3train/data/local_pools/r12_all120_realphotos   # r12/r13/r14와 동일 120곡/1638장
REPLAY_DIR=/workspace/data/r12_replay_merged   # r12/r13/r14와 동일 재사용(8486장)
REPLAY_COUNT=8486

RESUME_CKPT=/workspace/models/r14_cropfix/seq2seq_best.pt   # 크롭수정 체크포인트에서 재개
IN_CH=2   # CoordConv 추가 -- 유일한 구조 변경
EXTRA_HEIGHT_STAGES=4
POOL_H=1
STAGE_NAME=r15_cropfix_coordconv
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15
FREEZE_EPOCHS=0
GATE=70

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$REAL_POOL_DIR" ] || [ ! -d "$REPLAY_DIR" ]; then
  echo "[$STAGE_NAME] 데이터/replay 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_DATA" >> "$STATUS"
  exit 1
fi

REAL_N=$(find "$REAL_POOL_DIR" -maxdepth 1 \( -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, in_ch=${IN_CH}, 메인=실사 ${REAL_N}장, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$REAL_POOL_DIR" --out_dir "$TRAIN_OUT" \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- 신규 검증곡으로 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 15(크롭수정+CoordConv 결합) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
