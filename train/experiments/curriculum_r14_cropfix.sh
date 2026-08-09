#!/bin/bash
# round3train/curriculum_r14_cropfix.sh
#
# 오선 크롭/스케일 로직 근본 수정 단독 검증(2026-08-03). extract_staff_canvas/
# extract_system_canvas가 크롭 전체 높이(레저선·임시표 등 부수 콘텐츠 양에 따라
# 마디마다 다름) 기준으로 스케일+중앙정렬하던 것을, 오선 unit_size 기준 고정 스케일+
# 오선 첫 줄 고정 앵커 위치로 수정(dataset.py STAFF_UNIT_PX/STAFF_ANCHOR_TOP_PX 등,
# CANVAS_H/MARGIN_UNITS 설계 의도 "4+2*3=10 units"를 실제로 구현). 신규 6곡 표본에서
# 오선 상단 위치 편차가 33px -> 최대 12px(대부분 0~4px)로 줄어드는 것을 확인.
#
# CoordConv(r13_coordconv, 동시 진행 중, 다른 축)와 섞이지 않게 "한 번에 축 하나만"
# 원칙에 따라 r12_all120_realphotos(크롭 수정 전 학습된 현재 최선)에서 그대로 재개,
# in_ch=1(CoordConv 없음), 그 외 데이터/replay/노이즈 설정 전부 r12와 동일 -- 크롭 로직
# 수정 자체의 효과만 분리해서 본다. dataset.py의 _pre.npy/_staffs.json 캐시는 좌표만
# 캐싱하고 최종 캔버스는 매번 새로 계산하므로, 데이터 재생성 없이 바로 적용됨.
#
# 다만 시각적 입력 표현이 근본적으로 바뀌므로(모든 기존 체크포인트는 구 스케일/위치
# 관례로 학습됨) 가벼운 파인튜닝만으론 부족할 수 있음 -- 결과를 보고 필요시 에폭 수를
# 늘리거나 재학습 범위를 넓힐 것.

set -uo pipefail
LOG=/workspace/curriculum_r14_cropfix.log
STATUS=/workspace/curriculum_r14_cropfix_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 14(오선 크롭 로직 수정 단독 검증) 시작 ==="
: > "$STATUS"

REAL_POOL_DIR=/workspace/round3train/data/local_pools/r12_all120_realphotos   # r12와 동일 120곡/1638장
REPLAY_DIR=/workspace/data/r12_replay_merged   # r12/r13와 동일 재사용(8486장)
REPLAY_COUNT=8486

RESUME_CKPT=/workspace/models/r12_all120_realphotos/seq2seq_best.pt   # 크롭 수정 전 체크포인트에서 재개
IN_CH=1   # CoordConv 없음 -- 크롭 수정 단독 효과만 분리
EXTRA_HEIGHT_STAGES=4
POOL_H=1
STAGE_NAME=r14_cropfix
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=20
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
echo "=== $(date) Round 14(오선 크롭 로직 수정 단독 검증) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
