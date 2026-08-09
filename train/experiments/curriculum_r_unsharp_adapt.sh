#!/bin/bash
# round3train/curriculum_r_unsharp_adapt.sh
#
# 언샤프 마스크 질감 적응 라운드(2026-08-02) -- 실사 사진 전용 전처리(preprocess()의
# 언샤프 마스크)를 도입한 뒤 r9_chord_accidental 위에서 바로 실사(r8_1_realphotos)를
# 학습시켰더니(재학습), 기존(샤프닝 없음) 시도 대비 epoch1 Acc가 더 낮게 시작함
# (61.5% vs 72.9%) -- 사용자 지적: "실사 도메인"과 "언샤프 마스크 특유의 질감"이라는
# 두 가지 새로운 축이 동시에 도입돼 적응이 느려진 것으로 판단, 이번 세션 내내 확인된
# "한 번에 축 하나만" 원칙을 다시 적용.
#
# 실사 없이 합성만으로 이 질감(dataset.py의 apply_unsharp_like_real, augment_image에
# p_unsharp_adapt 확률로 노출)에 먼저 적응시킨다. 콘텐츠는 이미 검증된 r9의 화음/임시표
# 보강 합성 데이터를 그대로 재사용(신규 생성 없음) -- 이 라운드의 목적은 콘텐츠가 아니라
# "이 특유의 이미지 질감에 대한 인코더 적응"이므로 새 콘텐츠가 필요 없음.
#
# 사전 조건: curriculum_r9_chord_accidental.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r_unsharp_adapt.log
STATUS=/workspace/curriculum_r_unsharp_adapt_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 언샤프 마스크 질감 적응 라운드 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r9_chord_accidental" /workspace/curriculum_r9_status.txt 2>/dev/null; then
  echo "[unsharp_adapt] Round9가 STAGE_PASSED 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_R9_NOT_PASSED" >> "$STATUS"
  exit 1
fi

DATA_DIR=/workspace/data/r9_chord_accidental_synth   # 신규 생성 없음, r9 데이터 재사용
REPLAY_DIR=/workspace/data/replay_pool_v2
REPLAY_COUNT=2000

RESUME_CKPT=/workspace/models/r9_chord_accidental/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r_unsharp_adapt
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15
FREEZE_EPOCHS=0
GATE=85

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

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, replay=${REPLAY_COUNT}, noise L3/L4 + p_unsharp_adapt) ==="
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

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) 언샤프 마스크 질감 적응 라운드 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
