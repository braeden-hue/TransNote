#!/bin/bash
# round3train/curriculum_r10b_register_heavy.sh
#
# r10_register(2026-08-03) 실패 재시도. 1차 시도(메인 1500장 : replay 2000장, 거의
# 1:1.3)는 TF는 98.3%까지 깨끗하게 올라갔지만 held-out이 83.7%->76.5%로 급락하고
# newage07/09(애초 타깃)도 오히려 악화, key-C인데 없는 임시표/dynamic 환각까지 나타남
# -- 이번 세션에서 실사 도입 때 검증된 "소량 메인 + 5배 두꺼운 replay" 패턴을 안 따른
# 게 원인으로 진단. 같은 메인 데이터(register_focus_pool, 1500장)에 replay만 기존
# 검증된 합성 풀 전량(r7_l4_major_synth+r9_chord_accidental_synth+replay_pool_v2,
# 약 5.7배)으로 두껍게 깔아 재시도.
#
# RESUME은 실패한 r10_register가 아니라 r8_2_diversity(현재 최선)에서 다시 시작.

set -uo pipefail
LOG=/workspace/curriculum_r10b_register_heavy.log
STATUS=/workspace/curriculum_r10b_register_heavy_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 10b(레지스터 편향 교정, 두꺼운 replay 재시도) 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/register_focus_pool   # 1500장, 순수 합성, 대보표 전용 (재사용)
REPLAY_POOL_A=/workspace/data/r7_l4_major_synth
REPLAY_POOL_B=/workspace/data/r9_chord_accidental_synth
REPLAY_POOL_C=/workspace/data/replay_pool_v2
REPLAY_MERGED=/workspace/data/r10b_register_replay_merged
REPLAY_A_N=1999   # 풀 전량
REPLAY_B_N=1997   # 풀 전량
REPLAY_C_N=2500   # 풀 전량
REPLAY_COUNT=8496   # 실사 1500장의 약 5.7배

RESUME_CKPT=/workspace/models/r8_2_diversity/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r10b_register_heavy
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=20
FREEZE_EPOCHS=0
GATE=70

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "[$STAGE_NAME] 메인 데이터 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_DATA" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] Replay 풀 구성 중 (r7_l4_major_synth ${REPLAY_A_N} + r9_chord_accidental_synth ${REPLAY_B_N} + replay_pool_v2 ${REPLAY_C_N}, 실제 파일 복사)..."
rm -rf "$REPLAY_MERGED"
mkdir -p "$REPLAY_MERGED"
pick_and_copy () {
  local src=$1 n=$2 prefix=$3
  find "$src" -maxdepth 1 -name '*.json' -printf '%f\n' | sed 's/\.json$//' | \
    while read -r stem; do
      if [ -f "$src/$stem.png" ]; then echo "$stem"; fi
    done | shuf -n "$n" | while read -r stem; do
      cp "$src/$stem.png"  "$REPLAY_MERGED/${prefix}_$stem.png"
      cp "$src/$stem.json" "$REPLAY_MERGED/${prefix}_$stem.json"
    done
}
pick_and_copy "$REPLAY_POOL_A" "$REPLAY_A_N" a
pick_and_copy "$REPLAY_POOL_B" "$REPLAY_B_N" b
pick_and_copy "$REPLAY_POOL_C" "$REPLAY_C_N" c
REPLAY_N=$(find "$REPLAY_MERGED" -maxdepth 1 -name '*.png' | wc -l)
echo "[$STAGE_NAME] Replay 풀 완성: ${REPLAY_N}장"

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
  --replay_dir "$REPLAY_MERGED" --replay_count "$REPLAY_COUNT" \
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
echo "=== $(date) Round 10b(레지스터 편향 교정, 두꺼운 replay) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
