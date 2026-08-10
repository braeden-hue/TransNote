#!/bin/bash
# round3train/curriculum_r8_realphotos.sh
#
# Round8(2026-08-02) -- Round7(재정의, L4 안정화 완료)에서 이어받아 클래식 실사 사진
# (86곡, 1550장)만 단독으로 메인 데이터에 투입(합성 안 섞음 -- "실사 도입"이라는 변화
# 하나만 격리해서 dataset.py의 오선검출 버그 수정 효과를 깨끗하게 확인하기 위함).
#
# 사전 조건: curriculum_r7_l4_synth_only.sh가 STAGE_PASSED로 끝나있어야 함.
# newage(15곡)는 검증 전용으로 완전히 제외 -- 학습에 절대 포함하지 않음.

set -uo pipefail
LOG=/workspace/curriculum_r8.log
STATUS=/workspace/curriculum_r8_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 8(실사 사진 단독 도입) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r7b_l4_synth_only" /workspace/curriculum_r7b_status.txt 2>/dev/null; then
  echo "[r8] Round7(재정의)가 STAGE_PASSED 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_R7B_NOT_PASSED" >> "$STATUS"
  exit 1
fi

REAL_POOL_DIR=/workspace/data/classical_realphotos   # 1550장, 오선검출 버그 수정 반영됨
REPLAY_POOL_OLD=/workspace/data/replay_pool_v2
REPLAY_POOL_R7B=/workspace/data/r7_l4_major_synth   # Round7(재정의)가 학습한 데이터
REPLAY_MERGED=/workspace/data/r8_replay_merged
REPLAY_OLD_N=1200
REPLAY_R7B_N=800
REPLAY_COUNT=2000

RESUME_CKPT=/workspace/models/r7b_l4_synth_only/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r8_realphotos
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=20
FREEZE_EPOCHS=0
GATE=75

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$REAL_POOL_DIR" ]; then
  echo "[$STAGE_NAME] 클래식 실사 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_REALPHOTOS" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] Replay 풀 구성 중 (이전 라운드 ${REPLAY_OLD_N} + 직전 r7b ${REPLAY_R7B_N}, 실제 파일 복사)..."
rm -rf "$REPLAY_MERGED"
mkdir -p "$REPLAY_MERGED"
pick_and_copy() {
  local src=$1 n=$2 prefix=$3
  find "$src" -maxdepth 1 -name '*.json' -printf '%f\n' | sed 's/\.json$//' | \
    while read -r stem; do
      if [ -f "$src/$stem.png" ]; then echo "$stem"; fi
    done | shuf -n "$n" | while read -r stem; do
      cp "$src/$stem.png"  "$REPLAY_MERGED/${prefix}_$stem.png"
      cp "$src/$stem.json" "$REPLAY_MERGED/${prefix}_$stem.json"
    done
}
pick_and_copy "$REPLAY_POOL_OLD" "$REPLAY_OLD_N" old
pick_and_copy "$REPLAY_POOL_R7B" "$REPLAY_R7B_N" r7b
REPLAY_N=$(find "$REPLAY_MERGED" -maxdepth 1 -name '*.png' | wc -l)
echo "[$STAGE_NAME] Replay 풀 완성: ${REPLAY_N}장"

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, replay=${REPLAY_COUNT}, 메인=실사${REAL_POOL_DIR}만) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$REAL_POOL_DIR" --out_dir "$TRAIN_OUT" \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- newage 15곡(학습 미포함, 검증 전용) 실사 오류분석으로 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 8 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
