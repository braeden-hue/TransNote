#!/bin/bash
# round3train/curriculum_r8_1_unsharp.sh
#
# r8_1_r9와 완전히 동일한 "작은 실사(40곡/516장) + 두꺼운 5배 replay(2580장)" 레시피를,
# RESUME만 r_unsharp_adapt(언샤프 마스크 질감에 합성으로 미리 적응 완료, Acc 95.6%)로
# 바꿔서 시도(2026-08-02). 목적: "실사 도메인"과 "언샤프 마스크 질감"이라는 두 축을
# 분리했을 때 실사 도입 초반 적응이 더 빨라지는지(기존 r8_1_r9 재학습 시도에서 언샤프
# 마스크를 실사와 동시에 도입하니 epoch1 Acc가 61.5%로 더 낮게 시작했던 문제) 확인.
#
# resume 체크포인트 외 모든 조건(메인 실사 40곡/516장, replay 구성/비율, 노이즈 설정,
# 에폭 수)은 r8_1_r9와 완전히 동일 -- 이번 변경의 순수 효과만 보기 위함.
#
# 사전 조건: curriculum_r_unsharp_adapt.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r8_1_unsharp.log
STATUS=/workspace/curriculum_r8_1_unsharp_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 8 1단계(언샤프 적응 기반, 쉬운 40곡, 두꺼운 replay) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r_unsharp_adapt" /workspace/curriculum_r_unsharp_adapt_status.txt 2>/dev/null; then
  echo "[r8_1_unsharp] 언샤프 적응 라운드가 STAGE_PASSED 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_UNSHARP_NOT_PASSED" >> "$STATUS"
  exit 1
fi

REAL_POOL_DIR=/workspace/data/r8_1_realphotos   # 40곡(클래식 30+뉴에이지 10), 516장
REPLAY_POOL_A=/workspace/data/r9_chord_accidental_synth
REPLAY_POOL_B=/workspace/data/replay_pool_v2
REPLAY_MERGED=/workspace/data/r8_1_unsharp_replay_merged
REPLAY_A_N=1290
REPLAY_B_N=1290
REPLAY_COUNT=2580   # 실사 516장의 5배

RESUME_CKPT=/workspace/models/r_unsharp_adapt/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r8_1_unsharp
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=20
FREEZE_EPOCHS=0
GATE=70

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$REAL_POOL_DIR" ]; then
  echo "[$STAGE_NAME] 실사 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_REALPHOTOS" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] Replay 풀 구성 중 (r9_chord_accidental_synth ${REPLAY_A_N} + replay_pool_v2 ${REPLAY_B_N}, 실제 파일 복사)..."
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
pick_and_copy "$REPLAY_POOL_A" "$REPLAY_A_N" a
pick_and_copy "$REPLAY_POOL_B" "$REPLAY_B_N" b
REPLAY_N=$(find "$REPLAY_MERGED" -maxdepth 1 -name '*.png' | wc -l)
echo "[$STAGE_NAME] Replay 풀 완성: ${REPLAY_N}장"

REAL_N=$(find "$REAL_POOL_DIR" -maxdepth 1 \( -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, 메인=실사 ${REAL_N}장, replay=${REPLAY_COUNT}) ==="
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
echo "[$STAGE_NAME] 주의: 메인 실사(516)<replay(2580)라 val_acc는 replay 합성에 크게 좌우됨 -- newage 나머지 10곡(검증 전용) 실측 전까지는 참고용"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 8 1단계(언샤프 적응 기반) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
