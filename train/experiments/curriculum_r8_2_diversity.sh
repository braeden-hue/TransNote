#!/bin/bash
# round3train/curriculum_r8_2_diversity.sh
#
# r8_1_realphotos(클래식 30+뉴에이지 10=40곡/516장)의 "소량 실사 + 두꺼운 5배 replay"
# 레시피를 그대로 유지한 채, 메인 실사 데이터만 클래식 50+뉴에이지 10=60곡/904장으로
# 확대해서 "실사 데이터 다양성/수량 자체가 held-out 정확도의 병목인가"를 단독으로
# 테스트한다(2026-08-02). replay 비율/노이즈 설정/에폭 수 등 다른 조건은 전부 동일하게
# 유지 -- 지금까지 세션 내내 검증된 "한 번에 축 하나만" 원칙 적용.
#
# 배경: r8_1/r9-variant/unsharp 등 지금까지의 모든 시도가 held-out(뉴에이지 미학습
# 10곡) 정확도 54~58%에서 정체됨. 사용자가 "클래식 50곡 선별 + 반복 다회 학습"을
# 제안했으나, replay까지 동시에 대폭 축소하면 두 축이 겹쳐 원인 분리가 안 되고
# catastrophic forgetting 위험도 커짐 -- 검토 후 replay는 기존 5배 수준 유지로 합의,
# "실사 다양성 확대"라는 축 하나만 우선 테스트하기로 함.
#
# replay 소스 두 풀(r7_l4_major_synth 1999장, replay_pool_v2 2500장) 전량 사용 --
# 904장의 약 5배(4499장)에 해당.
#
# 사전 조건: curriculum_r7b_l4_synth_only.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r8_2_diversity.log
STATUS=/workspace/curriculum_r8_2_diversity_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 8 2단계(실사 다양성 확대, 60곡/904장, replay 5배 유지) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r7b_l4_synth_only" /workspace/curriculum_r7b_status.txt 2>/dev/null; then
  echo "[r8_2_diversity] Round7(재정의)가 STAGE_PASSED 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_R7B_NOT_PASSED" >> "$STATUS"
  exit 1
fi

REAL_POOL_DIR=/workspace/data/r8_2_realphotos   # 60곡(클래식 50+뉴에이지 10), 904장
REPLAY_POOL_A=/workspace/data/r7_l4_major_synth
REPLAY_POOL_B=/workspace/data/replay_pool_v2
REPLAY_MERGED=/workspace/data/r8_2_diversity_replay_merged
REPLAY_A_N=1999   # 풀 전량
REPLAY_B_N=2500   # 풀 전량
REPLAY_COUNT=4499   # 실사 904장의 약 5배

RESUME_CKPT=/workspace/models/r7b_l4_synth_only/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r8_2_diversity
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

echo "[$STAGE_NAME] Replay 풀 구성 중 (r7_l4_major_synth ${REPLAY_A_N} + replay_pool_v2 ${REPLAY_B_N}, 실제 파일 복사)..."
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- newage 나머지 10곡(검증 전용)으로 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 8 2단계(실사 다양성 확대) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
