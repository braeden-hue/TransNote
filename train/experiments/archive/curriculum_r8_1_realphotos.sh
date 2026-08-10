#!/bin/bash
# round3train/curriculum_r8_1_realphotos.sh
#
# Round8 2단계 커리큘럼의 1단계(2026-08-02) -- 클래식 100곡 전체를 한 번에 투입했던
# 기존 방식(v1/v2b/v2c/v2d/v2e)이 전부 에폭마다 심하게 출렁이며 v1의 77.8%를 못 넘는
# 문제 확인. LR을 낮춰도(v2e, LR 6e-5) 진폭만 v1 수준으로 줄었을 뿐 최종 성능은
# 그대로. 해상도(9배 차이지만 최악도 1.9배 업스케일 수준)와 오선 확대율(unit_size,
# 실사 CV=0.36 < 합성 CV=0.79로 오히려 합성이 더 다양)도 원인이 아닌 것으로 확인.
# 콘텐츠 난이도(임시표/화음/빠른리듬)도 클래식-뉴에이지-합성 간 극단적 차이는 없었음.
#
# 사용자 제안: "실사 데이터 규모를 작게 + replay를 훨씬 두껍게(5배)" 해서 도메인 충격을
# 완만하게 만드는 게 핵심 메커니즘일 것 -- 1단계는 난이도 하위 30곡(클래식) + 쉬운
# 10곡(뉴에이지, 이번이 최초로 학습에 포함됨) = 40곡/516장만 메인으로 쓰고, replay를
# 5배(2580장, r7_l4_major_synth+replay_pool_v2)로 두껍게 깔아 실사 비중을 낮춘다.
# 뉴에이지 나머지 10곡(03,04,05,06,07,09,11,14,19,20)은 계속 검증 전용 유지.
#
# 사전 조건: curriculum_r7_l4_synth_only.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r8_1.log
STATUS=/workspace/curriculum_r8_1_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 8 1단계(쉬운 40곡, 두꺼운 replay) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r7b_l4_synth_only" /workspace/curriculum_r7b_status.txt 2>/dev/null; then
  echo "[r8_1] Round7(재정의)가 STAGE_PASSED 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_R7B_NOT_PASSED" >> "$STATUS"
  exit 1
fi

REAL_POOL_DIR=/workspace/data/r8_1_realphotos   # 40곡(클래식 30+뉴에이지 10), 516장
REPLAY_POOL_A=/workspace/data/r7_l4_major_synth
REPLAY_POOL_B=/workspace/data/replay_pool_v2
REPLAY_MERGED=/workspace/data/r8_1_replay_merged
REPLAY_A_N=1290
REPLAY_B_N=1290
REPLAY_COUNT=2580   # 실사 516장의 5배

RESUME_CKPT=/workspace/models/r7b_l4_synth_only/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r8_1_realphotos
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
echo "=== $(date) Round 8 1단계 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
