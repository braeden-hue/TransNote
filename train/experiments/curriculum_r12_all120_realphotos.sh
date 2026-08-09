#!/bin/bash
# round3train/curriculum_r12_all120_realphotos.sh
#
# "실사 곡 수/사진 자체를 늘리기" 축 확장(2026-08-03) -- r8_2_diversity(60곡/904장)의
# 유일하게 검증된 성공 패턴을 120곡(exactPicture 전체, 기존 held-out 60곡 포함)/1638장
# (오선검출 불일치 33장 제외, prepare_r12_all120_realphotos.py)으로 더 확장한다.
#
# 배경: r8_2_diversity 위에 좁은 합성 데이터로 파인튜닝한 시도(r10/r10b/r11) 3연패
# ([[project_r8_2_diversity_is_best_checkpoint]]) -- 전부 held-out을 후퇴시킴. 반면
# "실사 다양성 확대" 축(r8_1->r8_2)은 유일하게 성공했던 전례라 그 축을 그대로 연장.
#
# 주의: 이후로는 이 60곡짜리 held-out 벤치마크가 학습에 포함되어 소멸함 -- 검증은
# 신규 곡(사용자 준비 중) + 120곡 자체 실측(적합도 확인, 일반화 검증 아님)으로 대체.
#
# replay: r8_1/r8_2 선례(실사 도입 라운드는 5배 비례)를 따라 1638장의 5배(~8190장) 확보.
# 기존 replay_pool_v2(2500)+v3(2000)+r7_l4_major_synth(1999)+r9_chord_accidental_synth(1997)
# 전량 합치면 8496장으로 5배 이상 -- 신규 생성 불필요.

set -uo pipefail
LOG=/workspace/curriculum_r12_all120_realphotos.log
STATUS=/workspace/curriculum_r12_all120_realphotos_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 12(실사 120곡 전체 확대) 시작 ==="
: > "$STATUS"

REAL_POOL_DIR=/workspace/round3train/data/local_pools/r12_all120_realphotos   # 120곡, 1638장
REPLAY_POOL_A=/workspace/data/replay_pool_v2
REPLAY_POOL_B=/workspace/data/replay_pool_v3
REPLAY_POOL_C=/workspace/data/r7_l4_major_synth
REPLAY_POOL_D=/workspace/data/r9_chord_accidental_synth
REPLAY_MERGED=/workspace/data/r12_replay_merged
REPLAY_A_N=2500   # 전량
REPLAY_B_N=2000   # 전량
REPLAY_C_N=1999   # 전량
REPLAY_D_N=1997   # 전량
REPLAY_COUNT=8496   # 4풀 전량 합산 -- 실사 1638장의 약 5.2배(>=5배 목표 충족)

RESUME_CKPT=/workspace/models/r8_2_diversity/seq2seq_best.pt
IN_CH=1
EXTRA_HEIGHT_STAGES=4   # r8_2_diversity와 동일(구조 변경 없음)
POOL_H=1
STAGE_NAME=r12_all120_realphotos
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

echo "[$STAGE_NAME] Replay 풀 구성 중 (v2 ${REPLAY_A_N} + v3 ${REPLAY_B_N} + r7_l4_major ${REPLAY_C_N} + r9_chord ${REPLAY_D_N}, 실제 파일 복사)..."
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
pick_and_copy "$REPLAY_POOL_C" "$REPLAY_C_N" c
pick_and_copy "$REPLAY_POOL_D" "$REPLAY_D_N" d
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가, 게다가 held-out 개념 자체가 없어짐 -- 신규곡/120곡 실측 별도 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 12(실사 120곡 전체 확대) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
