#!/bin/bash
# round3train/curriculum_r4_noise.sh
#
# Round4 -- 노이즈 증강(촬영 조건 강건성). Round1v2/2v2/3v2는 전부 --no_augment로
# 콘텐츠 분포만 다뤘고 노이즈에는 완전히 무방비였음(project.md 기록: 같은 체크포인트로
# 깨끗한 렌더링 72.2% vs 실제 폰카메라 사진 21.8% -- 순수 노이즈 강건성 부재).
#
# 신규 데이터 생성 없음 -- augment_image()가 dataset.py에서 매 배치 즉석으로 랜덤
# 노이즈(기울기/블러/조명/JPEG압축)를 적용하는 방식이라, 콘텐츠는 Round3 v2까지 이미
# 잘 잡혀있는 기존 풀을 그대로 재사용하고 --no_augment만 빼면 됨.
#
# 데이터 규모는 Round3 v2에서 8000장(신규5000+replay3000)일 때 OOM(oom_kill)이 났던
# 전례가 있어 6500장(신규5000+replay1500) 규모로 유지 -- Round2v2/3v2가 이 규모에서
# 문제없이 성공했음.
#
# 사전 조건: curriculum_r3_v2_mixed.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r4.log
STATUS=/workspace/curriculum_r4_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 4(노이즈 증강, 신규 생성 없음) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r3_v2_mixed" /workspace/curriculum_r3_v2_status.txt 2>/dev/null; then
  echo "[r4_noise] Round3 v2가 STAGE_PASSED 상태가 아님 -- 중단(먼저 확인 필요)"
  echo "PIPELINE_STOPPED_R3_NOT_PASSED" >> "$STATUS"
  exit 1
fi

DATA_DIR=/workspace/data/r3_v2_mixed
R1_POOL_DIR=/workspace/data/r1_v2_foundation
R2_POOL_DIR=/workspace/data/r2_v2_grandstaff
R4_REPLAY_MERGED=/workspace/data/r4_replay_merged
R1_REPLAY_N=500
R2_REPLAY_N=1000
REPLAY_COUNT=1500

RESUME_CKPT=/workspace/models/r3_v2_mixed/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r4_noise
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15
FREEZE_EPOCHS=0
GATE=85
# 노이즈 낀 이미지는 깨끗한 렌더링보다 근본적으로 어려워서 게이트를 90->85로 낮춤
# (TF 기준이라 어차피 참고용, 실측 재검증이 진짜 판단 기준).

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] Replay 풀 구성 중 (R1 ${R1_REPLAY_N}+R2 ${R2_REPLAY_N}, 심볼릭 링크)..."
rm -rf "$R4_REPLAY_MERGED"
mkdir -p "$R4_REPLAY_MERGED"
for stem in $(find "$R1_POOL_DIR" -maxdepth 1 -name '*.png' -printf '%f\n' | sed 's/\.png$//' | shuf -n "$R1_REPLAY_N"); do
  ln -s "$R1_POOL_DIR/$stem.png"  "$R4_REPLAY_MERGED/$stem.png"
  ln -s "$R1_POOL_DIR/$stem.json" "$R4_REPLAY_MERGED/$stem.json"
done
for stem in $(find "$R2_POOL_DIR" -maxdepth 1 -name '*.png' -printf '%f\n' | sed 's/\.png$//' | shuf -n "$R2_REPLAY_N"); do
  ln -s "$R2_POOL_DIR/$stem.png"  "$R4_REPLAY_MERGED/$stem.png"
  ln -s "$R2_POOL_DIR/$stem.json" "$R4_REPLAY_MERGED/$stem.json"
done
MERGED_N=$(find "$R4_REPLAY_MERGED" -maxdepth 1 -name '*.png' | wc -l)
echo "[$STAGE_NAME] Replay 풀 완성: ${MERGED_N}장"

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, noise L1~L3) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# p_level_max 0.65(0.5 아님): dataset.py 주석 기록 -- L2/L3 균등(0.5)으로 돌렸다가
# 재검출 성공률 격차(L2 92% vs L3 56%)로 실제 어려운 샘플 노출이 부족해져 L3 성적이
# 오히려 악화(41.5%->35.3%)된 전례가 있음. 성공률 역수 비율(~1.64)만큼 상위 레벨을
# 더 뽑도록 상향(2026-08-01, 동일 함정 회피).
python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$R4_REPLAY_MERGED" --replay_count "$REPLAY_COUNT" \
  --page_level_noise --noise_level 1 --noise_level_max 3 --p_level_max 0.65 \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- register_accuracy_r89.py/diagnose_third_confusion.py로 실측 재검증 필요 (노이즈 낀 실사 촬영 사진으로도 검증 권장)"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) Round 4 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
