#!/bin/bash
# 극단 음역 보강 재학습(2026-07-31): Round3 베이스에서, 극단 음역 데이터(3085장,
# --preferred-register-prob 교집합 버그 수정 후 생성 -- 옥타브6 0.0%->4.3%)를
# 주 데이터로 쓰고 Step1 데이터(6157장 중 3000장만 무작위 replay)를 섞는다.
# 배경: 89곡 실측에서 옥타브 2~5는 80.4%인데 옥타브 0~1/6+는 정확히 0.0%(121건 전부
# 오답) -- 노출 자체가 없어서 생긴 완전한 능력 부재로 확인됨. CoordConv(구조 변경)는
# 이번 세션에서 결론이 불확실했던 반면, 이건 데이터 노출 격차가 원인이라는 증거가
# 훨씬 명확해서 먼저 이 조합부터 검증.
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/extreme_train.log
STATUS=/workspace/extreme_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 극단음역+Step1(3000) 재학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/extreme_register_pool
REPLAY_DIR=/workspace/round3train/data/step1_pool
REPLAY_COUNT=3000
OUT_DIR=/workspace/models/extreme_register
RESUME=/workspace/checkpoints/seq2seq_r3_density_register_clef_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=30
FREEZE_EPOCHS=8

mkdir -p "$OUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --resume "$RESUME" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --batch 24 --epochs "$EPOCHS" --workers 8 \
  --freeze_epochs "$FREEZE_EPOCHS" --no_augment
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[extreme_register] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi

BEST_ACC=$(python3 -c "
import csv
best = 0.0
with open('$OUT_DIR/seq2seq_phase2_log.csv') as f:
    for row in csv.DictReader(f):
        a = float(row['val_acc'])
        if a > best:
            best = a
print(f'{best:.2f}')
")
echo "[extreme_register] 학습 완료 -- best val_acc(teacher-forcing) = ${BEST_ACC}%"
echo "[extreme_register] 주의: TF 기준이라 신뢰 불가 -- 실측(exactPicture)으로 재검증할 것"
echo "STAGE_DONE:extreme_register:${BEST_ACC}" >> "$STATUS"
echo "=== $(date) 극단음역+Step1(3000) 재학습 종료 ==="
