#!/bin/bash
# Step1 학습: Round3 체크포인트에서 resume, 방금 생성한 6000장(Pool A+B 통합)으로 학습.
# 노이즈 증강은 의도적으로 끔(--no_augment) -- Step2에서 별도로 도입 예정.
# replay_dir 없음: 이전 라운드 데이터 풀이 이 새 네트워크 볼륨에 없어서(다른 데이터센터의
# 예전 볼륨에 있었음) 이번 라운드는 replay 없이 진행 -- catastrophic forgetting 위험이
# 있을 수 있어 학습 후 실측(exactPicture 등)으로 반드시 재확인 필요.
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/step1_train.log
STATUS=/workspace/step1_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) Step1 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/step1_pool
OUT_DIR=/workspace/models/step1
RESUME=/workspace/checkpoints/seq2seq_r3_density_register_clef_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=30
FREEZE_EPOCHS=8
GATE=90

mkdir -p "$OUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2026-07-31: --workers 32(DataLoader num_workers -- pin_memory=True와 함께 각각
# 별도 프로세스로 fork됨, dataset.py의 ThreadPoolExecutor와는 전혀 다른 값)가 실제
# OOM-kill(exit 137) 원인으로 확인됨 -- 컨테이너 cgroup 메모리 한도가 440GB(호스트
# 표시치)가 아니라 실제 약 50GB뿐이었음. Round3 커리큘럼 원래 값(16)보다도 낮춰서 8로.
python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --resume "$RESUME" \
  --batch 24 --epochs "$EPOCHS" --workers 8 \
  --freeze_epochs "$FREEZE_EPOCHS" --no_augment
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[step1] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
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
echo "[step1] 학습 완료 -- best val_acc(teacher-forcing) = ${BEST_ACC}% (참고용, gate=${GATE}%)"
echo "[step1] 주의: TF 기준이라 신뢰 불가 -- 실측(exactPicture 등)으로 재검증할 것"
echo "STAGE_DONE:step1:${BEST_ACC}" >> "$STATUS"
echo "=== $(date) Step1 학습 종료 ==="
