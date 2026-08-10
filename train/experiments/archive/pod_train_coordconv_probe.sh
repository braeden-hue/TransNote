#!/bin/bash
# CoordConv 소규모 검증(2026-07-31): 세로 좌표 채널 추가가 단3도/옥타브 오독을 줄이는지
# 8에폭짜리 짧은 실험으로 먼저 확인. freeze_epochs=0 -- 인코더가 동결돼 있으면 새 좌표
# 채널(0으로 초기화된 첫 conv 가중치)이 전혀 학습되지 않으므로 이 검증에선 반드시 꺼야 함.
# Round3 베이스에서 resume(Step1 계열이 아니라 원본) -- train.py가 in_ch=1 체크포인트를
# in_ch=2 모델로 부분 로드하면서 기존 채널은 그대로 복사, 좌표 채널만 0으로 초기화.
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/coordconv_probe.log
STATUS=/workspace/coordconv_probe_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) CoordConv 검증 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/step1_pool
OUT_DIR=/workspace/models/coordconv_probe
RESUME=/workspace/checkpoints/seq2seq_r3_density_register_clef_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=8
FREEZE_EPOCHS=0

mkdir -p "$OUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --resume "$RESUME" \
  --in_ch 2 \
  --batch 24 --epochs "$EPOCHS" --workers 8 \
  --freeze_epochs "$FREEZE_EPOCHS" --no_augment
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[coordconv_probe] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:coordconv_probe" >> "$STATUS"
echo "=== $(date) CoordConv 검증 학습 종료 ==="
