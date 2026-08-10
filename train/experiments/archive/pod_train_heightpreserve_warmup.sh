#!/bin/bash
# heightpreserve(extra_height_stages=2, pool_h=4)의 후속 실험 -- 2026-08-01.
# naive 버전(pod_train_heightpreserve.sh, freeze_epochs=0으로 처음부터 전체를 같이 파인튜닝)이
# epoch7 Acc=74.8% 이후 epoch8~12까지 정체됐던 것에 대한 가설 검증: encoder.proj가 완전히
# 랜덤 초기화된 새 레이어라 처음부터 디코더+backbone과 같이(낮은 파인튜닝 학습률로) 학습시키면
# 이미 잘 학습된 부분이 초기 노이즈 신호에 끌려다닐 수 있다 -- proj만 먼저 5에폭 워밍업(일반
# 학습률)시킨 뒤 전체를 이어서 학습(lr/3)하도록 분리.
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/heightpreserve_warmup_train.log
STATUS=/workspace/heightpreserve_warmup_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 높이 부분 보존 + proj 워밍업(5에폭) 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/lowmarkov_pool
REPLAY_DIR=/workspace/round3train/data/step1_pool
REPLAY_COUNT=1000
OUT_DIR=/workspace/models/heightpreserve_warmup
RESUME=/workspace/models/lowmarkov/seq2seq_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=20
PROJ_WARMUP_EPOCHS=5

mkdir -p "$OUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --resume "$RESUME" \
  --in_ch 1 --extra_height_stages 2 --pool_h 4 \
  --proj_warmup_epochs "$PROJ_WARMUP_EPOCHS" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --batch 24 --epochs "$EPOCHS" --workers 8 \
  --freeze_epochs 0 --no_augment
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[heightpreserve_warmup] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:heightpreserve_warmup" >> "$STATUS"
echo "=== $(date) 높이 부분 보존 + proj 워밍업 학습 종료 ==="
