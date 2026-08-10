#!/bin/bash
# heightpreserve 후속 실험 2탄 (2026-08-01). warmup(1탄, proj만 워밍업)이 naive보다는
# 나았지만(epoch14 기준 Acc79.1%) 여전히 lowmarkov(TF 95%+대) 수준엔 크게 못 미쳤음.
# 가설: backbone.8/9(이식된 마지막 2단계)는 원래 "뒤에 conv가 2개 더 있다"는 전제로
# 학습된 가중치라, pooling 직전 최종 특징 추출 역할엔 최적화가 안 돼 있을 수 있음.
# proj뿐 아니라 backbone 마지막 2단계(8/9)도 워밍업 구간에 같이 학습되게 열어서 검증.
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/heightpreserve_warmup2_train.log
STATUS=/workspace/heightpreserve_warmup2_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 높이 부분 보존 + proj+backbone[8,9] 워밍업(5에폭) 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/lowmarkov_pool
REPLAY_DIR=/workspace/round3train/data/step1_pool
REPLAY_COUNT=1000
OUT_DIR=/workspace/models/heightpreserve_warmup2
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
  --proj_warmup_epochs "$PROJ_WARMUP_EPOCHS" --proj_warmup_extra_backbone_stages 2 \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --batch 24 --epochs "$EPOCHS" --workers 8 \
  --freeze_epochs 0 --no_augment
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[heightpreserve_warmup2] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:heightpreserve_warmup2" >> "$STATUS"
echo "=== $(date) 높이 부분 보존 + proj+backbone[8,9] 워밍업 학습 종료 ==="
