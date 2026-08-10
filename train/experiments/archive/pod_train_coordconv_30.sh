#!/bin/bash
# CoordConv 30에폭 확장(2026-07-31): 8에폭 검증(coordconv_probe)에서 D6->B5/C6->A5
# 단3도 오독은 줄었지만 D4->B5(44건)/G3->E5(34건) 등 새 대형 오류가 생김 -- 8에폭은
# 좌표 채널이 아직 수렴하기엔 너무 짧았을 가능성이 커서, 그 체크포인트에서 이어
# 30에폭까지 채워서 재검증한다(처음부터 다시 돌리지 않음 -- 이미 학습된 좌표 채널
# 적응을 그대로 이어받음). shape가 이미 in_ch=2로 일치하므로 특별한 채널 이식 없이
# 일반 load_ckpt_partial_vocab만으로 전량 로드됨.
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/coordconv_30.log
STATUS=/workspace/coordconv_30_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) CoordConv 30에폭 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/step1_pool
OUT_DIR=/workspace/models/coordconv_30
RESUME=/workspace/models/coordconv_probe/seq2seq_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=30
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
  echo "[coordconv_30] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:coordconv_30" >> "$STATUS"
echo "=== $(date) CoordConv 30에폭 학습 종료 ==="
