#!/bin/bash
# "높이 부분 보존" 인코더 구조 실험 (2026-08-01). CoordConv는 좌표 신호를 입력에 주입해
# 8단 conv를 통과하며 "살아남기를 바라는" 간접적 방식(효과는 있었음: 273->235건)이었던 반면,
# 이 실험은 backbone의 세로 축소 단계를 4->2로 줄여(extra_height_stages=2) pool 직전에
# 세로가 여러 줄 남게 하고, pool_h=4로 그 줄들을 채널에 이어붙여 보존한다 -- 위치 정보가
# 학습에 의존하지 않고 구조적으로 디코더까지 도달하도록 강제.
# lowmarkov 체크포인트(CoordConv 없음, in_ch=1)에서 이어받아 CoordConv와 독립적으로
# 3도 오독 건수를 비교하기 위함(235건/292건과 동일 선상 비교).
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/heightpreserve_train.log
STATUS=/workspace/heightpreserve_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 높이 부분 보존(extra_height_stages=2, pool_h=4) 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/lowmarkov_pool
REPLAY_DIR=/workspace/round3train/data/step1_pool
REPLAY_COUNT=1000
OUT_DIR=/workspace/models/heightpreserve
RESUME=/workspace/models/lowmarkov/seq2seq_best.pt
TOKENIZER=/workspace/round3train/tokenizer258.json
EPOCHS=20
FREEZE_EPOCHS=0

mkdir -p "$OUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --resume "$RESUME" \
  --in_ch 1 --extra_height_stages 2 --pool_h 4 \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --batch 24 --epochs "$EPOCHS" --workers 8 \
  --freeze_epochs "$FREEZE_EPOCHS" --no_augment
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[heightpreserve] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:heightpreserve" >> "$STATUS"
echo "=== $(date) 높이 부분 보존 학습 종료 ==="
