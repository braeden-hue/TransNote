#!/bin/bash
# 결합 실험: lowmarkov(markov-bias 0.3, 화음/리듬 2단계 누적) 체크포인트에 CoordConv
# (in_ch=2, 세로 좌표 채널)를 얹어서 이어서 학습. train.py가 in_ch=1->2 채널 이식을
# 자동 처리(기존 채널 복사 + 좌표 채널 0 초기화)하므로 처음부터 다시 학습할 필요 없음.
# 배경: 반복(오스티나토) 패턴 화성 환각(3도 오독)이 markov-bias만으로는 해결 안 됨
# (0.5->0.3에서 273->292건, 오히려 소폭 증가). 반면 기존 CoordConv(30에폭, 단독)는
# 235건으로 더 낮았음 -- 인코딩(위치 정보) 쪽이 이 문제엔 더 유효해 보여 두 개선을 결합.
set -uo pipefail
cd /workspace/round3train
LOG=/workspace/combined_coordconv_train.log
STATUS=/workspace/combined_coordconv_train_status.txt
exec >> "$LOG" 2>&1
echo "=== $(date) 결합(lowmarkov+CoordConv) 학습 시작 ==="
: > "$STATUS"

DATA_DIR=/workspace/round3train/data/lowmarkov_pool
REPLAY_DIR=/workspace/round3train/data/step1_pool
REPLAY_COUNT=1000
OUT_DIR=/workspace/models/combined_coordconv
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
  --in_ch 2 \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --batch 24 --epochs "$EPOCHS" --workers 8 \
  --freeze_epochs "$FREEZE_EPOCHS" --no_augment
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[combined_coordconv] 학습 프로세스 비정상 종료(exit=$TRAIN_RC)"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi
echo "STAGE_DONE:combined_coordconv" >> "$STATUS"
echo "=== $(date) 결합(lowmarkov+CoordConv) 학습 종료 ==="
