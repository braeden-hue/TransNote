#!/bin/bash
# run_chopin_style_gen.sh로 8000장 생성 완료 확인 후, 사용자 승인 받고 수동 실행하는 학습
# 단계. 조기중단: unfreeze+20epoch까지 65% 미만(실패/중단 케이스) / patience=8 /
# 94~95% 구간 5epoch 정체(성공 케이스로 취급).
set -e
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
    --data_dir /workspace/round3train/Round_chopin_style \
    --out_dir /workspace/models/round1_curriculum_chopin_style \
    --tokenizer /workspace/round3train/tokenizer258.json \
    --resume /workspace/models/round1_curriculum_p2s5n5/seq2seq_best.pt \
    --no_augment --batch 24 --workers 16 --epochs 40 \
    --target_acc 65 --target_check_after_unfreeze 20 --patience 8 \
    --plateau_band "94,95" --plateau_band_epochs 5

echo "=== TRAIN_DONE ==="
