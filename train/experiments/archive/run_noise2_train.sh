#!/bin/bash
# Round_noise2(5000장) 데이터로, 직전 라운드 체크포인트(seq2seq_chopin_style)에서 이어서
# 노이즈 level 2 파인튜닝. --no_augment 제거하고 --noise_level 2로 실사 강건성 학습.
set -e
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
    --data_dir /workspace/round3train/Round_noise2 \
    --out_dir /workspace/models/round1_curriculum_noise2 \
    --tokenizer /workspace/round3train/tokenizer258.json \
    --resume /workspace/models/round1_curriculum_chopin_style/seq2seq_best.pt \
    --noise_level 2 --batch 24 --workers 16 --epochs 40 \
    --target_acc 80 --target_check_epoch 20 --patience 8 \
    --plateau_band "94,95" --plateau_band_epochs 8 --plateau_min_epoch 20

echo "=== TRAIN_DONE ==="
