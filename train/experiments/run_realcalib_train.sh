#!/bin/bash
# Round_realcalib(5000장) 데이터로, chopin_style 체크포인트에서 이어서 파인튜닝.
# noise2와 달리 이번엔 dataset.py의 augment_image()가 real_texture_bank 그레인/조명 +
# curl_shade(모서리 그림자, 기하 왜곡 없음)를 실측 보정된 강도로 적용한다(p_real_texture
# 0.8~0.95). page_level_noise는 이번 라운드에서 의도적으로 끔 -- curl_shade가 이미
# 캔버스 레벨에서 항상 안전하게 적용되므로, 재검출 실패 위험이 있는 페이지 레벨 기하
# curl+회전(page_noise_and_redetect)을 중복으로 쓸 필요가 없다는 판단(2026-07-28).
set -e
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
    --data_dir /workspace/round3train/Round_realcalib \
    --out_dir /workspace/models/round1_curriculum_realcalib \
    --tokenizer /workspace/round3train/tokenizer258.json \
    --resume /workspace/models/round1_curriculum_chopin_style/seq2seq_best.pt \
    --noise_level 2 --batch 24 --workers 16 --epochs 40 \
    --target_acc 80 --target_check_epoch 20 --patience 8 \
    --plateau_band "94,95" --plateau_band_epochs 8 --plateau_min_epoch 20

echo "=== TRAIN_DONE ==="
