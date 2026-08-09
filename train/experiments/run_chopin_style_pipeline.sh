#!/bin/bash
# pod에서 실행: 오선1개/대보표 혼합 데이터 생성 -> 파인튜닝(조기중단 포함) 파이프라인.
# nohup으로 백그라운드 실행, ssh 세션 끊겨도 계속 진행됨.
set -e
cd /workspace/round3train

OUT_DATA=/workspace/round3train/Round_chopin_style
MUSESCORE=/workspace/musescore_wrapper.sh
COMMON_ARGS="--diatonic-bias 0.55 --chord-max-interval 16 --melodic-bias 0.7 --melodic-max-step 3 --short-note-bias 1.0 --fermata-prob 0 --hairpin-prob 0 --ottava-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0 --repeat-prob 0"

echo "=== [1/3] 단일 오선 4800장 생성 (최대 5마디, wide_page 스타일로 줄바꿈 방지) ==="
python3 generate_scores.py --count 4800 --start-idx 1 --output "$OUT_DATA" \
    --musescore "$MUSESCORE" --single-staff --min-measures 2 --max-measures 5 $COMMON_ARGS

echo "=== [2/3] 대보표 3200장 생성 (최대 6마디) ==="
python3 generate_scores.py --count 3200 --start-idx 4801 --output "$OUT_DATA" \
    --musescore "$MUSESCORE" --min-measures 2 --max-measures 6 $COMMON_ARGS

echo "=== [3/3] 파인튜닝 시작 (조기중단: unfreeze+20epoch까지 65% 미만, patience=8, 94~95% 5epoch 정체) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
    --data_dir "$OUT_DATA" --out_dir /workspace/models/round1_curriculum_chopin_style \
    --tokenizer /workspace/round3train/tokenizer258.json \
    --resume /workspace/models/round1_curriculum_p2s5n5/seq2seq_best.pt \
    --no_augment --batch 24 --workers 16 --epochs 40 \
    --target_acc 65 --target_check_after_unfreeze 20 --patience 8 \
    --plateau_band "94,95" --plateau_band_epochs 5

echo "=== PIPELINE_DONE ==="
