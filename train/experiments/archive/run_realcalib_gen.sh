#!/bin/bash
# 실사 40장 재보정 라운드(real_texture_bank 그레인/조명 + curl_shade + 캔버스 기하 수정 반영).
# 단일오선 3000 + 대보표 2000 = 5000장. 노이즈 자체는 여기서 굽지 않음 -- dataset.py가
# 학습 시 --noise_level 2로 즉석 적용(p_real_texture=0.85로 대부분 샘플에 실사 텍스처 적용).
set -e
cd /workspace/round3train

OUT_DATA=/workspace/round3train/Round_realcalib
MUSESCORE=/workspace/musescore_wrapper.sh
COMMON_ARGS="--diatonic-bias 0.55 --chord-max-interval 16 --melodic-bias 0.7 --melodic-max-step 3 --fermata-prob 0 --hairpin-prob 0 --ottava-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0 --repeat-prob 0"

echo "=== [1/2] 단일 오선 3000장 생성 (short-note-bias 0.4, 최대 5마디) ==="
python3 generate_scores.py --count 3000 --start-idx 1 --output "$OUT_DATA" \
    --musescore "$MUSESCORE" --single-staff --min-measures 2 --max-measures 5 \
    --short-note-bias 0.4 $COMMON_ARGS

echo "=== [2/2] 대보표 2000장 생성 (short-note-bias 1.0, 최대 6마디) ==="
python3 generate_scores.py --count 2000 --start-idx 3001 --output "$OUT_DATA" \
    --musescore "$MUSESCORE" --min-measures 2 --max-measures 6 \
    --short-note-bias 1.0 $COMMON_ARGS

echo "=== GEN_DONE: 5000장 생성 완료 ==="
