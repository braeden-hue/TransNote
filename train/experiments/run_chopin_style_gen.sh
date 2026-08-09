#!/bin/bash
# pod에서 실행: 오선1개/대보표 혼합 데이터 8000장 생성만. 학습은 여기서 자동으로 이어가지
# 않고 GEN_DONE 로그만 남기고 종료 -- 8000장 생성 끝나면 사용자 확인 후 별도로
# run_chopin_style_train.sh를 수동 실행하는 방식(2026-07-28, 자동 이어달리기 하지 않기로 함).
set -e
cd /workspace/round3train

OUT_DATA=/workspace/round3train/Round_chopin_style
MUSESCORE=/workspace/musescore_wrapper.sh
COMMON_ARGS="--diatonic-bias 0.55 --chord-max-interval 16 --melodic-bias 0.7 --melodic-max-step 3 --short-note-bias 1.0 --fermata-prob 0 --hairpin-prob 0 --ottava-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0 --repeat-prob 0"

echo "=== [1/2] 단일 오선 4800장 생성 (최대 5마디, wide_page 스타일로 줄바꿈 방지) ==="
python3 generate_scores.py --count 4800 --start-idx 1 --output "$OUT_DATA" \
    --musescore "$MUSESCORE" --single-staff --min-measures 2 --max-measures 5 $COMMON_ARGS

echo "=== [2/2] 대보표 3200장 생성 (최대 6마디) ==="
python3 generate_scores.py --count 3200 --start-idx 4801 --output "$OUT_DATA" \
    --musescore "$MUSESCORE" --min-measures 2 --max-measures 6 $COMMON_ARGS

echo "=== GEN_DONE: 8000장 생성 완료, 학습은 확인 후 run_chopin_style_train.sh로 수동 시작 ==="
