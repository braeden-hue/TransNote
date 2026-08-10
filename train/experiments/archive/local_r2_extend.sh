#!/bin/bash
cd /c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/round3train
POOL_DIR=data/local_pools/r2_grandstaff
TARGET=5000
MS="/c/Program Files/MuseScore 4/bin/MuseScore4.exe"
MT=markov_transitions.json
ARGS="--min-measures 1 --max-measures 4 --chord-prob 0.08 --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35 --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --cross-register-prob 0.15 --tie-prob 0.15 --dotted8-bias 10.0 --short-note-bias 2.0 --preferred-register-prob 0.3 --markov-bias 0.5 --markov-table $MT"

CUR=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
REMAIN=$(( TARGET - CUR ))
echo "[r2extend] 현재 ${CUR}장, 목표 ${TARGET}장, 남은 ${REMAIN}장 생성 시작"
if [ "$REMAIN" -gt 0 ]; then
  python generate_scores.py --count "$REMAIN" --output "$POOL_DIR" \
    --musescore "$MS" --start-idx 8200001 --seed 820001 $ARGS
fi
FINAL=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "[r2extend] 완료: ${FINAL}/${TARGET}장"
