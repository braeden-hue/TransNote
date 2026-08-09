#!/bin/bash
cd /c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/round3train
while :; do
  N=$(find data/local_pools/r2_grandstaff -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "$N" -ge 5000 ]; then break; fi
  sleep 20
done
echo "[r3watch] Round2 로컬 5000장 완료 감지 -- Round3 로컬 분량(단일오선 2500+대보표 2500, 2026-07-30 재축소분 5000장에 맞춰 조정) 시작"
MS="/c/Program Files/MuseScore 4/bin/MuseScore4.exe"
MT=markov_transitions.json
BASE_ARGS="--min-measures 1 --max-measures 4 --chord-prob 0.08 --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0 --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35 --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.15 --dotted8-bias 10.0 --short-note-bias 2.0 --clef-change-prob 0.15 --markov-bias 0.5 --markov-table $MT"
python generate_scores.py --count 2500 --output data/local_pools/r3_density_register_clef \
  --musescore "$MS" --single-staff --start-idx 40000001 --seed 4000001 $BASE_ARGS
python generate_scores.py --count 2500 --output data/local_pools/r3_density_register_clef \
  --musescore "$MS" --start-idx 41000001 --seed 4100001 $BASE_ARGS --cross-register-prob 0.30 --preferred-register-prob 0.6
echo "[r3watch] Round3 로컬 분량 완료 -- pod의 /workspace/data/r3_density_register_clef로 업로드해야 curriculum_r3 스크립트가 중복 생성 안 함"
