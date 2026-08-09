#!/bin/bash
# round3train/curriculum_r9_chord_accidental.sh
#
# Round9 -- 화음(2/3/4음)+임시표 콘텐츠 보강. newage 실사 오류분석(2026-08-02, Round8
# 77.8% 체크포인트 대상)에서 정확도 0%로 완전 실패한 사진 3장(newage04/05/07)을 직접
# 확인한 결과, 셋 다 저음부에 3~4음 화음이 빽빽한 구간이었음 -- 화음이 여전히 주요
# 오류원(project.md 기록: chord-prob 0.08->0.22 확정 당시에도 "화음이 새 주요 오류원"
# 이었음). 임시표(#,b) 누락도 다수 확인(Eb4->E4 22회, C#4->C4 13회 등).
#
# 실사 도입(Round8)과 축을 분리 -- 콘텐츠 보강은 실사 없이 합성만으로 먼저 검증
# (이번 세션 내내 확인된 "한 번에 축 하나만" 원칙). r7b(L3/L4 노이즈 안정화 완료,
# TF Acc 94.0%)에서 이어받아 노이즈 설정은 그대로 유지하고 화음/임시표 노출만 높인다.
#
# 사전 조건: curriculum_r7_l4_synth_only.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r9.log
STATUS=/workspace/curriculum_r9_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 9(화음 2/3/4음 + 임시표 보강, 실사 없음) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r7b_l4_synth_only" /workspace/curriculum_r7b_status.txt 2>/dev/null; then
  echo "[r9] Round7(재정의)가 STAGE_PASSED 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_R7B_NOT_PASSED" >> "$STATUS"
  exit 1
fi

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
SYN_POOL_DIR=/workspace/data/r9_chord_accidental_synth
SINGLE_COUNT=300
GRAND_COUNT=1700
START_IDX_SINGLE=12000001
START_IDX_GRAND=12500001
SEED_SINGLE=1200001
SEED_GRAND=1250001
N_SHARDS=6

REPLAY_DIR=/workspace/data/replay_pool_v3   # r6까지 검증된 합성 replay, 그대로 재사용
REPLAY_COUNT=2000

RESUME_CKPT=/workspace/models/r7b_l4_synth_only/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r9_chord_accidental
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15
FREEZE_EPOCHS=0   # 실사 도입이 아니라 순수 콘텐츠 보강이라 인코더가 이미 익숙한
                  # 도메인(합성+L3/L4 노이즈) 그대로라 freeze 불필요.
GATE=85
MARKOV_TABLE=/workspace/round3train/markov_transitions.json
MARKOV_BIAS=0.3

# diatonic-bias 0.75->0.55: 임시표 노출 확대(0=완전 무작위, 1=항상 조표에 맞는 음만).
# chord-prob 0.22->0.28, chord-size-weights로 2/3/4음 화음 명시적 분포(기존엔 2~3음
# 범위 안에서 암묵적 분포였고 4음 화음은 아예 없었음).
BASE_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.28
           --chord-min-notes 2 --chord-max-notes 4 --chord-size-weights "2:35,3:35,4:30"
           --repeat-prob 0
           --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
           --diatonic-bias 0.55 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
           --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.25
           --dotted8-bias 12.0 --eighth-run-prob 0.15 --sixteenth-run-prob 0.18
           --preferred-register-prob 0.3
           --markov-bias "$MARKOV_BIAS" --markov-table "$MARKOV_TABLE")
GRAND_ONLY_ARGS=(--cross-register-prob 0.20)

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] 신규 합성 데이터 생성 중 (화음/임시표 보강)..."
rm -rf "$SYN_POOL_DIR"
mkdir -p "$SYN_POOL_DIR"

echo "[$STAGE_NAME] 단일오선(${SINGLE_COUNT}장) 생성 중..."
SHARD_BASE=$(( SINGLE_COUNT / N_SHARDS ))
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
  SHARD_START=$(( START_IDX_SINGLE + i * 10000 ))
  SHARD_SEED=$(( SEED_SINGLE + i ))
  THIS_COUNT=$SHARD_BASE
  if [ "$i" -eq $((N_SHARDS - 1)) ]; then
    THIS_COUNT=$(( SINGLE_COUNT - SHARD_BASE * (N_SHARDS - 1) ))
  fi
  xvfb-run -a python3 /workspace/round3train/generate_scores.py \
    --count "$THIS_COUNT" --output "$SYN_POOL_DIR" --musescore "$MUSESCORE" \
    --single-staff --start-idx "$SHARD_START" --seed "$SHARD_SEED" \
    "${BASE_ARGS[@]}" > "/workspace/gen_shard_${STAGE_NAME}_single_${i}.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then
  echo "[$STAGE_NAME] 단일오선 생성 실패 -- 파이프라인 중단"
  echo "PIPELINE_STOPPED_GENFAIL:single" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] 대보표(${GRAND_COUNT}장) 생성 중..."
SHARD_BASE=$(( GRAND_COUNT / N_SHARDS ))
PIDS=()
for i in $(seq 0 $((N_SHARDS - 1))); do
  SHARD_START=$(( START_IDX_GRAND + i * 10000 ))
  SHARD_SEED=$(( SEED_GRAND + i ))
  THIS_COUNT=$SHARD_BASE
  if [ "$i" -eq $((N_SHARDS - 1)) ]; then
    THIS_COUNT=$(( GRAND_COUNT - SHARD_BASE * (N_SHARDS - 1) ))
  fi
  xvfb-run -a python3 /workspace/round3train/generate_scores.py \
    --count "$THIS_COUNT" --output "$SYN_POOL_DIR" --musescore "$MUSESCORE" \
    --start-idx "$SHARD_START" --seed "$SHARD_SEED" \
    "${BASE_ARGS[@]}" "${GRAND_ONLY_ARGS[@]}" > "/workspace/gen_shard_${STAGE_NAME}_grand_${i}.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then
  echo "[$STAGE_NAME] 대보표 생성 실패 -- 파이프라인 중단"
  echo "PIPELINE_STOPPED_GENFAIL:grand" >> "$STATUS"
  exit 1
fi
FINAL_N=$(find "$SYN_POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "[$STAGE_NAME] 합성 데이터 풀 완성: ${FINAL_N}/$((SINGLE_COUNT + GRAND_COUNT))장"
echo "STAGE_DATA_DONE:$STAGE_NAME" >> "$STATUS"

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, replay=${REPLAY_COUNT}, noise L3/L4 그대로) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$SYN_POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
  --page_level_noise --noise_level 3 --noise_level_max 4 --p_level_max 0.7 \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS"
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[$STAGE_NAME] 학습 프로세스 비정상 종료(exit=$TRAIN_RC) -- 파이프라인 중단"
  echo "PIPELINE_STOPPED_TRAINFAIL" >> "$STATUS"
  exit 1
fi

BEST_ACC=$(python3 -c "
import csv
best = 0.0
with open('$TRAIN_OUT/seq2seq_phase2_log.csv') as f:
    for row in csv.DictReader(f):
        a = float(row['val_acc'])
        if a > best:
            best = a
print(f'{best:.2f}')
")
echo "[$STAGE_NAME] 학습 완료 -- best val_acc(teacher-forcing) = ${BEST_ACC}% (gate=${GATE}%)"
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- 화음/임시표 비중이 높아진 합성 검증셋 또는"
echo "  register_accuracy_r89.py 등으로 재검증 필요. 실사 재검증은 Round8 재학습 이후에."

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 9 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
