#!/bin/bash
# round3train/curriculum_r7_l4_major_realphotos.sh
#
# L3/L4 노이즈 + 클래식 실사 사진(86곡, 1550장) 학습 투입(2026-08-02).
# newage(15곡)는 검증 전용으로 완전히 제외 -- 학습에 절대 포함하지 않음(오염 방지).
#
# 메인 데이터 = 신규 합성 2000장 + 클래식 실사 1550장(심볼릭 링크로 병합) = 3550장.
# 실사 사진도 OMRDataset이 그대로 읽을 수 있음(detect_staffs 파이프라인 공용, jpg 지원
# 이미 코드에 있음). dataset.py 수정으로 실사 사진(jpg/jpeg)에는 합성 노이즈를 전혀
# 얹지 않음(확장자로 자동 구분, 2026-08-02) -- 신규 합성(png)에만 augment 적용.
#
# 시도 이력(2026-08-02):
#   v1: freeze_epochs=0, noise_level 3~4, p_level_max=0.8 -- epoch1~3 Acc
#       4.4%->20.5%->8.5%로 크게 흔들려 중단.
#   v2: freeze_epochs=6, noise_level 2~4(주의: 이 옵션은 연속범위가 아니라 두 값 중
#       하나를 뽑는 이진 선택이라 실제로는 "L2 또는 L4"만 나오고 L3는 전혀 등장 안
#       했음 -- 설계 실수), p_level_max=0.5 -- epoch1 Acc 0.0%(인코더가 얼어있는 채로
#       완전히 낯선 실사+노이즈 입력을 받아 디코더 혼자 복구 불가능했던 것으로 추정).
#   v4(현재): noise_level 3~4로 되돌림(L3/L4만, 의도한 대로). freeze_epochs 6->0으로
#       재복귀 -- v2의 Acc 0.0%은 freeze 자체가 원인으로 재판단(인코더가 r6까지
#       합성 노이즈에만 적응했지 실사 사진은 한 번도 본 적이 없어서, 얼려두면 실사에서
#       특징을 뽑는 법 자체를 못 배움). p_level_max는 재검출 성공률 보정(0.7 계산,
#       L3 80%/L4 40% 실측)보다 "실사 사진 1550장이 이미 실제 난이도를 직접 가르치고
#       있으니 합성 노이즈를 극단으로 밀어붙일 필요는 적다"는 쪽을 우선해 0.5로 절충.
# r6_l4_noise에서 이어받음(v1/v2/v3의 흔들린 체크포인트 아님).
#
# replay = 이전 라운드들(1200) + 직전 라운드 r6(800) 혼합, Round6 완료 후에만 이 스크립트
# 실행 가능(r6_l4_noise 데이터가 있어야 함).
#
# 사전 조건: curriculum_r6_l4_noise.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r7.log
STATUS=/workspace/curriculum_r7_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 7(L4 위주 노이즈 + 클래식 실사 사진) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r6_l4_noise" /workspace/curriculum_r6_status.txt 2>/dev/null; then
  echo "[r7] Round6이 STAGE_PASSED 상태가 아님 -- 중단(먼저 확인 필요)"
  echo "PIPELINE_STOPPED_R6_NOT_PASSED" >> "$STATUS"
  exit 1
fi

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
SYN_POOL_DIR=/workspace/data/r7_l4_major_synth
REAL_POOL_DIR=/workspace/data/classical_realphotos
MERGED_DATA_DIR=/workspace/data/r7_merged
SINGLE_COUNT=300
GRAND_COUNT=1700
START_IDX_SINGLE=11000001
START_IDX_GRAND=11500001
SEED_SINGLE=1100001
SEED_GRAND=1150001
N_SHARDS=6

REPLAY_POOL_OLD=/workspace/data/replay_pool_v2
REPLAY_POOL_R6=/workspace/data/r6_l4_noise
REPLAY_MERGED=/workspace/data/r7_replay_merged
REPLAY_OLD_N=1200
REPLAY_R6_N=800
REPLAY_COUNT=2000

RESUME_CKPT=/workspace/models/r6_l4_noise/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r7_l4_major_realphotos
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=30
FREEZE_EPOCHS=0   # v2에서 freeze=6으로 해봤더니 epoch1 Acc 0.0% -- 인코더가 r6까지
                  # 합성 노이즈에만 적응했지 실사 사진은 한 번도 본 적이 없어서, 얼려두면
                  # 실사에서 유의미한 특징을 뽑는 법 자체를 못 배움(디코더 혼자 복구 불가).
                  # v1의 흔들림은 freeze 여부가 아니라 노이즈 강도(p_level_max=0.8) 문제로
                  # 재판단(2026-08-02) -- freeze는 다시 0으로.
GATE=80
MARKOV_TABLE=/workspace/round3train/markov_transitions.json
MARKOV_BIAS=0.3

BASE_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.22
           --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
           --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.05
           --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
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
if [ ! -d "$REAL_POOL_DIR" ]; then
  echo "[$STAGE_NAME] 클래식 실사 풀 없음($REAL_POOL_DIR) -- prepare_classical_realphotos.py 먼저 실행 필요"
  echo "PIPELINE_STOPPED_NO_REALPHOTOS" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] 신규 합성 데이터 풀 확인 중..."
TOTAL_SYN=$((SINGLE_COUNT + GRAND_COUNT))
EXISTING_N=$(find "$SYN_POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
MIN_OK_N=$(( TOTAL_SYN * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[$STAGE_NAME] 합성 데이터 풀 이미 존재(${EXISTING_N}/${TOTAL_SYN}장) -- 재생성 스킵"
else
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
  echo "[$STAGE_NAME] 합성 데이터 풀 완성: ${FINAL_N}/${TOTAL_SYN}장"
fi
echo "STAGE_DATA_DONE:$STAGE_NAME" >> "$STATUS"

echo "[$STAGE_NAME] 신규 합성 + 클래식 실사 병합 중 (심볼릭 링크)..."
rm -rf "$MERGED_DATA_DIR"
mkdir -p "$MERGED_DATA_DIR"
for f in "$SYN_POOL_DIR"/*.png "$SYN_POOL_DIR"/*.json; do ln -s "$f" "$MERGED_DATA_DIR/"; done
for f in "$REAL_POOL_DIR"/*; do ln -s "$f" "$MERGED_DATA_DIR/"; done
MERGED_N=$(find "$MERGED_DATA_DIR" -maxdepth 1 \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
echo "[$STAGE_NAME] 병합 데이터: ${MERGED_N}장(합성+실사)"

echo "[$STAGE_NAME] Replay 풀 구성 중 (이전 라운드 ${REPLAY_OLD_N} + 직전 r6 ${REPLAY_R6_N}, 실제 파일 복사)..."
rm -rf "$REPLAY_MERGED"
mkdir -p "$REPLAY_MERGED"
pick_and_copy() {
  local src=$1 n=$2 prefix=$3
  find "$src" -maxdepth 1 -name '*.json' -printf '%f\n' | sed 's/\.json$//' | \
    while read -r stem; do
      if [ -f "$src/$stem.png" ]; then echo "$stem"; fi
    done | shuf -n "$n" | while read -r stem; do
      cp "$src/$stem.png"  "$REPLAY_MERGED/${prefix}_$stem.png"
      cp "$src/$stem.json" "$REPLAY_MERGED/${prefix}_$stem.json"
    done
}
pick_and_copy "$REPLAY_POOL_OLD" "$REPLAY_OLD_N" old
pick_and_copy "$REPLAY_POOL_R6" "$REPLAY_R6_N" r6
REPLAY_N=$(find "$REPLAY_MERGED" -maxdepth 1 -name '*.png' | wc -l)
echo "[$STAGE_NAME] Replay 풀 완성: ${REPLAY_N}장"

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}, replay=${REPLAY_COUNT}, noise L3/L4 p=0.5) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$MERGED_DATA_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$REPLAY_MERGED" --replay_count "$REPLAY_COUNT" \
  --page_level_noise --noise_level 3 --noise_level_max 4 --p_level_max 0.5 \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- newage 14곡(학습 미포함, 검증 전용)으로 실측 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) Round 7 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
