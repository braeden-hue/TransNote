#!/bin/bash
# round3train/curriculum_r1_foundation.sh
#
# 2026-07-30 완전 재시작 결정: 포드 디스크가 전부 비워진 상태 + 기존 noise2 체인이
# "기본 음계 인식" 자체가 의심스럽다는 사용자 판단(TF 게이트는 항상 95%+였지만 실전
# 자기회귀 정확도는 최고 80.8%, 화음 대량 환각까지 실측됨) -- 예전처럼 15개 이상의
# 좁은 단계를 순차로 쌓지 않고, 지금까지 알아낸 모든 개선사항(간격 캡, 다이어토닉
# 바이어스, 화음 버그 수정, wide-page 렌더링 수정, tie 로직, 마르코프 체인)을 처음부터
# 반영한 튼튼한 기초를 다시 만든다.
#
# 대보표는 일부러 뺐다 -- PODPLAN.md 기록상 "대보표 추가"만으로 정확도가 97.9%->79.5%로
# 붕괴한 전례가 있어서(시퀀스 길이 1.7배 증가+ teacher forcing exposure bias), 다른 축과
# 동시에 처음부터 넣는 게 아니라 이 라운드로 기초를 다진 뒤 별도 라운드(r2_grandstaff)로
# 분리한다.
#
# 사전 조건: round3train/pod_bootstrap.sh를 먼저 실행해서 cv2/music21/xvfb/MuseScore가
# 준비돼 있어야 함(2026-07-30 실측 확인 완료 -- xvfb-run으로 감싸야 렌더링됨, venv는
# 네트워크 볼륨에서 너무 느려서 시스템 파이썬에 직접 설치하는 방식으로 변경됨).

set -uo pipefail
LOG=/workspace/curriculum_r1.log
STATUS=/workspace/curriculum_r1_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 1(기초: 단일오선+화음+기본기호+마르코프) 시작 ==="
: > "$STATUS"

MUSESCORE=/workspace/musescore/squashfs-root/AppRun
POOL_DIR=/workspace/data/r1_foundation
COUNT=3000
START_IDX=1000001
SEED=100001
N_SHARDS=6   # 2026-07-30: 컨테이너 cgroup 쿼터 실측 7.65코어 -- 순차 생성이 장당 13.5초로
             # 너무 느려서(8000장=~30시간) 코어 수만큼 병렬 프로세스로 분리. xvfb-run -a는
             # 동시 호출 시 빈 디스플레이 번호를 알아서 잡아주므로 충돌 없음.

STAGE_NAME=r1_foundation
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=20
FREEZE_EPOCHS=6
GATE=90
MARKOV_TABLE=/workspace/round3train/markov_transitions.json
MARKOV_BIAS=0.5

COMMON_ARGS=(--single-staff --min-measures 1 --max-measures 4 --chord-prob 0.08
             --chord-min-notes 2 --chord-max-notes 3 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.30
             --markov-bias "$MARKOV_BIAS" --markov-table "$MARKOV_TABLE")
             # tie-prob 0.30(2026-07-30 추가) -- local_pools(tie1)에서 이미 검증된 값 그대로.
             # cross-register-prob은 단일오선엔 개념이 없어서(치/베이스 스왑 자체가 불가능)
             # Round1엔 넣지 않음 -- Round2(대보표)에서만 의미 있음.

echo "[$STAGE_NAME] 데이터 풀 확인 중..."
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
MIN_OK_N=$(( COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[$STAGE_NAME] 데이터 풀 이미 존재(${EXISTING_N}/${COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"
  mkdir -p "$POOL_DIR"
  echo "[$STAGE_NAME] 데이터 생성 중 (${COUNT}장, ${N_SHARDS}개 병렬, markov-bias=${MARKOV_BIAS})..."
  SHARD_BASE=$(( COUNT / N_SHARDS ))
  PIDS=()
  for i in $(seq 0 $((N_SHARDS - 1))); do
    SHARD_START=$(( START_IDX + i * 100000 ))
    SHARD_SEED=$(( SEED + i ))
    THIS_COUNT=$SHARD_BASE
    if [ "$i" -eq $((N_SHARDS - 1)) ]; then
      THIS_COUNT=$(( COUNT - SHARD_BASE * (N_SHARDS - 1) ))
    fi
    xvfb-run -a python3 /workspace/round3train/generate_scores.py \
      --count "$THIS_COUNT" --output "$POOL_DIR" --musescore "$MUSESCORE" \
      --start-idx "$SHARD_START" --seed "$SHARD_SEED" "${COMMON_ARGS[@]}" \
      > "/workspace/gen_shard_${STAGE_NAME}_${i}.log" 2>&1 &
    PIDS+=($!)
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
  done
  if [ "$FAIL" -ne 0 ]; then
    echo "[$STAGE_NAME] 일부 샤드 생성 실패 -- /workspace/gen_shard_${STAGE_NAME}_*.log 확인 필요"
    echo "PIPELINE_STOPPED_GENFAIL" >> "$STATUS"
    exit 1
  fi
  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[$STAGE_NAME] 데이터 풀 완성: ${FINAL_N}/${COUNT}장"
fi

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (처음부터, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS" --no_augment
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
echo "[$STAGE_NAME] 주의: 이 숫자는 TF 기준이라 신뢰 불가 -- 반드시 eval_page_noise.py/eval_token_acc.py로 실측 재검증할 것"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) Round 1 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
