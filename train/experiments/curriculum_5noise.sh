#!/bin/bash
# 노이즈(촬영 시뮬레이션) 강건성 커리큘럼: 5n0(데이터 풀 준비) -> 5n1~5n4(--noise_level 1~4 단계적 상승)
# -> 5n5(2026-07-24 추가, --page_level_noise)
#
# 5n5 배경: 5n1~5n4는 캔버스에 직접 약한 기하 노이즈만 줬는데, 실제 추론 경로는
# 페이지 레벨 노이즈 -> correct_perspective(방금 이식) -> 오선 재검출을 거친다. 로컬 100장
# 테스트(2026-07-24)에서 correct_perspective 적용만으로는 오선 검출 실패가 60.6%->23%로
# 줄어도 완전 일치율은 9.1%->8.0%로 그대로였다 -- 인식 모델이 "노이즈+보정을 거친 캔버스"
# 자체를 학습해본 적이 없기 때문. 5n5는 OMRDataset.page_level_noise=True로 그 실제 분포를
# 30%가량의 샘플에 재현해서 학습시킨다(재검출 실패 시 기존 캔버스 레벨 경로로 자동 폴백,
# dataset.page_noise_and_redetect 참고).
#
# curriculum_4t_4sym.sh류(심볼 커리큘럼)와의 핵심 차이:
#   심볼 커리큘럼은 매 단계 "새 문법"을 배우므로 단계마다 새 데이터를 생성하고 이전 데이터는
#   replay용 500장만 남기고 정리했다. 이 커리큘럼은 새 기호를 배우는 게 아니라 이미 학습된
#   전체 스코프(4tup까지 포함) 위에서 촬영 노이즈(기울기/블러/조명/압축) 강도만 올리는 것이라,
#   5n0에서 최종 스코프 데이터 풀을 한 번만 크게 만들고 5n1~5n4가 그 풀을 그대로 재사용한다.
#   replay_dir/replay_count도, 단계 종료 후 디스크 정리도 없음.
#
# freeze_epochs=0 (심볼 커리큘럼과 다름): 새 어휘를 배우는 게 아니라 인코더(CNN)가 노이즈 낀
# 픽셀 입력에 적응해야 하는 문제라, 인코더를 얼리면 정작 적응해야 할 부분의 학습 시간만 깎인다.
#
# 게이트 임계값은 4단계 모두 95%로 고정(요청에 따름). 참고로 직전 심볼 커리큘럼 최종 단계
# (4span_w10, 노이즈 없는 조건)의 실측 best val_acc는 94.2%였는데, 그 측정엔 옥타브+헤어핀
# (크레센도/디크레센도) 8개 토큰이 포함돼 있었다(train.py SPAN_TOKENS, 확인 완료). 이 커리큘럼은
# 그 두 기호를 데이터 생성에서 아예 뺐으므로(2026-07-22 데모 스코프 제외 확정) 94.2%보다 순수
# 상승 여지는 있지만, 노이즈를 새로 얹는 효과와 상쇄될 수 있어 95% 통과를 장담할 순 없다.
# val_acc는 idx 기반으로 시드를 고정해 매 epoch 같은 노이즈로 측정되므로(dataset.split_dataset/
# _frozen_rng) 순수 측정 분산은 아니고, 못 넘기면 아래처럼 다음 단계로 안 넘어가고 멈춰서
# error_breakdown.py로 원인(어떤 토큰/음표가 틀렸는지)을 남긴다.
#
# 사전 조건(직접 확인 후 실행할 것):
#   - round3train/{dataset.py,train.py,generate_scores.py,error_breakdown.py,gen_render_local.sh} 최신본 배포됨
#     (dataset.py는 NOISE_LEVELS[레벨]에 max_concurrent 키가 있는 버전이어야 함 -- 0단계에서 자동 확인.
#     5n5 실행 전엔 correct_perspective/page_noise_and_redetect가 있는 버전인지도 확인할 것)
#   - PREV_NAME_INITIAL/PREV_OUT_INITIAL: pod SSH로 직접 확인 완료(/workspace/models/round1_curriculum_p2s4span_w10 실존)
#   - SPAN_WEIGHT=1(비활성)로 둔다 -- 옥타브/헤어핀을 5n0에서 아예 생성 안 하므로 그 8개 토큰이
#     학습 데이터에 등장하지 않아 가중치를 줘도 의미가 없음. resume 체크포인트 자체는
#     span_weight=10으로 학습된 상태지만 그건 그 체크포인트 학습 당시 얘기고 이후 미적용.
#   - 5n0 데이터 풀은 disk 정리로 삭제됐으므로(2026-07-23) 이번 실행에서 자동 재생성됨(~40분)
#
# 5n5 gate=65% (2026-07-25 재조정, 90->65) -- 5n5 학습 전 로컬 베이스라인 실측: 5n4+
# correct_perspective만으로 이미 L2 토큰정확도 76.7%/L3 41.5%(예: round3train/eval_token_acc.py
# 결과). L3 쪽이 훨씬 낮게 시작하고 5n5의 val_acc는 L2~L3를 섞어서 측정하므로, 90%는 "성공한
# 학습도 gate에서 걸려 멈추는" 오탐 위험이 큼. 65%는 "발산하지 않았는지"만 보는 최소 안전장치고,
# 진짜 검증은 학습 후 로컬 100장 재테스트(eval_token_acc.py)로 한다(사용자 확인, 2026-07-24).
#
# RESUME_FROM 환경변수로 중간 단계부터 재개 가능 (0=5n1부터, 1=5n2부터, 2=5n3부터, 3=5n4부터,
# 4=5n5부터, 5=5n6부터 -- 단 5n6는 resume 소스가 5n4로 강제 오버라이드됨, 위 5n6 코멘트 참고).
# 5n0(데이터 생성)는 RESUME_FROM 값과 무관하게 항상 먼저 존재 여부를 확인한다.

set -uo pipefail
LOG=/workspace/curriculum_5noise.log
STATUS=/workspace/curriculum_5noise_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 노이즈 강건성 커리큘럼 시작 (5n0~5n4) ==="
: > "$STATUS"

# --- 0단계: 배포된 코드가 최신인지 확인 (max_concurrent/page_noise_and_redetect 없으면 구버전) ---
cd /workspace/round3train
python3 -c "from dataset import NOISE_LEVELS, page_noise_and_redetect, correct_perspective; assert 'max_concurrent' in NOISE_LEVELS[2]" \
  || { echo "[0단계] dataset.py가 구버전입니다 -- 최신본 배포 후 재실행하세요"
       echo "PIPELINE_STOPPED_STALE_CODE" >> "$STATUS"
       exit 1; }
echo "[0단계] 배포 코드 확인 완료 (NOISE_LEVELS.max_concurrent + page_noise_and_redetect 존재)"

STAGE_NAMES=(5n1 5n2 5n3 5n4 5n5 5n6)
EPOCHS=(10 12 15 15 15 15)
FREEZE_EPOCHS=(0 0 0 0 0 2)   # 5n6만 2 -- 아래 5n6 코멘트 참고
NOISE_LEVELS_ARR=(1 2 3 4 2 2)
NOISE_LEVEL_MAX_ARR=(0 0 0 0 3 3)   # 0=단일 레벨. 5n5/5n6는 [2,3] 범위(목표 촬영조건=약~중간 기울기)
P_LEVEL_MAX_ARR=(0 0 0 0 0 0.65)    # 0=미지정(균등 0.5). 5n6만 L3 쪽 편향 -- 아래 코멘트 참고
GATE_THRESHOLDS=(95 95 95 95 65 65)
PAGE_LEVEL_NOISE_ARR=(0 0 0 0 1 1)

# 5n6 (2026-07-25 추가): 5n5 사후분석 결과 L2는 76.7%->80.8%로 개선됐지만 L3는 오히려
# 41.5%->35.3%로 악화됐다(round3train/eval_token_acc.py 로컬 100장 재테스트). 두 가지를
# 고쳐서 재시도한다 -- (a) freeze_epochs=0->2: 5n5처럼 이질적인(L2/L3/폴백 세 갈래) 분포에
# 처음부터 인코더 전체를 노출시키면 불안정할 수 있어 짧은 워밍업을 둠 (b) p_level_max=0.65:
# 재검출 성공률이 L2 92% vs L3 56%로 갈려서 노이즈 레벨을 균등(0.5)으로 뽑아도 실제
# "진짜" 학습 신호는 L2 쪽에 훨씬 많이 쏠렸었다 -- 성공률 역수 비율(0.92/0.56≈1.64)만큼
# L3(=noise_level_max)를 더 자주 뽑아 실효 노출량을 맞춘다.
# **resume 소스는 5n5가 아니라 5n4** -- 5n5는 이미 L3 방향 편향이 가중치에 박혀있어서,
# 검증된 5n4에서 다시 시작해 원인을 격리한다(아래 루프 안에서 강제 오버라이드).

POOL_DIR=/workspace/data/round1_stage5n0_pool
POOL_COUNT=4000
POOL_START_IDX=3300001
POOL_SEED=305001

SPAN_WEIGHT=1   # 비활성(사전 조건 코멘트 참고) -- 옥타브/헤어핀을 생성하지 않으므로 무의미

PREV_NAME_INITIAL=4span_w10   # 실제 pod 디렉토리명과 다르면 여기 수정
PREV_OUT_INITIAL=/workspace/models/round1_curriculum_p2s4span_w10

# --- 5n0: 최종 스코프 통합 데이터 풀 (1회만, 있으면 재사용) ---
# 4tup(curriculum_4t_4sym.sh 마지막 단계) 옵션에서 옥타브/헤어핀(크레센도·디크레센도)만 뺐다.
# 데모 스코프에서 이 두 기호를 제외하기로 확정됐고(2026-07-22), 직전 span_w10 단계가
# --span_weight 10x까지 줘가며 씨름했던 게 바로 이 8개 토큰(ottava-8va/8vb start/end,
# hairpin-cresc/dim start/end)이라 -- 제외하면 val_acc가 그만큼 오를 걸로 예상.
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
MIN_OK_N=$(( POOL_COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[5n0] 데이터 풀 이미 존재(${EXISTING_N}/${POOL_COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"
  GEN_ARGS=(--min-measures 1 --max-measures 4 --density-break --chord-prob 0.08 --repeat-prob 0
            --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35 --ottava-prob 0
            --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
            --hairpin-prob 0 --fermata-prob 0.04
            --start-idx "$POOL_START_IDX" --seed "$POOL_SEED")
  echo "[5n0] 데이터 생성: ${GEN_ARGS[*]}"
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$POOL_COUNT" "${GEN_ARGS[@]}"
  if [ $? -ne 0 ]; then
    echo "[5n0] 데이터 생성/검증 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:5n0" >> "$STATUS"
    exit 1
  fi
fi

RESUME_FROM=${RESUME_FROM:-0}

if [ "$RESUME_FROM" -eq 0 ]; then
  PREV_NAME="$PREV_NAME_INITIAL"
  PREV_OUT="$PREV_OUT_INITIAL"
else
  PN=${STAGE_NAMES[$((RESUME_FROM-1))]}
  PREV_NAME="$PN"
  PREV_OUT="/workspace/models/round1_curriculum_p2s${PN}"
  echo "=== $(date) RESUME_FROM=${RESUME_FROM} (${STAGE_NAMES[$RESUME_FROM]}부터 재개, 직전=${PREV_NAME}) ==="
fi

for i in "${!STAGE_NAMES[@]}"; do
  if [ "$i" -lt "$RESUME_FROM" ]; then
    continue
  fi
  NAME=${STAGE_NAMES[$i]}
  EP=${EPOCHS[$i]}
  FRZ=${FREEZE_EPOCHS[$i]}
  NL=${NOISE_LEVELS_ARR[$i]}
  GATE=${GATE_THRESHOLDS[$i]}
  PLN_ARGS=()
  if [ "${PAGE_LEVEL_NOISE_ARR[$i]}" = "1" ]; then
    PLN_ARGS+=(--page_level_noise)
  fi
  if [ "${NOISE_LEVEL_MAX_ARR[$i]}" != "0" ]; then
    PLN_ARGS+=(--noise_level_max "${NOISE_LEVEL_MAX_ARR[$i]}")
  fi
  if [ "${P_LEVEL_MAX_ARR[$i]}" != "0" ]; then
    PLN_ARGS+=(--p_level_max "${P_LEVEL_MAX_ARR[$i]}")
  fi

  # 5n6는 5n5(직전 배열 원소)가 아니라 5n4에서 resume -- 위 5n6 코멘트 참고.
  if [ "$NAME" = "5n6" ]; then
    PREV_NAME="5n4"
    PREV_OUT="/workspace/models/round1_curriculum_p2s5n4"
    echo "[$NAME] resume 소스를 5n4로 강제 지정(5n5 아님) -- 5n5는 L3 방향 편향이 학습된 상태라 재사용 안 함"
  fi

  TRAIN_OUT=/workspace/models/round1_curriculum_p2s${NAME}

  echo ""
  echo "=== $(date) [$NAME] 시작 (noise_level=${NL}, epoch ${EP}, freeze=${FRZ}, resume=${PREV_NAME}) ==="

  # --- 1. 디스크 정리 ---
  # quota 확인 + 직전에 통과한 단계의 seq2seq_last.pt 제거(best.pt만 있으면 됨 -- last.pt는
  # 그 단계 학습 도중 재개용이었고, 이미 통과해서 다음 단계 resume은 항상 best.pt를 씀).
  echo "[$NAME] 1) 디스크 정리: quota 확인 중..."
  if ! dd if=/dev/zero of=/workspace/_qcheck bs=1M count=100 2>/dev/null; then
    echo "[$NAME] 경고: quota 초과 위험 -- 파이프라인 중단"
    rm -f /workspace/_qcheck
    echo "PIPELINE_STOPPED_QUOTA:$NAME" >> "$STATUS"
    exit 1
  fi
  rm -f /workspace/_qcheck
  if [ -f "$PREV_OUT/seq2seq_last.pt" ] && [ "$PREV_OUT" != "$PREV_OUT_INITIAL" ]; then
    rm -f "$PREV_OUT/seq2seq_last.pt"
    echo "[$NAME] 1) 디스크 정리: ${PREV_OUT}/seq2seq_last.pt 삭제(best.pt만 보존)"
  fi

  # --- 2. 체크포인트 설정(직전 단계 best.pt에서 resume) 후 이번 단계 학습 시작 ---
  echo "[$NAME] 2) 체크포인트=${PREV_OUT}/seq2seq_best.pt 에서 resume, 학습 시작"
  mkdir -p "$TRAIN_OUT"
  cd /workspace/round3train
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 -u train.py --phase 2 \
    --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
    --resume "$PREV_OUT/seq2seq_best.pt" \
    --batch 24 --epochs "$EP" --workers 16 \
    --freeze_epochs "$FRZ" --noise_level "$NL" --span_weight "$SPAN_WEIGHT" \
    "${PLN_ARGS[@]}"
  TRAIN_RC=$?
  if [ $TRAIN_RC -ne 0 ]; then
    echo "[$NAME] 학습 프로세스 비정상 종료(exit=$TRAIN_RC) -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_TRAINFAIL:$NAME" >> "$STATUS"
    exit 1
  fi

  # --- 3. 마지막 epoch 기준 정확도 게이트 ---
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
  echo "[$NAME] 학습 완료 -- best val_acc = ${BEST_ACC}% (gate=${GATE}%)"

  PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
  if [ "$PASS" != "1" ]; then
    echo "[$NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%) -- 파이프라인 중단, 오류 분석 실행"
    echo "PIPELINE_STOPPED_LOW_ACC:$NAME:$BEST_ACC" >> "$STATUS"
    python3 /workspace/round3train/error_breakdown.py \
      --seq2seq "$TRAIN_OUT/seq2seq_best.pt" \
      --tokenizer /workspace/round3train/tokenizer258.json \
      --data_dir "$POOL_DIR" \
      > "$TRAIN_OUT/error_breakdown.log" 2>&1
    echo "[$NAME] 오류 분석 결과: $TRAIN_OUT/error_breakdown.log 확인 (다음 단계 진행 안 함, 데이터 삭제 안 함)"
    exit 2
  fi

  echo "[$NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
  echo "STAGE_PASSED:$NAME:$BEST_ACC" >> "$STATUS"

  PREV_NAME="$NAME"
  PREV_OUT="$TRAIN_OUT"
done

echo ""
echo "=== $(date) 5n6까지 전체 통과 -- 노이즈 강건성 커리큘럼 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
