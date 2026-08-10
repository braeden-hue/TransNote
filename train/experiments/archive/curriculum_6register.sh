#!/bin/bash
# 교차 음역(덧줄 많은 케이스) 복구 커리큘럼 (6reg1): 2026-07-28 로컬 점검에서, 대보표 두
# 오선이 모두 같은 음역(둘 다 치음역/둘 다 베이스음역)이거나 서로 뒤바뀐 경우(치가 낮은
# 음역, 베이스가 높은 음역) -- 즉 덧줄이 많이 붙는 케이스 -- 를 seq2seq_noise2_best.pt에
# 돌려보니 완전일치 0/10, 특히 "실제 음높이(덧줄 개수)"를 무시하고 그 음자리표의 전형적
# 음역(치=C4대, 베이스=B3~C2대)으로 회귀하는 뚜렷한 편향을 확인함. 클렙 판독 자체는
# 정상이었고(clef-G/clef-F 토큰 자체는 GT와 일치), 리듬(duration)도 대부분 맞아서 --
# 문제가 정확히 "그 클렙 기준으로 덧줄을 세어 정확한 음이름을 계산하는" 부분에 국한됨.
#
# generate_scores.py에 CROSS_REGISTER_PROB/--cross-register-prob 추가함(같은 파일의
# SHORT_NOTE_BIAS/DOTTED8_BIAS와 동일 패턴) -- 이 확률로 치/베이스 피치 풀을 swap(치가
# 베이스 음역, 베이스가 치음역) / both_high(둘 다 치음역) / both_low(둘 다 베이스음역)
# 중 하나로 바꿔서 덧줄 많은 케이스를 생성. 클렙 표기 자체(치 위/베이스 아래)는 항상
# 정상 유지 -- 바뀌는 건 그 안에 실제로 찍히는 피치뿐.
#
# 사용자 요청에 따라 다음 순서로 진행: 이 교차음역 복구 단계(6reg1)를 먼저 통과시킨 뒤,
# 원래 계획했던 리듬 밀도 단계(curriculum_7density.sh -- 8분/16분음표 2·4개 묶음 +
# 점8분+16분음표 리듬 셀)로 이어감. curriculum_7density.sh의 RESUME은 이 단계(6reg1)
# 산출물을 사용하도록 이미 갱신해둠.
#
# *** 이 파일은 로컬(포드 미접속 상태)에서 설계만 해둔 것 -- 아래 사전 조건을
# 포드 재접속 후 반드시 먼저 확인/수정하고 실행할 것. ***
#
# 사전 조건(포드 SSH 재접속 후 직접 확인 후 실행할 것):
#   - RESUME_NAME/RESUME_OUT: secrets/checkpoints 백업 기준 최신은 noise2(chopin_style에서
#     resume)이지만, 포드 쪽 /workspace/models/round1_curriculum_noise2가 실제로 학습
#     완료(TRAIN_DONE) 상태인지 먼저 확인. 진행 중/실패면 chopin_style로 되돌릴 것.
#   - GRAND_START_IDX/SINGLE_START_IDX/SEED: /workspace/data 아래 기존 풀들의 최대 인덱스를
#     확인해서 충돌 없는 범위로 교체할 것(아래 숫자는 자리표시자).
#   - round3train/generate_scores.py 최신본(--cross-register-prob 포함)이 포드에 배포됐는지 확인.
#   - disk quota/파이프라인 상태 먼저 확인.

set -uo pipefail
LOG=/workspace/curriculum_6register.log
STATUS=/workspace/curriculum_6register_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 교차 음역(덧줄 많은 케이스) 복구 커리큘럼 시작 (6reg1) ==="
: > "$STATUS"

POOL_DIR=/workspace/data/round1_stage6reg1_pool
GRAND_COUNT=1500     # 60% -- 대보표 (이번 단계 목표: 교차음역은 대보표에서만 의미 있음)
SINGLE_COUNT=1000    # 40% -- 단일 오선 (replay, 망각 방지 -- 단일 오선엔 교차음역 개념 자체가 없음)
                     # (2026-07-30: 스텝당 총 2500장으로 축소, 60/40 비율은 유지)
GRAND_START_IDX=4500001   # TODO: 포드에서 기존 풀 최대 인덱스 확인 후 교체(자리표시자)
SINGLE_START_IDX=4700001  # TODO: 위와 동일
SEED=318001               # TODO: 위와 동일

RESUME_NAME=noise2        # TODO: 포드에서 실제 완료 여부 확인 후 필요시 chopin_style로 교체
RESUME_OUT=/workspace/models/round1_curriculum_noise2
STAGE_NAME=6reg1
TRAIN_OUT=/workspace/models/round1_curriculum_p2s${STAGE_NAME}
EPOCHS=15
FREEZE_EPOCHS=2
NOISE_LEVEL=2
GATE=90
SPAN_WEIGHT=1              # 옥타브/헤어핀 생성 안 하므로 무의미(기존 단계들과 동일)
CROSS_REGISTER_PROB=0.35   # 대보표의 35%가 swap/both_high/both_low 중 하나 -- 나머지 65%는
                           # 정상 음역이라 기존 정상 케이스 인식률을 해치지 않으면서 노출을 확보

# 공통 스코프 인자(현재 확정 데모 스코프, 2026-07-22 + 이번 단계의 cross-register-prob)
# 2026-07-30: --hairpin-prob/--ottava-prob 0을 대보표 호출에만 붙이던 걸 여기 공통으로
# 옮김 -- 단일오선 호출엔 안 붙어있어서 크레센도가 새고 있었음(사용자 지적). 이제
# 크레센도/디크레센도·옥타브 기호는 대보표/단일오선 둘 다 완전히 안 나옴.
# --preferred-register-prob 0.7: range.mscz로 지정한 "덧줄 많이 필요한" 선호 구간(치
# D3~A3/A5~A6, 베이스 C2~E2/F4~B4)에서 70% 확률로 음을 뽑음 -- 이 스텝의 목표(교차음역/
# 덧줄 노출)와 직결.
COMMON_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.08 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0
             --preferred-register-prob 0.7)

echo "[6reg1] 데이터 풀 확인 중..."
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
TOTAL_COUNT=$((GRAND_COUNT + SINGLE_COUNT))
MIN_OK_N=$(( TOTAL_COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[6reg1] 데이터 풀 이미 존재(${EXISTING_N}/${TOTAL_COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"

  echo "[6reg1] 대보표(60%, ${GRAND_COUNT}장, cross-register-prob=${CROSS_REGISTER_PROB}) 생성 중..."
  # --density-break 제거(2026-07-30): 실제 카메라 캡처가 항상 시스템 1개만 담는데
  # (guided_camera_screen.dart), density-break는 내용이 조밀하면 의도적으로 시스템을
  # 2개 이상으로 쪼갬 -- 라벨-이미지 매칭 자체는 맞더라도 학습 이미지 형태가 실사용과
  # 안 맞아서 아예 안 씀. generate_scores.py의 wide_page_grand.mss 적용으로 이제
  # --density-break 없이도(즉 이 호출처럼) 내용이 조밀해도 조용히 2번째 시스템으로
  # 밀려나지 않고 항상 한 시스템에 들어감(2026-07-30 로컬 실측 확인).
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$GRAND_COUNT" \
    "${COMMON_ARGS[@]}" \
    --cross-register-prob "$CROSS_REGISTER_PROB" \
    --start-idx "$GRAND_START_IDX" --seed "$SEED"
  if [ $? -ne 0 ]; then
    echo "[6reg1] 대보표 데이터 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:grand" >> "$STATUS"
    exit 1
  fi

  echo "[6reg1] 단일 오선(40%, ${SINGLE_COUNT}장, replay) 생성 중..."
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$SINGLE_COUNT" \
    "${COMMON_ARGS[@]}" --single-staff \
    --start-idx "$SINGLE_START_IDX" --seed "$((SEED + 1))"
  if [ $? -ne 0 ]; then
    echo "[6reg1] 단일 오선 데이터 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:single" >> "$STATUS"
    exit 1
  fi

  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[6reg1] 데이터 풀 완성: ${FINAL_N}/${TOTAL_COUNT}장 (대보표 ${GRAND_COUNT} + 단일오선 ${SINGLE_COUNT})"
fi

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=${RESUME_NAME}, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}) ==="

echo "[$STAGE_NAME] quota 확인 중..."
if ! dd if=/dev/zero of=/workspace/_qcheck bs=1M count=100 2>/dev/null; then
  echo "[$STAGE_NAME] 경고: quota 초과 위험 -- 파이프라인 중단"
  rm -f /workspace/_qcheck
  echo "PIPELINE_STOPPED_QUOTA:$STAGE_NAME" >> "$STATUS"
  exit 1
fi
rm -f /workspace/_qcheck

mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$POOL_DIR" --out_dir "$TRAIN_OUT" \
  --resume "$RESUME_OUT/seq2seq_best.pt" \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS" --noise_level "$NOISE_LEVEL" --span_weight "$SPAN_WEIGHT"
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
  echo "[$STAGE_NAME] 학습 프로세스 비정상 종료(exit=$TRAIN_RC) -- 파이프라인 중단"
  echo "PIPELINE_STOPPED_TRAINFAIL:$STAGE_NAME" >> "$STATUS"
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
echo "[$STAGE_NAME] 학습 완료 -- best val_acc = ${BEST_ACC}% (gate=${GATE}%)"

# 전체 val_acc(토큰 평균)엔 교차음역 케이스 하나의 개선 여부가 묻힘 -- 성공/실패와
# 무관하게 항상 error_breakdown으로 pitch_wrong 계열 오류(같은 옥타브인지/다른 음이름인지)를
# 별도 확인한다. 필요하면 이 단계 전용으로 만든 소량 교차음역 테스트셋을 --data_dir로
# 따로 넣어 재확인할 것(이번 학습 풀엔 정상 케이스도 65% 섞여있어 희석됨).
echo "[$STAGE_NAME] 오류 분석(전체 풀 기준) 확인 중..."
python3 /workspace/round3train/error_breakdown.py \
  --seq2seq "$TRAIN_OUT/seq2seq_best.pt" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --data_dir "$POOL_DIR" \
  > "$TRAIN_OUT/error_breakdown.log" 2>&1
echo "[$STAGE_NAME] 오류 분석 결과: $TRAIN_OUT/error_breakdown.log (pitch_wrong_* 비율 확인)"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) 6reg1 완료 -- 교차 음역 복구 커리큘럼 종료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
