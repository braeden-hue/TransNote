#!/bin/bash
# 리듬 밀도 복구 커리큘럼 (7den1): 8분/16분음표가 2개·4개씩 묶인 형태 + 점8분+16분음표
# (또는 16분+점8분) 리듬 셀의 노출을 늘리는 단계. 2026-07-28 로컬 점검에서
# generate_scores.py의 DURATIONS 기본 가중치가 dur-3/16(점8분음표)에 0.5%밖에 배정돼
# 있지 않음을 확인(1/16의 8%, 3/8의 1.5%보다도 낮음) -- 사용자 요청(정확한 계이름 인식이
# 최우선)에 따라 이 리듬 셀 노출을 늘리는 별도 단계를 판다.
#
# *** 순서 변경(2026-07-28): curriculum_6register.sh(교차 음역/덧줄 많은 케이스 복구)를
# 이 단계보다 먼저 통과시키기로 함 -- 같은 점검에서 대보표 두 오선이 같은 음역이거나
# 뒤바뀐 경우(덧줄 많음) 모델이 실제 음높이 대신 그 클렙의 "전형적 음역"으로 회귀하는
# 더 근본적인 편향이 발견됐기 때문. 이 파일의 RESUME은 6reg1 산출물을 쓰도록 갱신함
# (기존엔 5ss1/5n5를 가리켰음). ***
#
# *** 순서 변경(2026-07-30): curriculum_6b_tie.sh(붙임줄 복구)를 6reg1과 이 단계 사이에
# 끼워 넣기로 함 -- 사용자가 이전 학습에서 tie 미인식이 그 뒤 음표 인식까지 연쇄적으로
# 무너뜨렸다고 보고(실전에서 관찰된 붕괴라 리듬 밀도보다 우선). 이 파일의 RESUME을 6reg1
# 대신 tie1 산출물을 쓰도록 갱신함. tie1은 vocab을 258->259로 늘렸지만 이 단계는 그
# 이후이므로(둘 다 현재 tokenizer258.json=259개 사용) --resume_tokenizer 불필요 -- vocab이
# 실제로 달라지는 건 6reg1->tie1 전환 시점뿐. ***
#
# generate_scores.py에 DOTTED8_BIAS/--dotted8-bias, SHORT_NOTE_BIAS/--short-note-bias
# (기존에 있던 것) 둘 다 사용 -- --dotted8-bias 10.0이면 dur-3/16 가중치가 (1+10)배 ->
# 약 5.2%로 상승(1/16의 7.6%와 비슷한 수준). --short-note-bias 2.0이면 1/8·1/16
# 가중치가 3배로 올라 8분/16분음표 2개·4개 묶음(연속 실행) 빈도가 뚜렷하게 늘어남 --
# 2026-07-28 로컬 10장 테스트(local_dense_pool)에서 실측: 8분음표 2·3·4개 연속, 16분음표
# 2·3개 연속, 점8분+16분 조합 모두 등장 확인.
#
# error_breakdown.py에 dotted8_context_mask()를 추가해서 "점8분음표 자신 + 그 바로 다음
# 음(통상 16분음표)"에 해당하는 note- 토큰만 따로 골라 계이름 정확도를 리포트하도록 함 --
# 전체 val_acc(토큰 평균)만 보면 이 리듬 셀 하나의 개선 여부가 묻히므로, 학습 성공 여부와
# 무관하게 항상 이 지표를 같이 출력.
#
# *** 이 파일은 로컬(포드 미접속 상태)에서 설계만 해둔 것 -- 아래 사전 조건을
# 포드 재접속 후 반드시 먼저 확인/수정하고 실행할 것. ***
#
# 사전 조건(포드 SSH 재접속 후 직접 확인 후 실행할 것):
#   - RESUME_NAME/RESUME_OUT: curriculum_6b_tie.sh(tie1)이 실제로 GATE(90%) 통과했는지
#     curriculum_6b_tie_status.txt로 확인. 통과했으면 tie1 유지, 미통과/미완료면 6reg1로
#     되돌릴 것(단, 그 경우 tie 커리큘럼을 건너뛰게 되므로 tie1 실패 원인을 먼저 파악하는
#     쪽을 우선할 것 -- 붙임줄 문제가 실전에서 관찰된 연쇄 붕괴이기 때문).
#   - GRAND_START_IDX/SINGLE_START_IDX/SEED: /workspace/data 아래 기존 풀들의 최대 인덱스를
#     확인해서 충돌 없는 범위로 교체할 것(6reg1은 450만/470만대, tie1은 530만/550만대 사용
#     예정 -- 여기 숫자는 자리표시자).
#   - round3train/{generate_scores.py,error_breakdown.py} 최신본(--dotted8-bias,
#     --cross-register-prob, --tie-prob, dotted8_context_mask, tie_context_mask 포함)이
#     포드에 배포됐는지 확인. tokenizer258.json이 259개(tie 포함)인지도 확인.
#   - disk quota/파이프라인 상태(curriculum_6b_tie_status.txt에 PIPELINE_STOPPED_*
#     있는지) 먼저 확인.

set -uo pipefail
LOG=/workspace/curriculum_7density.log
STATUS=/workspace/curriculum_7density_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 리듬 밀도(8분/16분 묶음 + 점8분+16분) 복구 커리큘럼 시작 (7den1) ==="
: > "$STATUS"

POOL_DIR=/workspace/data/round1_stage7den1_pool
GRAND_COUNT=1250     # 50% -- 대보표 (replay + 이번 단계 목표 동시 적용)
SINGLE_COUNT=1250    # 50% -- 단일 오선 (5ss1 회복분 replay + 이번 단계 목표 동시 적용)
                     # (2026-07-30: 스텝당 총 2500장으로 축소, 50/50 비율은 유지)
GRAND_START_IDX=5700001   # TODO: 포드에서 기존 풀 최대 인덱스 확인 후 교체(자리표시자,
                          # tie1이 530만/550만대를 쓰므로 그 다음 대역)
SINGLE_START_IDX=5900001  # TODO: 위와 동일
SEED=338001               # TODO: 위와 동일

RESUME_NAME=tie1     # TODO: tie1 GATE 통과 여부 확인 후 필요시 6reg1로 교체
RESUME_OUT=/workspace/models/round1_curriculum_p2stie1
STAGE_NAME=7den1
TRAIN_OUT=/workspace/models/round1_curriculum_p2s${STAGE_NAME}
EPOCHS=15
FREEZE_EPOCHS=2
NOISE_LEVEL=2
GATE=90
SPAN_WEIGHT=1        # 옥타브/헤어핀 생성 안 하므로 무의미(이전 단계들과 동일)
DOTTED8_BIAS=10.0     # dur-3/16 가중치 0.5% -> 약 5.2%로 상승
SHORT_NOTE_BIAS=2.0   # 1/8·1/16 가중치 3배 -> 8분/16분음표 2개·4개 연속 묶음 빈도 상승
CARRY_TIE_PROB=0.15             # 2026-07-30(사용자 요청): Step 1.5의 핵심 축을 낮은 확률로
                                 # 계속 섞음(원래 tie1의 0.30보다 낮춤 -- 이번 단계 주 목표인
                                 # 밀도를 가리지 않게). 대보표/단일오선 둘 다 적용(tie는
                                 # 오선 형태 무관).
CARRY_CROSS_REGISTER_PROB=0.15  # Step 1의 핵심 축도 낮은 확률로 계속 섞음(6reg1의 0.35보다
                                 # 낮춤). 대보표에만 의미 있음(단일오선엔 개념 자체가 없음).

# 공통 스코프 인자(현재 확정 데모 스코프, 2026-07-22 + 이번 단계의 밀도 바이어스)
# 2026-07-30: --hairpin-prob/--ottava-prob 0을 공통으로 옮김(대보표 호출에만 붙어있어서
# 단일오선에 크레센도가 새고 있었음, curriculum_6register.sh와 동일 이유). --tie-prob도
# 공통으로 이동(누적 -- 아래 CARRY_TIE_PROB 주석 참고).
COMMON_ARGS=(--min-measures 1 --max-measures 4 --chord-prob 0.08 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob "$CARRY_TIE_PROB"
             --dotted8-bias "$DOTTED8_BIAS" --short-note-bias "$SHORT_NOTE_BIAS")

echo "[7den1] 데이터 풀 확인 중..."
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
TOTAL_COUNT=$((GRAND_COUNT + SINGLE_COUNT))
MIN_OK_N=$(( TOTAL_COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[7den1] 데이터 풀 이미 존재(${EXISTING_N}/${TOTAL_COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"

  echo "[7den1] 대보표(50%, ${GRAND_COUNT}장, tie-prob=${CARRY_TIE_PROB}(누적), cross-register-prob=${CARRY_CROSS_REGISTER_PROB}(누적)) 생성 중..."
  # --density-break 제거(2026-07-30): 실제 카메라 캡처가 항상 시스템 1개만 담으므로
  # (guided_camera_screen.dart) 학습 이미지도 항상 한 시스템이어야 함 -- 상세 이유는
  # curriculum_6register.sh 주석 참고. generate_scores.py의 wide_page_grand.mss로
  # --density-break 없이도 조밀한 내용(이 스텝의 short-note-bias/dotted8-bias 포함)이
  # 조용히 2번째 시스템으로 밀려나지 않음(2026-07-30 로컬 실측으로 이 정확한 케이스에서
  # 문제 재현+수정 확인).
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$GRAND_COUNT" \
    "${COMMON_ARGS[@]}" --cross-register-prob "$CARRY_CROSS_REGISTER_PROB" \
    --start-idx "$GRAND_START_IDX" --seed "$SEED"
  if [ $? -ne 0 ]; then
    echo "[7den1] 대보표 데이터 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:grand" >> "$STATUS"
    exit 1
  fi

  echo "[7den1] 단일 오선(50%, ${SINGLE_COUNT}장) 생성 중..."
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$SINGLE_COUNT" \
    "${COMMON_ARGS[@]}" --single-staff \
    --start-idx "$SINGLE_START_IDX" --seed "$((SEED + 1))"
  if [ $? -ne 0 ]; then
    echo "[7den1] 단일 오선 데이터 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:single" >> "$STATUS"
    exit 1
  fi

  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[7den1] 데이터 풀 완성: ${FINAL_N}/${TOTAL_COUNT}장 (대보표 ${GRAND_COUNT} + 단일오선 ${SINGLE_COUNT})"
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

# 전체 val_acc는 토큰 평균이라 "점8분+16분" 리듬 셀 하나의 개선 여부가 묻힘.
# 성공/실패와 무관하게 항상 이 리듬 셀 전용 계이름 정확도를 별도로 확인한다.
echo "[$STAGE_NAME] 점8분+16분음표 리듬 셀 계이름 정확도 확인 중..."
python3 /workspace/round3train/error_breakdown.py \
  --seq2seq "$TRAIN_OUT/seq2seq_best.pt" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --data_dir "$POOL_DIR" \
  > "$TRAIN_OUT/error_breakdown.log" 2>&1
echo "[$STAGE_NAME] 오류 분석 결과: $TRAIN_OUT/error_breakdown.log ('점8분+16분음표 리듬 셀 계이름 정확도' 항목 확인)"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) 7den1 완료 -- 리듬 밀도 복구 커리큘럼 종료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
