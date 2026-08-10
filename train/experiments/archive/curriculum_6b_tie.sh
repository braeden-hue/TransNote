#!/bin/bash
# 붙임줄(tie) 복구 커리큘럼 (tie1): 2026-07-30, 사용자가 이전 학습 실행에서 붙임줄이 있는
# 구간을 모델이 인식하지 못했고, 그 여파로 붙임줄 이후에 이어지는 음표 인식까지 연쇄적으로
# 무너졌다고 보고함(오토리그레시브 디코더 특성상 한 지점의 오류가 그 뒤 컨텍스트 전체를
# 오염시켰을 가능성). 원인은 tokenizer258.json에 애초에 tie 토큰이 없었고
# (mscz_to_tokens.py도 music21 Note.tie를 조회하지 않아서, 실사 정답 mscz에 실제 붙임줄이
# 있어도 각 조각이 독립된 반복음처럼 라벨링됐음) -- generate_scores.py도 마디를 넘는 음표를
# 아예 생성하지 않아 합성 데이터에도 tie 노출이 전무했음.
#
# vocab에 `tie` 토큰 1개 추가로 해결(258->259, tokenizer258.json). generate_scores.py에
# TIE_PROB/--tie-prob 추가 -- 마디 끝 음표를 다음 마디 첫 음표와 같은 피치로 강제하고
# 실제 music21 Tie 객체도 삽입(렌더링에 반영됨). single-staff/grand-staff 양쪽 다 적용.
# error_breakdown.py에 tie_context_mask() 추가 -- tie 토큰 자신 + 착지점 음표의 정확도만
# 따로 리포트(dotted8_context_mask와 동일 패턴, 이 단계가 실제로 보고된 문제를 고쳤는지
# 직접 확인하기 위함).
#
# *** 순서(2026-07-30, 사용자 확인): 실전에서 관찰된 연쇄 붕괴라는 근거로 이 단계를
# curriculum_7density.sh(리듬 밀도)보다 먼저 통과시키기로 함. curriculum_6register.sh(교차
# 음역)는 이미 그보다 더 근본적인 편향이라 판단해 먼저였으므로, 최종 순서는
# 6register(6reg1) -> 이 파일(tie1) -> 7density(7den1). curriculum_7density.sh의 RESUME도
# tie1 산출물을 쓰도록 갱신함(기존엔 6reg1을 가리켰음). ***
#
# *** vocab이 258->259로 실제로 늘어나는 유일한 단계라 --resume_tokenizer가 필수임 --
# 6reg1 체크포인트는 258 vocab으로 학습됐으므로, train.py의 load_ckpt_partial_vocab()이
# 토큰 문자열 기준으로 겹치는 258개 토큰의 가중치를 이어받고 새 tie 행만 랜덤 초기화하도록
# tokenizer258_pre_tie.json(258개, tie 없음 -- git HEAD 스냅샷)을 넘겨야 함. 이 단계 이후
# (7density 등)는 현재 tokenizer258.json(259개)로 vocab이 고정되므로 --resume_tokenizer 불필요. ***
#
# *** 이 파일은 로컬(포드 미접속 상태)에서 설계만 해둔 것 -- 아래 사전 조건을
# 포드 재접속 후 반드시 먼저 확인/수정하고 실행할 것. ***
#
# 사전 조건(포드 SSH 재접속 후 직접 확인 후 실행할 것):
#   - RESUME_NAME/RESUME_OUT: curriculum_6register.sh(6reg1)이 실제로 GATE(90%) 통과했는지
#     curriculum_6register_status.txt로 확인. 통과했으면 6reg1 유지, 미통과/미완료면
#     noise2로 되돌릴 것(단, noise2는 258 vocab이므로 그 경우도 --resume_tokenizer는 그대로 필요).
#   - GRAND_START_IDX/SINGLE_START_IDX/SEED: /workspace/data 아래 기존 풀들의 최대 인덱스를
#     확인해서 충돌 없는 범위로 교체할 것(6reg1은 450만/470만대 사용 -- 아래 숫자는 자리표시자).
#   - round3train/{generate_scores.py,mscz_to_tokens.py,error_breakdown.py,tokenizer258.json,
#     tokenizer258_pre_tie.json} 최신본(tie 토큰/TIE_PROB/tie_context_mask 포함)이 포드에
#     배포됐는지 확인 -- tokenizer258.json이 259개인지(구버전 258개 그대로 남아있지 않은지)
#     반드시 재확인.
#   - disk quota/파이프라인 상태(curriculum_6register_status.txt에 PIPELINE_STOPPED_* 있는지)
#     먼저 확인.

set -uo pipefail
LOG=/workspace/curriculum_6b_tie.log
STATUS=/workspace/curriculum_6b_tie_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 붙임줄(tie) 복구 커리큘럼 시작 (tie1) ==="
: > "$STATUS"

POOL_DIR=/workspace/data/round1_stagetie1_pool
GRAND_COUNT=1250     # 50% -- 대보표 (실사 정답 mscz 대부분 그랜드 스태프라 노출 확보)
SINGLE_COUNT=1250    # 50% -- 단일 오선 (replay + 이번 단계 목표 동시 적용)
                     # (2026-07-30: 스텝당 총 2500장으로 축소, 50/50 비율은 유지)
GRAND_START_IDX=5300001   # TODO: 포드에서 기존 풀 최대 인덱스 확인 후 교체(자리표시자,
                          # 7den1이 490만/510만대를 쓸 예정이었으나 이 단계가 그 앞으로
                          # 와서 6reg1(450만/470만대) 다음 대역을 씀)
SINGLE_START_IDX=5500001  # TODO: 위와 동일
SEED=331001               # TODO: 위와 동일

RESUME_NAME=6reg1    # TODO: 6reg1 GATE 통과 여부 확인 후 필요시 noise2로 교체
RESUME_OUT=/workspace/models/round1_curriculum_p2s6reg1
RESUME_TOKENIZER=/workspace/round3train/tokenizer258_pre_tie.json  # 258개(tie 없음) -- 필수
STAGE_NAME=tie1
TRAIN_OUT=/workspace/models/round1_curriculum_p2s${STAGE_NAME}
EPOCHS=15
FREEZE_EPOCHS=2
NOISE_LEVEL=2
GATE=90
SPAN_WEIGHT=1        # 옥타브/헤어핀 생성 안 하므로 무의미(이전 단계들과 동일)
TIE_PROB=0.30         # 마디 30%가 다음 마디로 붙임줄이 이어짐 -- 나머지 70%는 정상이라
                      # 기존 정상 케이스 인식률을 해치지 않으면서 노출을 확보
CARRY_CROSS_REGISTER_PROB=0.15   # 2026-07-30(사용자 요청): Step 1의 핵심 축을 여기서도
                                  # 낮은 확률로 계속 섞어서 "체크포인트 연속성에만 의존"하지
                                  # 않고 데이터로도 누적 -- 6reg1 자체의 0.35보다 낮춰서 이번
                                  # 단계의 주 목표(tie)를 가리지 않게 함. 대보표에만 의미 있음
                                  # (단일오선엔 교차음역 개념 자체가 없음, 6reg1과 동일 이유).

# 공통 스코프 인자(현재 확정 데모 스코프, 2026-07-22 + 이번 단계의 tie-prob)
# 2026-07-30: --hairpin-prob/--ottava-prob 0을 공통으로 옮김(대보표 호출에만 붙어있어서
# 단일오선에 크레센도가 새고 있었음, curriculum_6register.sh와 동일 이유).
COMMON_ARGS=(--min-measures 2 --max-measures 4 --chord-prob 0.08 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
             --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
             --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob "$TIE_PROB")
             # min-measures 2 고정: tie는 최소 2마디가 있어야 의미가 있음(마지막 마디에서는
             # 시작 안 함 -- generate_scores.py TIE_PROB 로직 참고)

echo "[tie1] 데이터 풀 확인 중..."
EXISTING_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
TOTAL_COUNT=$((GRAND_COUNT + SINGLE_COUNT))
MIN_OK_N=$(( TOTAL_COUNT * 97 / 100 ))
if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
  echo "[tie1] 데이터 풀 이미 존재(${EXISTING_N}/${TOTAL_COUNT}장) -- 재생성 스킵"
else
  rm -rf "$POOL_DIR"

  echo "[tie1] 대보표(50%, ${GRAND_COUNT}장, tie-prob=${TIE_PROB}, cross-register-prob=${CARRY_CROSS_REGISTER_PROB}(누적)) 생성 중..."
  # --density-break 제거(2026-07-30): 실제 카메라 캡처가 항상 시스템 1개만 담으므로
  # (guided_camera_screen.dart) 학습 이미지도 항상 한 시스템이어야 함 -- 상세 이유는
  # curriculum_6register.sh 주석 참고. generate_scores.py의 wide_page_grand.mss로
  # --density-break 없이도 조밀한 내용이 조용히 2번째 시스템으로 밀려나지 않음.
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$GRAND_COUNT" \
    "${COMMON_ARGS[@]}" --cross-register-prob "$CARRY_CROSS_REGISTER_PROB" \
    --start-idx "$GRAND_START_IDX" --seed "$SEED"
  if [ $? -ne 0 ]; then
    echo "[tie1] 대보표 데이터 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:grand" >> "$STATUS"
    exit 1
  fi

  echo "[tie1] 단일 오선(50%, ${SINGLE_COUNT}장, tie-prob=${TIE_PROB}) 생성 중..."
  bash /workspace/round3train/gen_render_local.sh "$POOL_DIR" "$SINGLE_COUNT" \
    "${COMMON_ARGS[@]}" --single-staff \
    --start-idx "$SINGLE_START_IDX" --seed "$((SEED + 1))"
  if [ $? -ne 0 ]; then
    echo "[tie1] 단일 오선 데이터 생성 실패 -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_GENFAIL:single" >> "$STATUS"
    exit 1
  fi

  FINAL_N=$(find "$POOL_DIR" -maxdepth 1 -name '*.png' | wc -l)
  echo "[tie1] 데이터 풀 완성: ${FINAL_N}/${TOTAL_COUNT}장 (대보표 ${GRAND_COUNT} + 단일오선 ${SINGLE_COUNT})"
fi

echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=${RESUME_NAME}, resume_tokenizer=pre_tie(258), epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}) ==="

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
  --resume_tokenizer "$RESUME_TOKENIZER" \
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

# 전체 val_acc(토큰 평균)엔 tie 케이스 하나의 개선 여부가 묻힘 -- 성공/실패와 무관하게
# 항상 error_breakdown으로 "붙임줄(tie) 및 착지점 정확도"를 별도 확인한다. 이게 바로
# 사용자가 보고한 "tie 미인식 -> 그 뒤 음표 인식 붕괴" 증상이 실제로 고쳐졌는지 보는 지표.
echo "[$STAGE_NAME] 붙임줄 및 착지점 정확도 확인 중..."
python3 /workspace/round3train/error_breakdown.py \
  --seq2seq "$TRAIN_OUT/seq2seq_best.pt" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --data_dir "$POOL_DIR" \
  > "$TRAIN_OUT/error_breakdown.log" 2>&1
echo "[$STAGE_NAME] 오류 분석 결과: $TRAIN_OUT/error_breakdown.log ('붙임줄(tie) 및 착지점 정확도' 항목 확인)"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo ""
echo "=== $(date) tie1 완료 -- 붙임줄 복구 커리큘럼 종료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
