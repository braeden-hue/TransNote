#!/bin/bash
# 4gm 다음 단계: 박자(6/8) -> 조표(13종) -> 다이나믹(p,f,pp,ff,mp,mf) -> 기호(페르마타+헤어핀) 커리큘럼.
# 각 단계: 신규 2500장/25epoch, 직전 단계가 남겨둔 replay 데이터로 이어 학습 -> val_acc 게이트(95%)
# 통과 시 그 단계가 만든 2500장 중 500장만 남기고 나머지 삭제(다음 단계 replay용 최소 보존) -> 다음 단계.
# 게이트 미달 시 즉시 중단하고 error_breakdown.py로 오류 종류 분석(음이름/길이/note-rest 혼동/누락 등).
#
# 각 단계는 --difficulty 플래그를 쓰지 않고 모든 심볼 확률을 명시적으로 지정한다 -- 안 그러면
# --difficulty easy/medium/full이 이번 단계에서 아직 등장 안 한 축(artic/ornament/slur/tuplet/ottava)
# 뿐 아니라 이미 4gm에서 학습된 화음(chord)까지 조용히 0으로 덮어써버릴 수 있다.
#
# RESUME_FROM 환경변수로 중간 단계부터 재개 가능 (0=4t부터, 1=4k부터, 2=4dyn부터, 3=4sym부터,
# 4=4ott부터, 5=4tup부터).
#
# 4sym 이후: 2026-07-21 스코프 재조정으로 옥타브(4ott)+셋잇단음표(4tup) 추가(tie/붙임줄은 미구현이라
# 제외). 둘 다 좁은 축이라 1500장/15epoch로 축소. 옥타브 span은 generate_scores.py에 이미 1~2마디로
# 제한돼 있고, 셋잇단음표도 이미 단음 8분음표 3개(임시표 가능)로 구현돼 있어 코드 추가 수정 불필요.
#
# 사전 조건(직접 확인 후 실행할 것):
#   - cv2/music21/xvfb/libEGL 등 설치 확인 (pod 재시작 시 초기화됨)
#   - round3train/{dataset.py,train.py,generate_scores.py,error_breakdown.py,gen_render_local.sh} 최신본 배포됨
#     (generate_scores.py는 KEY_SIGS 13종, TIME_SIGS에 6/8, --dynamics-subset 플래그가 반영된 버전이어야 함)
#   - round1_curriculum_p2s4gm(4gm) 체크포인트 + round1_stage4gm_new(4gm 학습 데이터, 1800장 보존분) 존재

set -uo pipefail
LOG=/workspace/curriculum_4t_4sym.log
STATUS=/workspace/curriculum_4t_4sym_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) 박자/조표/다이나믹/기호 커리큘럼 시작 (4t~4sym) ==="
: > "$STATUS"

STAGE_NAMES=(4t 4k 4dyn 4sym 4ott 4tup)
COUNTS=(2500 2500 2500 2500 1500 1500)
EPOCHS=(25 25 25 25 15 15)
FREEZE_EPOCHS=(5 5 5 5 3 3)
START_IDX_BASE=(3000001 3050001 3100001 3150001 3200001 3250001)
SEEDS=(300001 300501 301001 301501 302001 302501)
GATE_THRESHOLDS=(95 95 95 95 95 95)
KEEP_REPLAY=500   # 각 단계 통과 후, 그 단계가 만든 신규 데이터 중 다음 단계 replay용으로 남길 장수

# scheduled sampling(word-dropout 방식 teacher-forcing 완화) -- exposure bias 완화용.
# 4t/4k는 이미 완료돼서 값을 바꿔도 영향 없음(재실행 시에만 의미). 4dyn부터 켬:
# freeze 종료 후 남은 epoch의 대부분에 걸쳐 tf 1.0->0.5로 선형 감소, 나머지는 0.5 유지.
MIN_TF_RATIOS=(1.0 1.0 0.5 0.5 0.5 0.5)
SS_EPOCHS_ARR=(0 0 15 15 10 10)

# 단계별 generate_scores.py 인자 -- 공통 축(마디 1~4, density-break, 화음)은 매 단계 동일하게
# 유지하고, 그 단계에서 새로 도입하는 축만 0이 아닌 값으로 바꾼다.
# 반복기호(barline-start-repeat/end-repeat)는 4t 잔여 오류의 지배적 원인으로 확인되어(오류의
# ~70%) 프로젝트 범위에서 제외 -- repeat-prob 0으로 더 이상 생성하지 않는다.
# chord-prob은 무작위 생성 특유의 과밀 완화를 위해 generate_scores.py 기본값(0.08, 기존 0.15)과
# 맞춰 명시적으로 낮춰준다 -- 이 값을 안 맞추면 CLI 인자가 모듈 기본값을 덮어써서 무의미해짐.
COMMON_ARGS=(--min-measures 1 --max-measures 4 --density-break --chord-prob 0.08 --repeat-prob 0
             --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0 --ottava-prob 0)

build_gen_args() {
  local name="$1"
  case "$name" in
    4t)
      # 박자만 새로 도입(6/8) -- 조표는 C장조로 고정해 축 격리, 다이나믹/기호는 아직 0
      echo "${COMMON_ARGS[@]} --force-c-major --dynamic-prob 0 --hairpin-prob 0 --fermata-prob 0"
      ;;
    4k)
      # 조표 다양화(13종) -- force-c-major 해제, 박자는 4t에서 넓힌 4종(4/4,3/4,2/4,6/8) 그대로 유지
      echo "${COMMON_ARGS[@]} --diatonic-bias 0.75 --dynamic-prob 0 --hairpin-prob 0 --fermata-prob 0"
      ;;
    4dyn)
      # 다이나믹 최소셋(p,f,pp,ff,mp,mf, fp 제외) 도입
      echo "${COMMON_ARGS[@]} --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf --hairpin-prob 0 --fermata-prob 0"
      ;;
    4sym)
      # 페르마타+헤어핀(크레센도/디크레센도) 도입, 다이나믹은 유지(replay 아닌 신규 데이터에도 계속 노출)
      echo "${COMMON_ARGS[@]} --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf --hairpin-prob 0.20 --fermata-prob 0.04"
      ;;
    4ott)
      # 옥타브(8va/8vb) 도입, 이전 축(조표/다이나믹/헤어핀/페르마타) 전부 유지
      echo "${COMMON_ARGS[@]} --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf --hairpin-prob 0.20 --fermata-prob 0.04 --ottava-prob 0.35"
      ;;
    4tup)
      # 셋잇단음표 도입, 이전 축(옥타브 포함) 전부 유지
      echo "${COMMON_ARGS[@]} --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf --hairpin-prob 0.20 --fermata-prob 0.04 --ottava-prob 0.35 --tuplet-prob 0.35"
      ;;
  esac
}

RESUME_FROM=${RESUME_FROM:-0}

if [ "$RESUME_FROM" -eq 0 ]; then
  PREV_NAME=4gm
  PREV_OUT=/workspace/models/round1_curriculum_p2s4gm
  PREV_DATA=/workspace/data/round1_stage4gm_new
else
  PN=${STAGE_NAMES[$((RESUME_FROM-1))]}
  PREV_NAME="$PN"
  PREV_OUT="/workspace/models/round1_curriculum_p2s${PN}"
  PREV_DATA="/workspace/data/round1_stage${PN}_new"
  echo "=== $(date) RESUME_FROM=${RESUME_FROM} (${STAGE_NAMES[$RESUME_FROM]}부터 재개, 직전=${PREV_NAME}) ==="
fi

for i in "${!STAGE_NAMES[@]}"; do
  if [ "$i" -lt "$RESUME_FROM" ]; then
    continue
  fi
  NAME=${STAGE_NAMES[$i]}
  COUNT=${COUNTS[$i]}
  EP=${EPOCHS[$i]}
  FRZ=${FREEZE_EPOCHS[$i]}
  SIDX=${START_IDX_BASE[$i]}
  SEED=${SEEDS[$i]}
  GATE=${GATE_THRESHOLDS[$i]}
  MIN_TF=${MIN_TF_RATIOS[$i]}
  SS_EP=${SS_EPOCHS_ARR[$i]}

  NEW_DIR=/workspace/data/round1_stage${NAME}_new
  TRAIN_OUT=/workspace/models/round1_curriculum_p2s${NAME}

  echo ""
  echo "=== $(date) [$NAME] 시작 (신규 ${COUNT}장, epoch ${EP}, resume=${PREV_NAME}, replay=${PREV_DATA}) ==="

  echo "[$NAME] quota 확인 중..."
  if ! dd if=/dev/zero of=/workspace/_qcheck bs=1M count=100 2>/dev/null; then
    echo "[$NAME] 경고: quota 초과 위험 -- 파이프라인 중단"
    rm -f /workspace/_qcheck
    echo "PIPELINE_STOPPED_QUOTA:$NAME" >> "$STATUS"
    exit 1
  fi
  rm -f /workspace/_qcheck

  # --- 데이터 생성+렌더링+검증 (로컬 파이프라인, 이미 존재하면 재사용) ---
  # MuseScore가 드물게 개별 파일 렌더링에 실패하는 건 정상 노이즈라(gen_render_local.sh가
  # 고아 파일은 알아서 제외하고 복사) COUNT의 97% 이상이면 "이미 존재"로 간주한다.
  EXISTING_N=$(find "$NEW_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  MIN_OK_N=$(( COUNT * 97 / 100 ))
  if [ "$EXISTING_N" -ge "$MIN_OK_N" ]; then
    echo "[$NAME] 데이터 이미 존재(${EXISTING_N}/${COUNT}장) -- 재생성 스킵"
  else
    rm -rf "$NEW_DIR"
    read -ra GEN_ARGS <<< "$(build_gen_args "$NAME")"
    GEN_ARGS=(--start-idx "$SIDX" --seed "$SEED" "${GEN_ARGS[@]}")
    echo "[$NAME] 데이터 생성: ${GEN_ARGS[*]}"
    bash /workspace/round3train/gen_render_local.sh "$NEW_DIR" "$COUNT" "${GEN_ARGS[@]}"
    if [ $? -ne 0 ]; then
      echo "[$NAME] 데이터 생성/검증 실패 -- 파이프라인 중단"
      echo "PIPELINE_STOPPED_GENFAIL:$NAME" >> "$STATUS"
      exit 1
    fi
  fi

  # --- replay 수: PREV_DATA에 실제로 남아있는 장수 그대로 사용(1단계는 4gm의 1800장 보존분,
  #     2단계부터는 직전 단계가 남긴 500장) ---
  REPLAY_COUNT=$(find "$PREV_DATA" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  echo "[$NAME] replay: ${REPLAY_COUNT}장 (${PREV_DATA}에서 직접, 복사 없음)"

  # --- 학습 ---
  mkdir -p "$TRAIN_OUT"
  cd /workspace/round3train
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 -u train.py --phase 2 \
    --data_dir "$NEW_DIR" --out_dir "$TRAIN_OUT" \
    --replay_dir "$PREV_DATA" --replay_count "$REPLAY_COUNT" \
    --resume "$PREV_OUT/seq2seq_best.pt" \
    --batch 24 --epochs "$EP" --workers 16 \
    --freeze_epochs "$FRZ" --tf_ratio 1.0 --min_tf_ratio "$MIN_TF" --ss_epochs "$SS_EP"
  TRAIN_RC=$?
  if [ $TRAIN_RC -ne 0 ]; then
    echo "[$NAME] 학습 프로세스 비정상 종료(exit=$TRAIN_RC) -- 파이프라인 중단"
    echo "PIPELINE_STOPPED_TRAINFAIL:$NAME" >> "$STATUS"
    exit 1
  fi

  # --- 정확도 게이트 ---
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
  echo "[$NAME] 학습 완료 -- best val_acc = ${BEST_ACC}%"

  PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
  if [ "$PASS" != "1" ]; then
    echo "[$NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%) -- 파이프라인 중단, 오류 분석 실행"
    echo "PIPELINE_STOPPED_LOW_ACC:$NAME:$BEST_ACC" >> "$STATUS"
    python3 /workspace/round3train/error_breakdown.py \
      --seq2seq "$TRAIN_OUT/seq2seq_best.pt" \
      --tokenizer /workspace/round3train/tokenizer258.json \
      --data_dir "$NEW_DIR" \
      > "$TRAIN_OUT/error_breakdown.log" 2>&1
    echo "[$NAME] 오류 분석 결과: $TRAIN_OUT/error_breakdown.log 확인 (다음 단계 진행 안 함, 데이터/데이터 삭제 안 함)"
    exit 2
  fi

  echo "[$NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
  echo "STAGE_PASSED:$NAME:$BEST_ACC" >> "$STATUS"

  # --- 디스크 정리: 이 단계가 만든 ${COUNT}장 중 ${KEEP_REPLAY}장만 남기고 나머지 삭제
  #     (다음 단계 replay 소스가 됨) ---
  echo "[$NAME] 디스크 정리: $NEW_DIR 를 ${KEEP_REPLAY}장으로 축소"
  python3 -c "
import glob, os, random
random.seed(42)
stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob('$NEW_DIR/*.png'))
keep = set(random.sample(stems, min($KEEP_REPLAY, len(stems))))
removed = 0
for stem in stems:
    if stem in keep:
        continue
    for ext in ('.png', '.json', '_pre.npy', '_staffs.json'):
        p = os.path.join('$NEW_DIR', stem + ext)
        if os.path.exists(p):
            os.remove(p)
            removed += 1
print(f'[cleanup] {len(keep)}장 보존, {removed}개 파일 삭제')
"

  PREV_NAME="$NAME"
  PREV_OUT="$TRAIN_OUT"
  PREV_DATA="$NEW_DIR"
done

echo ""
echo "=== $(date) 4tup까지 전체 통과 -- 박자/조표/다이나믹/기호/옥타브/셋잇단음표 커리큘럼 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
