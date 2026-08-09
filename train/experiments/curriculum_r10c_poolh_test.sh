#!/bin/bash
# round3train/curriculum_r10c_poolh_test.sh
#
# "높이 부분 보존"(pool_h>1) 재검증 라운드(2026-08-03). curriculum_r10_register.sh와
# 완전히 동일한 데이터/재개지점/노이즈 설정(apples-to-apples)으로, 구조만
# extra_height_stages=4->2, pool_h=1->4로 바꿔서 학습한다.
#
# 배경: held-out 60곡 실측에서 음이름 오독의 84.6%가 3도(선/칸) 오독이고 이게 첫
# 음부터 이탈 캐스케이드를 유발한다는 게 확인됨(project_r10_register_heldout60_eval.md).
# model.py 주석에 이미 2026-07-31에 같은 가설(세로 해상도가 8단계 conv로 완전히
# 뭉개짐)을 CoordConv로 시도해 89곡 273건->235건으로만 줄어(미해결) 기록돼 있었고,
# pool_h>1 실험(heightpreserve_*)도 있었지만 --no_augment(노이즈 없는 깨끗한 렌더링)로만
# 테스트됐고 이 실사 held-out-60 벤치마크도 그때는 없었음 -- 제대로 된 비교가 이번이
# 처음. proj_warmup_epochs(2026-08-01, heightpreserve_warmup에서 검증된 패턴: 새로
# 초기화된 encoder.proj가 이미 학습된 디코더/backbone을 초기 노이즈로 흔들지 않게
# 먼저 워밍업)도 같이 적용.
#
# 사전 조건: curriculum_r10_register.sh와 동일(pod_gen_register_focus.sh GEN_COMPLETE).

set -uo pipefail
LOG=/workspace/curriculum_r10c_poolh_test.log
STATUS=/workspace/curriculum_r10c_poolh_test_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 10c(pool_h>1 재검증) 시작 ==="
: > "$STATUS"

if ! grep -q "^GEN_COMPLETE" /workspace/gen_register_focus_status.txt 2>/dev/null; then
  echo "[r10c_poolh_test] 레지스터 집중 데이터 생성이 GEN_COMPLETE 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_GEN_NOT_COMPLETE" >> "$STATUS"
  exit 1
fi

DATA_DIR=/workspace/round3train/data/register_focus_pool   # r10_register와 동일
REPLAY_DIR=/workspace/data/replay_pool_v2
REPLAY_COUNT=2000

RESUME_CKPT=/workspace/models/r8_2_diversity/seq2seq_best.pt   # r10_register와 동일 재개지점
IN_CH=1
EXTRA_HEIGHT_STAGES=2   # r10_register(기본 4)와 다른 유일한 구조 변경 -- 세로가 완전히
POOL_H=4                # 뭉개지기 전에 ~5~8줄 남긴 채 pool_h개 밴드로 구조적으로 보존
PROJ_WARMUP_EPOCHS=5    # heightpreserve_warmup(2026-08-01)에서 검증된 패턴
STAGE_NAME=r10c_poolh_test
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15               # r10_register와 동일
FREEZE_EPOCHS=0
GATE=80

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$DATA_DIR" ] || [ ! -d "$REPLAY_DIR" ]; then
  echo "[$STAGE_NAME] 데이터/replay 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_DATA" >> "$STATUS"
  exit 1
fi

DATA_N=$(find "$DATA_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, "
echo "    extra_height_stages=${EXTRA_HEIGHT_STAGES} pool_h=${POOL_H} proj_warmup=${PROJ_WARMUP_EPOCHS}, "
echo "    메인=합성 ${DATA_N}장, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --proj_warmup_epochs "$PROJ_WARMUP_EPOCHS" \
  --replay_dir "$REPLAY_DIR" --replay_count "$REPLAY_COUNT" \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- held-out 60곡으로 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 10c(pool_h>1 재검증) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
