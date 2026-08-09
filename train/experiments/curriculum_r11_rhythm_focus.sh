#!/bin/bash
# round3train/curriculum_r11_rhythm_focus.sh
#
# 밀집 리듬(8분/16분 런) 집중 라운드(2026-08-03). held-out 60곡 실측 최저 성적권인
# sonata_30_44(13.8%, r8_2_diversity 기준) 등이 오선 검출은 정상인데 극도로 밀집된
# 16분음표 패시지에서 붕괴하는 패턴 확인 -- [[project_dense_rhythm_failure]]와 동일
# 계열. pod_gen_r11_rhythm_focus.sh(--dotted8-bias 20.0 --eighth/sixteenth-run-prob
# 0.30/0.35)로 이 패턴을 직접 타깃한 순수 합성 데이터 1500장 생성.
#
# r8_2_diversity(현재 확인된 최선 체크포인트, [[project_r8_2_diversity_is_best_checkpoint]])
# 에서 재개. r10/r10b(순수 합성 데이터 축 확대 시도)가 실패했던 전례가 있으므로,
# "실사 도입"이 아니라 "순수 콘텐츠 보강"류 라운드(r5_rare_durations, r9_chord_accidental)의
# 검증된 패턴을 그대로 따름: replay는 비례 확대가 아니라 고정 2000장(replay_pool_v2),
# 노이즈 설정은 r8_2_diversity 학습 때와 동일하게 유지(실사 강건성 퇴행 방지).
#
# 사전 조건: pod_gen_r11_rhythm_focus.sh가 GEN_COMPLETE로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r11_rhythm_focus.log
STATUS=/workspace/curriculum_r11_rhythm_focus_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 11(밀집 리듬 집중) 시작 ==="
: > "$STATUS"

if ! grep -q "^GEN_COMPLETE" /workspace/gen_r11_rhythm_focus_status.txt 2>/dev/null; then
  echo "[r11_rhythm_focus] 밀집 리듬 집중 데이터 생성이 GEN_COMPLETE 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_GEN_NOT_COMPLETE" >> "$STATUS"
  exit 1
fi

DATA_DIR=/workspace/round3train/data/r11_rhythm_focus_pool   # 1500장, 순수 합성
REPLAY_DIR=/workspace/data/replay_pool_v2
REPLAY_COUNT=2000   # r5/r9류 "순수 콘텐츠 보강" 패턴 -- 실사 도입이 아니므로 고정치(비례 확대 아님)

RESUME_CKPT=/workspace/models/r8_2_diversity/seq2seq_best.pt
IN_CH=1
EXTRA_HEIGHT_STAGES=4   # r8_2_diversity와 동일(구조 변경 없음 -- pool_h 실험은 이미 음성 결과로 종결됨)
POOL_H=1
STAGE_NAME=r11_rhythm_focus
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15
FREEZE_EPOCHS=0
GATE=70

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
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, 메인=합성 ${DATA_N}장, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$DATA_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
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
echo "=== $(date) Round 11(밀집 리듬 집중) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
