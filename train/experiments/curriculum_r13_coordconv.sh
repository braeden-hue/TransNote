#!/bin/bash
# round3train/curriculum_r13_coordconv.sh
#
# CoordConv 재시도(2026-08-03) -- 3도(선/칸) 음이름 오독 완화 시도. `pool_h>1`(백본 뒷단
# 2단계 제거 + 구조 변경, r10c_poolh_test)은 목표 지표(3도 오독 비율)가 전혀 안 움직이고
# 전체 정확도만 대폭 후퇴(-12.7pp)해 실패. CoordConv는 2026-07-31에 이미 한 번 시도돼
# 89곡 3도 오독 273건->235건(14% 감소, model.py 주석 참고)으로 부분 효과가 있었지만
# 그때는 combined_coordconv(초기 실험 체크포인트, 지금 계보와 무관) 위에서만 테스트됐고
# 이후 r1_v2_foundation 재시작 때 in_ch=1로 되돌아가면서 유실됨(load_ckpt_partial_vocab이
# shape mismatch난 첫 conv 레이어만 재초기화, 나머지는 유지) -- 지금까지 검증된 가장
# 성숙한 체크포인트(r12_all120_realphotos, 실사 120곡 학습)에서 다시 테스트하는 게 처음.
#
# pool_h 변경(백본 후반부, 시퀀스 형성에 가까운 여러 층 제거)보다 훨씬 작은 변경(첫
# conv 레이어 입력 채널 1->2만 바뀜, 나머지 전부 그대로 이어받음)이라 재적응 부담이
# 적을 것으로 예상 -- dataset.py의 make_model_input(in_ch=2)이 이미 구현돼 있어 좌표
# 채널 자동 생성, inference.py도 이미 범용 지원(코드 수정 불필요).
#
# replay 풀은 r12에서 만든 것을 그대로 재사용(r12_replay_merged, 8486장, 이미 실사
# 120곡의 5배 이상 확보돼 있음) -- 재복사 불필요.

set -uo pipefail
LOG=/workspace/curriculum_r13_coordconv.log
STATUS=/workspace/curriculum_r13_coordconv_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 13(CoordConv 재시도) 시작 ==="
: > "$STATUS"

REAL_POOL_DIR=/workspace/round3train/data/local_pools/r12_all120_realphotos   # r12와 동일 120곡/1638장
REPLAY_DIR=/workspace/data/r12_replay_merged   # r12에서 이미 만든 걸 재사용(8486장)
REPLAY_COUNT=8486

RESUME_CKPT=/workspace/models/r12_all120_realphotos/seq2seq_best.pt   # 현재 최선에서 재개
IN_CH=2   # CoordConv -- 유일한 구조 변경
EXTRA_HEIGHT_STAGES=4   # r12와 동일(pool_h 실험과 무관하게 되돌려둠)
POOL_H=1
STAGE_NAME=r13_coordconv
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=15
FREEZE_EPOCHS=0
GATE=70

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$REAL_POOL_DIR" ] || [ ! -d "$REPLAY_DIR" ]; then
  echo "[$STAGE_NAME] 데이터/replay 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_DATA" >> "$STATUS"
  exit 1
fi

REAL_N=$(find "$REAL_POOL_DIR" -maxdepth 1 \( -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, in_ch=${IN_CH}, 메인=실사 ${REAL_N}장, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$REAL_POOL_DIR" --out_dir "$TRAIN_OUT" \
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- 신규 검증곡으로 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 13(CoordConv 재시도) 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
