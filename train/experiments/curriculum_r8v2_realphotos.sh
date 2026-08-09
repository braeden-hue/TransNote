#!/bin/bash
# round3train/curriculum_r8v2_realphotos.sh
#
# Round8 재실행(2026-08-02) -- 클래식 실사 사진 100곡(86곡 기존 + 신규 14곡, 1609장)을
# 단독으로 투입. newage(20곡)는 검증 전용으로 완전히 제외.
#
# 시도 이력:
#   v1(RESUME=r7b, freeze=0, epoch20): 77.8% (매 에폭 41~78%로 심하게 출렁임)
#   v2a(RESUME=r9_chord_accidental, freeze=2->3, epoch20->30 조정): epoch1부터
#     완전 붕괴(freeze=3: Acc 0%/15.2%/17.0%, unfreeze 후 4epoch=20.3%로도 회복
#     안 됨). freeze=0으로 되돌려도 epoch1=0%(TER 124%), epoch2=0%(TER 133%,
#     오히려 악화) -- r9_chord_accidental 체크포인트 자체가 원인으로 판단해 중단.
#   v2b(RESUME=r7b_l4_synth_only, freeze=0, epoch30, replay에 r9_chord_accidental_synth
#     800장 섞음): v1보다 진폭이 더 커짐(최대 ~49%p vs v1의 ~20%p), best 72.8%에서
#     못 넘고 22에폭에서 중단. r9 체크포인트를 newage 20곡 깨끗한 렌더링(실사 아님)
#     으로 별도 실측하니 96.2%로 콘텐츠 인식 자체는 전혀 문제 없었음 -- 즉 화음
#     강화 "콘텐츠"는 멀쩡한데, 화음 강화 "합성 replay 데이터"가 실사 도입과 같은
#     배치에 섞이면 난이도 분산이 커져 불안정해지는 것으로 판단.
#   v2c(현재): v2b에서 replay만 v1 방식(r7_l4_major_synth)으로 되돌림. 메인
#     데이터(100곡)와 epoch(30), 나머지 설정은 v2b 그대로 유지 -- "데이터 확장
#     자체"와 "r9 replay 혼합" 중 어느 쪽이 진폭 확대의 원인인지 분리해서 확인.
#
# 기존 curriculum_r8_realphotos.sh(77.8%, v1) 대비 변경점:
#   - 메인 데이터: 86곡/1550장 -> 100곡/1609장(신규 14곡 추가)
#   - epoch: 20 -> 30(OneCycleLR을 더 완만하게 해서 후반부 출렁임 완화 시도)
#   - dataset.py 오선검출 버그 수정(합성 이미지 페이지-쿼드 오검출) 및 inference.py
#     barline-final 강제종료 안전장치 반영 -- 둘 다 실사 경로엔 영향 없음(측정 확인됨)
#
# 사전 조건: curriculum_r7_l4_synth_only.sh가 STAGE_PASSED로 끝나있어야 함.

set -uo pipefail
LOG=/workspace/curriculum_r8v2.log
STATUS=/workspace/curriculum_r8v2_status.txt
exec >> "$LOG" 2>&1

echo "=== $(date) Round 8 재실행(v2c, r7b 기반+replay v1 복원, 실사 100곡) 시작 ==="
: > "$STATUS"

if ! grep -q "^STAGE_PASSED:r7b_l4_synth_only" /workspace/curriculum_r7b_status.txt 2>/dev/null; then
  echo "[r8v2] Round7(재정의)가 STAGE_PASSED 상태가 아님 -- 중단"
  echo "PIPELINE_STOPPED_R7B_NOT_PASSED" >> "$STATUS"
  exit 1
fi

REAL_POOL_DIR=/workspace/data/classical_realphotos   # 100곡, 1609장
REPLAY_POOL_OLD=/workspace/data/replay_pool_v2
REPLAY_POOL_R7=/workspace/data/r7_l4_major_synth   # v1과 동일(화음 강화 아닌 일반 L4 노이즈 데이터)
REPLAY_MERGED=/workspace/data/r8v2_replay_merged
REPLAY_OLD_N=1200
REPLAY_R7_N=800
REPLAY_COUNT=2000

RESUME_CKPT=/workspace/models/r7b_l4_synth_only/seq2seq_best.pt
IN_CH=${IN_CH:-1}
EXTRA_HEIGHT_STAGES=${EXTRA_HEIGHT_STAGES:-4}
POOL_H=${POOL_H:-1}
STAGE_NAME=r8v2_realphotos
TRAIN_OUT=/workspace/models/$STAGE_NAME
EPOCHS=30
FREEZE_EPOCHS=0
GATE=75
LR=6e-5   # 3e-5(v2d)는 진폭은 크게 줄었으나(epoch4~6: 25.4/24.1/27.6%) 너무 낮은
          # 수준에서 정체 -- 기본값(1e-4)과 3e-5의 중간으로 재시도(2026-08-02).
          # unfreeze 직후 max_lr=LR/3.0로 워밍업 없이(pct_start=0.0) 곧바로 최고
          # 학습률부터 시작하는 구조.

if [ ! -f "$RESUME_CKPT" ]; then
  echo "[$STAGE_NAME] RESUME_CKPT 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_RESUME_CKPT" >> "$STATUS"
  exit 1
fi
if [ ! -d "$REAL_POOL_DIR" ]; then
  echo "[$STAGE_NAME] 클래식 실사 풀 없음 -- 중단"
  echo "PIPELINE_STOPPED_NO_REALPHOTOS" >> "$STATUS"
  exit 1
fi

echo "[$STAGE_NAME] Replay 풀 구성 중 (이전 라운드 ${REPLAY_OLD_N} + r7_l4_major_synth ${REPLAY_R7_N}, 실제 파일 복사)..."
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
pick_and_copy "$REPLAY_POOL_R7" "$REPLAY_R7_N" r7
REPLAY_N=$(find "$REPLAY_MERGED" -maxdepth 1 -name '*.png' | wc -l)
echo "[$STAGE_NAME] Replay 풀 완성: ${REPLAY_N}장"

REAL_N=$(find "$REAL_POOL_DIR" -maxdepth 1 \( -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
echo ""
echo "=== $(date) [$STAGE_NAME] 학습 시작 (resume=$RESUME_CKPT, epoch ${EPOCHS}, freeze=${FREEZE_EPOCHS}, 메인=실사 ${REAL_N}장, replay=${REPLAY_COUNT}) ==="
mkdir -p "$TRAIN_OUT"
cd /workspace/round3train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train.py --phase 2 \
  --data_dir "$REAL_POOL_DIR" --out_dir "$TRAIN_OUT" \
  --tokenizer /workspace/round3train/tokenizer258.json \
  --resume "$RESUME_CKPT" --in_ch "$IN_CH" \
  --extra_height_stages "$EXTRA_HEIGHT_STAGES" --pool_h "$POOL_H" \
  --replay_dir "$REPLAY_MERGED" --replay_count "$REPLAY_COUNT" \
  --page_level_noise --noise_level 3 --noise_level_max 4 --p_level_max 0.5 \
  --batch 24 --epochs "$EPOCHS" --workers 16 \
  --freeze_epochs "$FREEZE_EPOCHS" --lr "$LR"
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
echo "[$STAGE_NAME] 주의: TF 기준이라 신뢰 불가 -- newage 20곡(학습 미포함, 검증 전용) 실사 오류분석으로 재검증 필요"

PASS=$(python3 -c "print(1 if float('$BEST_ACC') >= float('$GATE') else 0)")
if [ "$PASS" != "1" ]; then
  echo "[$STAGE_NAME] 정확도 ${GATE}% 미만(${BEST_ACC}%)"
  echo "PIPELINE_STOPPED_LOW_ACC:$BEST_ACC" >> "$STATUS"
  exit 2
fi

echo "[$STAGE_NAME] 통과(${BEST_ACC}% >= ${GATE}%)"
echo "STAGE_PASSED:$STAGE_NAME:$BEST_ACC" >> "$STATUS"
echo "=== $(date) Round 8 재실행 완료 ==="
echo "PIPELINE_COMPLETE" >> "$STATUS"
