#!/bin/bash
# round3train/local_generate_all.sh
#
# 노트북(GPU 불필요, MuseScore CLI만 있으면 됨)에서 Step 1/1.5/2 학습 데이터를 전부
# 생성한다. 각 스테이지의 파라미터는 해당 curriculum_*.sh의 COMMON_ARGS/스테이지별
# 옵션·START_IDX·SEED를 그대로 복사한 것 -- 나중에 포드로 옮길 때 그 스크립트들이
# "의도한" 인덱스/시드와 그대로 맞아떨어지게 하기 위함(포드 쪽 기존 풀과의 충돌 여부는
# 각 curriculum_*.sh 상단 TODO대로 업로드 시점에 별도 확인 필요 -- 이 로컬 생성 자체는
# 무관).
#
# 출력: round3train/data/local_pools/{6reg1,tie1,7den1}/ (.gitignore의 round3train/data/
# 규칙으로 이미 커밋 제외됨)
#
# 병렬화 안 함(순차 실행) -- 여러 MuseScore4.exe GUI 인스턴스를 동시에 띄우는 게
# Windows에서 안정적인지 검증이 안 돼서 안전하게 직렬로 감.
#
# 2026-07-30 변경 두 가지:
#   1) 스텝당 총 2500장으로 축소(기존 4000장) -- 대보표/단일오선 비율은 그대로 유지.
#   2) 대보표 생성에서 --density-break 제거 -- 실제 카메라 캡처가 항상 시스템 1개만
#      담는데(guided_camera_screen.dart) density-break는 내용이 조밀하면 의도적으로
#      시스템을 2개 이상 만듦. 게다가 --density-break 없이도 내용이 조밀하면 MuseScore가
#      기본 페이지 높이 기준으로 "조용히" 2번째 시스템으로 밀려나고 system_breaks는 여전히
#      []로 기록되는(라벨-이미지가 실제로 어긋나는) 사고를 로컬 실측으로 발견해서,
#      generate_scores.py에 대보표 전용 wide_page_grand.mss(높이 확장) + 렌더 결과가
#      빈 페이지면 자동 스킵하는 안전장치를 추가함. 이제 대보표도 항상 시스템 1개만
#      생성됨(단일 오선은 기존부터 이미 wide_page.mss로 보장돼 있었음).
#
# 2026-07-30 추가 변경:
#   3) gen() 함수가 마지막 파일(num{start+count-1}.png) 존재 여부로 완료된 하위 호출을
#      건너뜀 -- 중간에 멈춰도 이어서 실행 가능(6reg1 대보표 도중 사용자가 폴더 구조를
#      물어봐서 확인차 중단했다가 재개한 게 계기).
#   4) 각 스텝의 핵심 축(cross-register-prob, tie-prob)을 다음 스텝 데이터에도 낮은
#      확률로 계속 섞음(사용자 요청) -- 예전엔 체크포인트 연속성에만 의존하고 데이터는
#      스텝마다 그 스텝 축 하나만 켜져 있었음. tie1은 cross-register-prob 0.15(6reg1의
#      0.35보다 낮춤) 추가, 7den1은 tie-prob 0.15(tie1의 0.30보다 낮춤) +
#      cross-register-prob 0.15 둘 다 추가 -- 이번 단계 주 목표를 가리지 않는 선에서.

set -uo pipefail

MUSESCORE="/c/Program Files/MuseScore 4/bin/MuseScore4.exe"
ROOT="round3train/data/local_pools"
LOG="round3train/data/local_pools/generate_all.log"
mkdir -p "$ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) local_generate_all 시작 ==="

gen() {
  local out="$1" count="$2" start="$3" seed="$4"; shift 4
  local last=$((start + count - 1))
  if [ -f "$out/num${last}.png" ]; then
    echo ""
    echo "--- $out (start-idx=$start count=$count) 이미 완료됨(num${last}.png 존재) -- 스킵 ---"
    return 0
  fi
  echo ""
  echo "--- $(date) $out (count=$count start-idx=$start seed=$seed) ---"
  python round3train/generate_scores.py \
    --count "$count" --output "$out" --musescore "$MUSESCORE" \
    --start-idx "$start" --seed "$seed" "$@"
  local rc=$?
  local n=$(find "$out" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  echo "--- $out 완료(exit=$rc): PNG ${n}장 ---"
  return $rc
}

# ── Step 1 (6reg1): 교차음역 — curriculum_6register.sh와 동일 파라미터 ───────
# --preferred-register-prob 0.7: range.mscz 선호 구간(치 D3~A3/A5~A6, 베이스 C2~E2/F4~B4)
# 70% 확률 노출. --hairpin-prob/--ottava-prob 0: 크레센도/디크레센도·옥타브 기호 완전 제거
# (2026-07-30, 대보표/단일오선 공통 -- 예전엔 대보표 호출에만 붙어있어서 단일오선에 샜음).
S1_COMMON=(--min-measures 1 --max-measures 4 --chord-prob 0.08 --repeat-prob 0
           --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
           --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
           --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0
           --preferred-register-prob 0.7)
gen "$ROOT/6reg1" 1500 4500001 318001 \
  "${S1_COMMON[@]}" --cross-register-prob 0.35 \
  || { echo "[6reg1] 대보표 생성 실패 -- 중단"; exit 1; }
gen "$ROOT/6reg1" 1000 4700001 318002 \
  "${S1_COMMON[@]}" --single-staff \
  || { echo "[6reg1] 단일오선 생성 실패 -- 중단"; exit 1; }

# ── Step 1.5 (tie1): 붙임줄 — curriculum_6b_tie.sh와 동일 파라미터 ──────────
S15_COMMON=(--min-measures 2 --max-measures 4 --chord-prob 0.08 --repeat-prob 0
            --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
            --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
            --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.30)
gen "$ROOT/tie1" 1250 5300001 331001 \
  "${S15_COMMON[@]}" --cross-register-prob 0.15 \
  || { echo "[tie1] 대보표 생성 실패 -- 중단"; exit 1; }
gen "$ROOT/tie1" 1250 5500001 331002 \
  "${S15_COMMON[@]}" --single-staff \
  || { echo "[tie1] 단일오선 생성 실패 -- 중단"; exit 1; }

# ── Step 2 (7den1): 리듬 밀도 — curriculum_7density.sh와 동일 파라미터 ──────
S2_COMMON=(--min-measures 1 --max-measures 4 --chord-prob 0.08 --repeat-prob 0
           --artic-prob 0 --ornament-prob 0 --slur-prob 0 --tuplet-prob 0.35
           --diatonic-bias 0.75 --dynamic-prob 0.35 --dynamics-subset p,f,pp,ff,mp,mf
           --fermata-prob 0.04 --hairpin-prob 0 --ottava-prob 0 --tie-prob 0.15
           --dotted8-bias 10.0 --short-note-bias 2.0)
gen "$ROOT/7den1" 1250 5700001 338001 \
  "${S2_COMMON[@]}" --cross-register-prob 0.15 \
  || { echo "[7den1] 대보표 생성 실패 -- 중단"; exit 1; }
gen "$ROOT/7den1" 1250 5900001 338002 \
  "${S2_COMMON[@]}" --single-staff \
  || { echo "[7den1] 단일오선 생성 실패 -- 중단"; exit 1; }

echo ""
echo "=== $(date) 전체 완료 ==="
for d in 6reg1 tie1 7den1; do
  n=$(find "$ROOT/$d" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  sz=$(du -sh "$ROOT/$d" 2>/dev/null | cut -f1)
  echo "  $d: ${n}장, ${sz}"
done
echo "ALL_DONE" >> "$LOG"
