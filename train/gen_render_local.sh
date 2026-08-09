#!/bin/bash
# 안전하고 빠른 데이터 생성+렌더링+검증 파이프라인.
# 사용법: gen_render_local.sh <최종_output_dir> <count> <extra generate_scores.py args...>
#
# 이번 세션에서 반복됐던 지연/사고 원인을 전부 해소:
#   1) 렌더링/rename을 네트워크 마운트(/workspace)가 아니라 로컬 디스크(/tmp)에서 수행
#      -> 파일당 네트워크 왕복 지연 제거
#   2) 검증도 로컬에서 병렬로 수행, 통과한 것만 네트워크로 일괄 복사
#      -> 손상 데이터가 네트워크 볼륨에 남는 사고 원천 차단
#   3) 매번 새 임시 디렉토리(mktemp)에서 시작
#      -> 이전 중단 시도의 orphan/부분 rename 잔여물이 절대 안 쌓임
#   4) 시작 전 잔여 렌더링 프로세스 강제 정리 + 복사 전 실제 쓰기 테스트로 quota 확인

set -uo pipefail

# 스크립트 자기 위치 기준 절대경로 -- /workspace 루트에 예전 사본이 남아있어도 그쪽을 절대
# 참조하지 않도록 함(2026-07-21: /workspace/validate_stage4_data.py가 낡은 사본이라 검증
# 관대화 수정이 적용 안 된 채로 4k가 두 번 막혔던 사고 재발 방지).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FINAL_DIR="$1"
COUNT="$2"
shift 2
RENDER_DPI="${RENDER_DPI:-150}"  # 환경변수로 오버라이드 가능 (예: RENDER_DPI=300 bash gen_render_local.sh ...)
                                  # dataset.py의 preprocess()가 TARGET_W=1920px로 강제 리사이즈하는데
                                  # 150dpi 렌더는 보통 1240px라 매번 업스케일(블러)됨 -- 300dpi면
                                  # ~2480px로 나와서 다운스케일(디테일 보존)로 바뀜

echo "[gen_render_local] 잔여 렌더링 프로세스 정리 중..."
pkill -9 -f musescore_wrapper.sh 2>/dev/null
pkill -9 -f 'squashfs-root/AppRun' 2>/dev/null
pkill -9 -f mscore4portable 2>/dev/null
pkill -9 -f '/usr/bin/Xvfb' 2>/dev/null
sleep 1

WORK=$(mktemp -d /tmp/genrender.XXXXXX)
echo "[gen_render_local] 로컬 작업 디렉토리: $WORK"

echo "[gen_render_local] XML+JSON 생성 중 (${COUNT}장)..."
cd /workspace/round3train
python3 generate_scores.py --count "$COUNT" --output "$WORK" --no-png "$@"
if [ $? -ne 0 ]; then
  echo "[gen_render_local] XML 생성 실패"
  exit 1
fi

echo "[gen_render_local] 렌더링 중 (로컬, 32-way, DPI=${RENDER_DPI}, 대략 1~2장/초 예상)..."
cd "$WORK"
ls *.musicxml | xargs -P 32 -I{} /workspace/musescore_wrapper.sh -o {}.png -r "$RENDER_DPI" {} >/dev/null 2>&1

echo "[gen_render_local] rename 중..."
for f in *.musicxml-1.png; do
  [ -e "$f" ] || continue
  stem="${f%.musicxml-1.png}"
  mv "$f" "${stem}.png"
done

echo "[gen_render_local] 로컬 검증 중 (병렬)..."
python3 "$SCRIPT_DIR/validate_stage4_data.py" "$WORK" --skip-decode
VALIDATE_RC=$?
n_png=$(find "$WORK" -maxdepth 1 -name '*.png' | wc -l)
n_json=$(find "$WORK" -maxdepth 1 -name '*.json' -not -name '*_staffs.json' | wc -l)
echo "[gen_render_local] 로컬 검증 결과: PNG=${n_png} JSON=${n_json} (exit=${VALIDATE_RC})"

if [ "$VALIDATE_RC" -ne 0 ]; then
  echo "[gen_render_local] 검증 실패 -- 네트워크로 복사하지 않음."
  echo "[gen_render_local] 문제 파일은 로컬 $WORK 에 그대로 남아있으니 직접 확인 후 재시도하세요."
  exit 1
fi

echo "[gen_render_local] quota 확인 중 (실제 쓰기 테스트)..."
if ! dd if=/dev/zero of=/workspace/_quota_check bs=1M count=50 2>/dev/null; then
  echo "[gen_render_local] 경고: quota 초과 위험 -- 복사 중단, 정리 필요"
  rm -f /workspace/_quota_check
  exit 1
fi
rm -f /workspace/_quota_check

echo "[gen_render_local] 네트워크로 일괄 복사 중 (짝 맞는 png+json만, 고아 파일 제외)..."
mkdir -p "$FINAL_DIR"
python3 -c "
import glob, os, shutil
work, final = '$WORK', '$FINAL_DIR'
pngs  = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(work, '*.png'))}
jsons = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(work, '*.json'))
         if not p.endswith('_staffs.json')}
pairs = pngs & jsons
for stem in pairs:
    shutil.copy2(os.path.join(work, stem + '.png'),  os.path.join(final, stem + '.png'))
    shutil.copy2(os.path.join(work, stem + '.json'), os.path.join(final, stem + '.json'))
skipped = (pngs | jsons) - pairs
print(f'[gen_render_local] 페어 {len(pairs)}개 복사, 고아(짝 없음) {len(skipped)}개 제외')
"
n_final=$(find "$FINAL_DIR" -maxdepth 1 -name '*.png' | wc -l)
echo "[gen_render_local] 복사 완료: $FINAL_DIR (PNG ${n_final}장)"

rm -rf "$WORK"
echo "[gen_render_local] === 완료 (로컬 임시 디렉토리 정리됨) ==="
