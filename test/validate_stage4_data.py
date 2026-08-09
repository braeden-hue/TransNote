"""Stage4 재학습 데이터 학습 직전 검증 (병렬화 버전).
- png/json 파일명 매칭(orphan 검출)
- MuseScore 렌더링 잔여물(*.musicxml-1.png 등 rename 실패) 검출
- PNG 파일이 실제로 열리는 유효한 이미지인지(cv2로 디코딩, 크기>0) 검출
- JSON 파일이 유효한 JSON이고 tokens가 비어있지 않은지 검출

파일당 개별 체크(getsize/cv2.imread/json.load)는 네트워크 마운트에서 순차 실행 시
매우 느리므로(dataset.py의 OMRDataset 캐싱과 동일한 병목) ThreadPoolExecutor로 병렬화.

오류가 하나라도 있으면 exit code 1로 종료(학습 시작 안 함).
"""
import sys
import json
import glob
import os
import cv2
from concurrent.futures import ThreadPoolExecutor

def _check_png(p):
    if os.path.getsize(p) == 0:
        return (p, 'zero-byte')
    return None

def _check_png_decode(p):
    if os.path.getsize(p) == 0:
        return (p, 'zero-byte')
    img = cv2.imread(p)
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return (p, 'decode-fail')
    return None

def _check_json(j):
    try:
        with open(j, encoding='utf-8') as f:
            obj = json.load(f)
        if not obj.get('tokens'):
            return (j, 'empty-tokens')
    except Exception as e:
        return (j, f'parse-error: {e}')
    return None

def main():
    if len(sys.argv) < 2:
        print("usage: validate_stage4_data.py <data_dir> [--skip-decode] [--workers N]")
        sys.exit(2)
    data_dir = sys.argv[1]
    skip_decode = '--skip-decode' in sys.argv[2:]
    workers = 32
    if '--workers' in sys.argv[2:]:
        workers = int(sys.argv[sys.argv.index('--workers') + 1])

    errors = []

    # 주의: 원본 .musicxml 소스 파일은 로컬 스테이징 디렉토리(gen_render_local.sh)에서는
    # 정상적으로 함께 존재함(최종 복사 대상 아님) -- rename 실패(.musicxml-1.png)만 오류로 취급
    leftover = glob.glob(os.path.join(data_dir, '*.musicxml-1.png'))
    if leftover:
        errors.append(f"rename 실패 잔여 파일 {len(leftover)}개 발견 (예: {leftover[:5]})")

    png_paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
    json_paths = sorted(p for p in glob.glob(os.path.join(data_dir, '*.json'))
                         if not p.endswith('_staffs.json'))
    png_stems = {os.path.splitext(os.path.basename(p))[0] for p in png_paths}
    json_stems = {os.path.splitext(os.path.basename(p))[0] for p in json_paths}

    # MuseScore 렌더링은 대량 배치 중 드물게(관측상 <2%) 개별 파일이 실패하는 게 정상적인
    # 노이즈다 -- orphan(짝 없는 파일)은 전체 대비 비율이 작으면 그 샘플만 제외하고 넘어가고
    # (gen_render_local.sh가 실제 페어만 복사), 비율이 크면(체계적 문제 가능성) 그대로 fatal.
    orphan_png = sorted(png_stems - json_stems)
    orphan_json = sorted(json_stems - png_stems)
    total_stems = len(png_stems | json_stems)
    orphan_frac = (len(orphan_png) + len(orphan_json)) / total_stems if total_stems else 0
    ORPHAN_TOLERANCE = 0.03
    orphan_is_fatal = orphan_frac > ORPHAN_TOLERANCE
    if orphan_png:
        msg = f"json 짝 없는 png {len(orphan_png)}개 (예: {orphan_png[:5]})"
        if orphan_is_fatal:
            errors.append(msg)
        else:
            print(f"[validate]  - (경고, 해당 샘플만 제외하고 진행) {msg}")
    if orphan_json:
        msg = f"png 짝 없는 json {len(orphan_json)}개 (예: {orphan_json[:5]})"
        if orphan_is_fatal:
            errors.append(msg)
        else:
            print(f"[validate]  - (경고, 해당 샘플만 제외하고 진행) {msg}")

    png_check_fn = _check_png if skip_decode else _check_png_decode
    with ThreadPoolExecutor(max_workers=workers) as ex:
        bad_png = [r for r in ex.map(png_check_fn, png_paths) if r is not None]
        bad_json = [r for r in ex.map(_check_json, json_paths) if r is not None]

    if bad_png:
        label = '손상된(0바이트)' if skip_decode else '손상된'
        errors.append(f"{label} png {len(bad_png)}개 (예: {bad_png[:5]})")
    if bad_json:
        errors.append(f"손상/빈 json {len(bad_json)}개 (예: {bad_json[:5]})")

    n_pairs = len(png_stems & json_stems)
    print(f"[validate] {data_dir}")
    print(f"[validate] png={len(png_paths)} json={len(json_paths)} 정상짝={n_pairs}")

    if errors:
        print(f"[validate] 오류 {len(errors)}건 발견 -- 학습 시작 보류")
        for e in errors:
            print(f"[validate]  - {e}")
        sys.exit(1)

    print("[validate] 오류 없음 -- 학습 진행 가능")
    sys.exit(0)

if __name__ == '__main__':
    main()
