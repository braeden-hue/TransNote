"""Step2(노이즈 증강) 도입 전 시각 확인용: 기존 Round3 학습에 실제로 쓰인 깨끗한 이미지
5장을 골라, page_noise_and_redetect()(실제 추론 경로와 동일한 노이즈->보정->재검출)로
기울기(회전/원근)+그림자(종이 굴곡 셰이딩)를 적용한 결과를 저장. 2026-07-31 사용자 요청 --
p_rotate/p_curl/p_curl_shade를 1.0으로 강제해서 매 장마다 두 효과가 반드시 보이게 함
(그 외 확률/강도는 NOISE_LEVELS[2] 그대로 -- 기존에 검증된 "실사용 기준 L2" 강도 유지).
"""
import copy
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as ds

SRC_DIR = Path(__file__).resolve().parent / 'data' / 'local_pools' / 'r3_density_register_clef'
OUT_DIR = Path(__file__).resolve().parent / 'data' / 'local_pools' / 'tilt_shadow_test'
N_SAMPLES = 5
LEVEL = 2

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    level_cfg = copy.deepcopy(ds.NOISE_LEVELS[LEVEL])
    level_cfg['p_rotate'] = 1.0
    level_cfg['p_curl'] = 1.0
    level_cfg['p_curl_shade'] = 1.0
    ds.NOISE_LEVELS[LEVEL] = level_cfg

    all_pngs = sorted(p for p in SRC_DIR.glob('*.png') if '_ms_tmp' not in p.stem)
    random.seed(42)
    picked = random.sample(all_pngs, N_SAMPLES)

    for png_path in picked:
        stem = png_path.stem
        gray0 = ds.load_preprocessed(str(png_path))
        staffs, gray = ds.best_effort_staff_detection(gray0)
        if not staffs:
            print(f"[{stem}] 오선 검출 실패 -- 스킵")
            continue

        n = len(staffs)
        staff_arg = staffs if (n >= 2 and n % 2 == 0) else staffs[0]

        # 원본(깨끗한) 캔버스도 같이 저장해서 비교 가능하게 함.
        if isinstance(staff_arg, list):
            clean_tile = ds.extract_system_canvas(gray, staff_arg[:2])
        else:
            clean_tile = ds.extract_staff_canvas(gray, staff_arg)
        cv2.imwrite(str(OUT_DIR / f"{stem}_clean.png"), clean_tile)

        noisy_tile = ds.page_noise_and_redetect(gray0, staff_arg, LEVEL)
        if noisy_tile is None:
            print(f"[{stem}] 노이즈+재검출 실패(재검출 오선 구조 불일치) -- 스킵")
            continue
        cv2.imwrite(str(OUT_DIR / f"{stem}_noisy.png"), noisy_tile)
        print(f"[{stem}] 완료")

    print(f"\n출력: {OUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
