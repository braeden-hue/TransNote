import os
import random

import cv2

from dataset import detect_staffs, page_noise_and_redetect

SRC_DIR = 'data/local_pools/exactpicture_test_full'
OUT_DIR = 'data/local_pools/noise_samples_pagelevel'
os.makedirs(OUT_DIR, exist_ok=True)

candidates = sorted(
    f for f in os.listdir(SRC_DIR)
    if f.endswith('.png') and '_ms_tmp' not in f
)
random.seed(7)
chosen = random.sample(candidates, 5)

for level in (3, 4):
    for i, fname in enumerate(chosen, 1):
        gray = cv2.imread(os.path.join(SRC_DIR, fname), cv2.IMREAD_GRAYSCALE)
        staffs = detect_staffs(gray)
        if not staffs:
            print(f'skip {fname}: 오선 검출 실패')
            continue
        staff_arg = staffs if len(staffs) > 1 else staffs[0]
        random.seed(100 + i)  # 같은 (level, i) 조합이 아니면 다른 왜곡 -- level별 비교를 위해 i마다 시드 고정
        noisy = page_noise_and_redetect(gray, staff_arg, level)
        if noisy is None:
            print(f'  L{level} {fname}: 재검출 실패(폴백 없이 None 반환)')
            continue
        out_name = f'L{level}_{i}_{os.path.splitext(fname)[0]}.png'
        cv2.imwrite(os.path.join(OUT_DIR, out_name), noisy)
        print(f'saved {out_name}  (src={fname})')
