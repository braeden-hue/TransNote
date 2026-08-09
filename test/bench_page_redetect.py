"""page_noise_and_redetect() 실전 성공률 측정 (5n6 게이트 판단용).
test100_local의 각 페이지에서 대보표 시스템(treble+bass 페어)을 뽑아 level=2/3로
반복 시도, None(재검출 실패 -> 캔버스 레벨 폴백)이 아닌 비율을 잰다."""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_preprocessed, page_noise_and_redetect

TEST_DIR = os.path.join(os.path.dirname(__file__), "test100_local")
N_TRIALS_PER_SAMPLE = 3  # 노이즈가 확률적이므로 샘플당 여러 번 시도해 평균


def main():
    pngs = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    systems = []
    for p in pngs:
        sj = p[:-4] + "_staffs.json"
        if not os.path.exists(sj):
            continue
        with open(sj, encoding="utf-8") as f:
            staffs = json.load(f)
        if len(staffs) < 2 or len(staffs) % 2 != 0:
            continue
        gray0 = load_preprocessed(p)
        for i in range(0, len(staffs), 2):
            systems.append((gray0, [staffs[i], staffs[i + 1]]))

    print(f"{len(systems)} grand-staff systems from {len(pngs)} pages")

    for level in (2, 3):
        ok, total = 0, 0
        for gray0, staff_pair in systems:
            for _ in range(N_TRIALS_PER_SAMPLE):
                total += 1
                tile = page_noise_and_redetect(gray0, staff_pair, level)
                if tile is not None:
                    ok += 1
        print(f"level={level}: {ok}/{total} = {ok/total*100:.1f}% redetect success")


if __name__ == "__main__":
    main()
