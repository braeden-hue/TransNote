"""빠른 파라미터 스윕용 (전체 100장 대신 40개 시스템, L3 geo+pixel 조합 고정)."""
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import (NOISE_LEVELS, augment_image, correct_perspective,
                      geometric_augment, load_preprocessed,
                      _redetect_with_fallback)

TEST_DIR = os.path.join(os.path.dirname(__file__), "test100_local")
LEVEL = 3
N_TRIALS = 3
SEED = 777
N_SYSTEMS = 40


def load_systems(n):
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
            if len(systems) >= n:
                return systems
    return systems


def crop_for(gray0, staff_pair, pad=350):
    y_top = int(min(s['y_lines'][0] for s in staff_pair))
    y_bot = int(max(s['y_lines'][4] for s in staff_pair))
    H, W = gray0.shape
    return gray0[max(0, y_top - pad):min(H, y_bot + pad), :]


def main():
    systems = load_systems(N_SYSTEMS)
    lvl = NOISE_LEVELS[LEVEL]
    ok, total = 0, 0
    for i, (gray0, staff_pair) in enumerate(systems):
        crop = crop_for(gray0, staff_pair)
        if crop.shape[0] < 50:
            continue
        for t in range(N_TRIALS):
            random.seed(SEED + i * 100 + t)
            np.random.seed((SEED + i * 100 + t) % (2**32 - 1))
            total += 1
            noisy, _ = geometric_augment(crop, max_angle_deg=lvl['angle_page'],
                                         persp_margin_frac=lvl['persp_page'],
                                         p_rotate=lvl['p_rotate'], p_persp=lvl['p_persp'])
            noisy = augment_image(noisy, level=LEVEL)
            corrected = correct_perspective(noisy)
            staffs_new, _ = _redetect_with_fallback(noisy, corrected, 2)
            if len(staffs_new) == 2:
                ok += 1
    print(f"{ok}/{total} = {ok/total*100:.1f}%")


if __name__ == "__main__":
    main()
