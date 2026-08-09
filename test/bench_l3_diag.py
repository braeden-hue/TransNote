"""L3 재검출 실패 원인 분해: geo(회전+원근)만 / geo+curl / geo+curl+pixel(밝기/블러/jpeg 등)
단계별로 켜가며 성공률을 잰다. 어느 단계에서 성공률이 크게 꺾이는지로 원인을 좁힌다."""
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import (NOISE_LEVELS, apply_page_curl, augment_image,
                      extract_system_canvas,
                      geometric_augment, load_preprocessed,
                      _redetect_with_fallback, _deskew)

TEST_DIR = os.path.join(os.path.dirname(__file__), "test100_local")
LEVEL = 3
N_TRIALS = 3
SEED = 777


def load_systems():
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
    return systems


def crop_for(gray0, staff_pair, pad=350):
    y_top = int(min(s['y_lines'][0] for s in staff_pair))
    y_bot = int(max(s['y_lines'][4] for s in staff_pair))
    H, W = gray0.shape
    return gray0[max(0, y_top - pad):min(H, y_bot + pad), :]


def run_stage(systems, use_geo, use_curl, use_pixel, seed_base):
    lvl = NOISE_LEVELS[LEVEL]
    ok, total = 0, 0
    for i, (gray0, staff_pair) in enumerate(systems):
        crop = crop_for(gray0, staff_pair)
        if crop.shape[0] < 50:
            continue
        for t in range(N_TRIALS):
            random.seed(seed_base + i * 100 + t)
            np.random.seed((seed_base + i * 100 + t) % (2**32 - 1))
            total += 1
            noisy = crop
            if use_geo:
                noisy, _ = geometric_augment(noisy, max_angle_deg=lvl['angle_page'],
                                             persp_margin_frac=lvl['persp_page'],
                                             p_rotate=lvl['p_rotate'], p_persp=lvl['p_persp'])
            if use_curl and random.random() < lvl['p_curl']:
                lo, hi = lvl['curl_px']
                noisy = apply_page_curl(noisy, random.uniform(lo, hi))
            if use_pixel:
                noisy = augment_image(noisy, level=LEVEL)
            corrected = _deskew(noisy)
            staffs_new, _ = _redetect_with_fallback(noisy, corrected, 2)
            if len(staffs_new) == 2:
                ok += 1
    return ok, total


def main():
    systems = load_systems()
    print(f"{len(systems)} grand-staff systems, {N_TRIALS} trials each")

    stages = [
        ("baseline (no noise)", False, False, False),
        ("geo only",             True,  False, False),
        ("geo+curl",             True,  True,  False),
        ("geo+curl+pixel(=full L3)", True, True, True),
        ("geo+pixel (no curl)",  True,  False, True),
    ]
    for name, g, c, px in stages:
        ok, total = run_stage(systems, g, c, px, SEED)
        print(f"{name:30s}: {ok}/{total} = {ok/total*100:.1f}%")


if __name__ == "__main__":
    main()
