"""detect_staffs (전역 투영) vs detect_staffs_curved (스트립 기반) 성공률 비교.
회전 + apply_page_curl 합성 왜곡 하에서, 보정(correct_perspective) 유무별로 4가지
조합(raw/global, corrected/global, raw/curved, corrected/curved)의 오선 검출 성공률을 잰다.
성공 기준: 검출된 오선 그룹 개수가 정답(_staffs.json)과 일치.
"""
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import (apply_page_curl, correct_perspective, detect_staffs,
                      detect_staffs_curved, dewarp_page, geometric_augment,
                      load_preprocessed)

TEST_DIR = os.path.join(os.path.dirname(__file__), "test100_local")
N_SAMPLES = 100
CURL_STRENGTHS = [0, 15, 25, 40]
ANGLE_DEG = 6.0
SEED = 12345


def load_samples(n):
    pngs = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    random.Random(SEED).shuffle(pngs)
    out = []
    for p in pngs[:n]:
        base = p[:-4]
        sj = base + "_staffs.json"
        if not os.path.exists(sj):
            continue
        with open(sj, "r", encoding="utf-8") as f:
            gt = json.load(f)
        gray = load_preprocessed(p)
        if gray is None:
            continue
        out.append((os.path.basename(p), gray, len(gt)))
    return out


def distort(gray, curl_px, rng_seed):
    random.seed(rng_seed)
    np.random.seed(rng_seed % (2**32 - 1))
    out, _ = geometric_augment(gray, max_angle_deg=ANGLE_DEG, persp_margin_frac=0.03,
                               p_rotate=1.0, p_persp=0.0)
    if curl_px > 0:
        out = apply_page_curl(out, curl_px)
    return out


def main():
    samples = load_samples(N_SAMPLES)
    print(f"{len(samples)} samples loaded from {TEST_DIR}")

    keys = ["raw_global", "corr_global", "raw_curved", "corr_curved", "corr_curved_dewarp"]
    results = {}
    for curl in CURL_STRENGTHS:
        counts = {k: 0 for k in keys}
        n = 0
        for i, (name, gray, gt_n) in enumerate(samples):
            distorted = distort(gray, curl, rng_seed=SEED + i)
            n += 1

            g_raw = detect_staffs(distorted)
            if len(g_raw) == gt_n:
                counts["raw_global"] += 1

            corrected = correct_perspective(distorted)
            g_corr = detect_staffs(corrected)
            if len(g_corr) == gt_n:
                counts["corr_global"] += 1

            c_raw = detect_staffs_curved(distorted)
            if len(c_raw) == gt_n:
                counts["raw_curved"] += 1

            c_corr = detect_staffs_curved(corrected)
            if len(c_corr) == gt_n:
                counts["corr_curved"] += 1

            if c_corr:
                dewarped, staffs_after = dewarp_page(corrected, c_corr)
                c_corr_dw = detect_staffs_curved(dewarped)
                if len(c_corr_dw) == gt_n:
                    counts["corr_curved_dewarp"] += 1

        results[curl] = {k: v / n * 100 for k, v in counts.items()}
        print(f"\n=== curl={curl}px, rotate<= {ANGLE_DEG} deg, n={n} ===")
        for k, v in results[curl].items():
            print(f"  {k:20s}: {v:5.1f}%")

    print("\n\nSummary table (success %):")
    header = f"{'curl_px':>8} | " + " | ".join(f"{k:>18}" for k in keys)
    print(header)
    print("-" * len(header))
    for curl in CURL_STRENGTHS:
        r = results[curl]
        print(f"{curl:>8} | " + " | ".join(f"{r[k]:>17.1f}%" for k in keys))


if __name__ == "__main__":
    main()
