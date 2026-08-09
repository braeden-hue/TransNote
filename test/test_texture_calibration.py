"""real_texture_augment.py가 실측 목표치(analyze_real_photos.py)에 실제로
가까워지는지 로컬에서 빠르게 확인하는 스크립트. designKit/mscz_clean_test/의
클린 캔버스 이미지에 적용해보고 밝기/대비/조명범위를 실측값과 비교한다.
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from real_texture_augment import apply_real_texture

CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'designKit', 'mscz_clean_test')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'texture_calibration_out')

# analyze_real_photos.py 실측 목표(화면촬영 제외 37장)
REAL_TARGET = dict(mean=174.5, std=58.5, grad_range=73.5)


def lighting_range(gray):
    small = cv2.resize(gray, (64, 64))
    blurred = cv2.GaussianBlur(small.astype(np.float32), (0, 0), sigmaX=8)
    return float(blurred.max() - blurred.min())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(CLEAN_DIR, '*_canvas_sys0.png')))
    print(f"테스트 대상 클린 캔버스: {len(paths)}장")
    print(f"실측 목표: mean={REAL_TARGET['mean']} std={REAL_TARGET['std']} grad_range={REAL_TARGET['grad_range']}\n")

    means, stds, ranges = [], [], []
    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        clean_mean, clean_std = float(gray.mean()), float(gray.std())
        clean_range = lighting_range(gray)

        out = apply_real_texture(gray)
        out_mean, out_std = float(out.mean()), float(out.std())
        out_range = lighting_range(out)

        name = os.path.basename(p)
        cv2.imwrite(os.path.join(OUT_DIR, name.replace('.png', '_textured.png')), out)

        print(f"{name:40s} clean(mean={clean_mean:6.1f} std={clean_std:5.1f} range={clean_range:5.1f}) "
              f"-> textured(mean={out_mean:6.1f} std={out_std:5.1f} range={out_range:5.1f})")
        means.append(out_mean); stds.append(out_std); ranges.append(out_range)

    print(f"\n=== 증강 후 집계 (n={len(paths)}) vs 실측 목표 ===")
    print(f"mean:       {np.median(means):6.1f}  (목표 {REAL_TARGET['mean']})")
    print(f"std(대비):  {np.median(stds):6.1f}  (목표 {REAL_TARGET['std']})")
    print(f"grad_range: {np.median(ranges):6.1f}  (목표 {REAL_TARGET['grad_range']})")
    print(f"\n결과 이미지: {OUT_DIR}")


if __name__ == '__main__':
    main()
