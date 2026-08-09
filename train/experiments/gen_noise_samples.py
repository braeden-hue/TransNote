import os
import random

import cv2

from dataset import augment_image

SRC_DIR = 'data/local_pools/exactpicture_test_full'
OUT_DIR = 'data/local_pools/noise_samples'
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
        noisy = augment_image(gray, level=level)
        out_name = f'L{level}_{i}_{os.path.splitext(fname)[0]}.png'
        cv2.imwrite(os.path.join(OUT_DIR, out_name), noisy)
        print(f'saved {out_name}  (src={fname})')
