"""sample10_realtexture/의 num1~10.png(클린 렌더)에서 오선 캔버스를 뽑아,
실사 텍스처 증강(augment_image, level=2/4) 전/후를 나란히 붙여 눈으로 비교할 수
있게 저장한다. 사용자가 실사 느낌이 얼마나 나는지 직접 확인하기 위한 용도."""
import glob
import os

import cv2
import numpy as np

from dataset import load_preprocessed, best_effort_staff_detection, \
    extract_system_canvas, extract_staff_canvas, augment_image

SRC_DIR = os.path.join(os.path.dirname(__file__), 'sample10_realtexture')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'sample10_comparison')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(SRC_DIR, 'num*.png')))
    print(f"대상: {len(paths)}장")

    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        gray = load_preprocessed(p)
        staffs, corrected = best_effort_staff_detection(gray)
        if not staffs:
            print(f"  {stem}: 오선 검출 실패, 스킵")
            continue

        if len(staffs) >= 2:
            canvas = extract_system_canvas(corrected, staffs[:2])
        else:
            canvas = extract_staff_canvas(corrected, staffs[0])

        level = 4 if len(paths) % 2 == 0 else 2  # 절반은 L2, 절반은 L4로 다양하게
        idx = paths.index(p)
        level = 2 if idx % 2 == 0 else 4
        aug = augment_image(canvas, level=level)

        # 클린/증강 나란히 붙이기(같은 높이로 맞춤)
        h = max(canvas.shape[0], aug.shape[0])
        w = canvas.shape[1] + aug.shape[1] + 20
        combo = np.full((h, w), 255, dtype=np.uint8)
        combo[:canvas.shape[0], :canvas.shape[1]] = canvas
        combo[:aug.shape[0], canvas.shape[1] + 20:canvas.shape[1] + 20 + aug.shape[1]] = aug

        out_path = os.path.join(OUT_DIR, f'{stem}_L{level}_compare.png')
        cv2.imwrite(out_path, combo)
        print(f"  {stem}: 저장 -> {os.path.basename(out_path)} (level={level}, staffs={len(staffs)})")

    print(f"\n완료: {OUT_DIR}")


if __name__ == '__main__':
    main()
