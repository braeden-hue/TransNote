"""run_image()이 실제로 모델에 넣는 캔버스(오선 검출+크롭 후)를 그대로 저장해서
눈으로 확인. 오선 검출/크롭 단계에서 내용이 잘리는지 직접 검증."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import cv2
from dataset import (best_effort_staff_detection, extract_staff_canvas,
                      extract_system_canvas, load_preprocessed)

paths = sys.argv[1:]
for p in paths:
    gray = load_preprocessed(p)
    staffs, src = best_effort_staff_detection(gray)
    n = len(staffs)
    print(f"{p}: {n}개 오선 검출")
    stem = os.path.splitext(p)[0]

    if n >= 2 and n % 2 == 0:
        n_systems = n // 2
        for sys_i in range(n_systems):
            canvas = extract_system_canvas(src, [staffs[sys_i * 2], staffs[sys_i * 2 + 1]])
            out = f"{stem}_canvas_sys{sys_i}.png"
            cv2.imwrite(out, canvas)
            print(f"  -> {out} ({canvas.shape[1]}x{canvas.shape[0]})")
    else:
        for i, s in enumerate(staffs):
            canvas = extract_staff_canvas(src, s)
            out = f"{stem}_canvas_staff{i}.png"
            cv2.imwrite(out, canvas)
            print(f"  -> {out} ({canvas.shape[1]}x{canvas.shape[0]})")
