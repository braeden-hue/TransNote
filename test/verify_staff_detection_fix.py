"""load_staffs_cached()를 best_effort_staff_detection()으로 바꾸면 실제로 개선되는지
간단 검증 -- 실사 사진 표본에 대해 기존(detect_staffs)과 개선안(best_effort)의 검출
성공률/검출 개수를 비교. 코드 수정 전 사전 검증용, 학습에는 영향 없음(읽기 전용).
"""
import glob
import os
import random

import cv2

from dataset import detect_staffs, best_effort_staff_detection, preprocess

REAL_DIR = '/workspace/data/classical_realphotos'


def main():
    photos = sorted(glob.glob(os.path.join(REAL_DIR, '*.jpg')))
    random.seed(3)
    sample = random.sample(photos, min(40, len(photos)))

    old_fail = old_zero = 0
    new_fail = new_zero = 0
    old_counts = []
    new_counts = []
    improved = []
    for p in sample:
        raw = cv2.imread(p, cv2.IMREAD_COLOR)
        if raw is None:
            continue
        gray = preprocess(raw)
        old_staffs = detect_staffs(gray)
        new_staffs, _ = best_effort_staff_detection(gray)
        old_counts.append(len(old_staffs))
        new_counts.append(len(new_staffs))
        if len(old_staffs) == 0:
            old_zero += 1
        if len(new_staffs) == 0:
            new_zero += 1
        if len(new_staffs) > len(old_staffs):
            improved.append((os.path.basename(p), len(old_staffs), len(new_staffs)))

    n = len(old_counts)
    print(f"표본 {n}장")
    print(f"기존 detect_staffs():        0개 검출(완전실패) {old_zero}/{n}, 평균 검출개수 {sum(old_counts)/n:.2f}")
    print(f"best_effort_staff_detection(): 0개 검출(완전실패) {new_zero}/{n}, 평균 검출개수 {sum(new_counts)/n:.2f}")
    print(f"\nbest_effort가 더 많이 찾은 경우: {len(improved)}/{n}")
    for name, o, nn in improved[:15]:
        print(f"  {name}: {o} -> {nn}")

    # 1 또는 2(단일오선/대보표 한 시스템)가 "정상"으로 기대되는 값
    old_reasonable = sum(1 for c in old_counts if c in (1, 2))
    new_reasonable = sum(1 for c in new_counts if c in (1, 2))
    print(f"\n검출개수가 1 또는 2(정상 범위)인 비율: 기존 {old_reasonable}/{n}, best_effort {new_reasonable}/{n}")


if __name__ == '__main__':
    main()
