"""L2 기본 강도 vs 데모에서 쓴 강한 강도를 같은 원본 이미지로 나란히 비교.
2026-07-31 사용자 요청 -- "강도 약한데?" 피드백에 대한 시각적 판단 자료.
두 버전 다 효과 발생 확률은 100%로 고정(비교 목적, 등장 여부가 아니라 강도 자체만 비교),
강도(수치)만 다르게 적용.
"""
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as ds

SRC = Path(__file__).resolve().parent / 'data' / 'local_pools' / 'r3_density_register_clef' / 'num40000920.png'
OUT_DIR = Path(__file__).resolve().parent / 'data' / 'local_pools' / 'intensity_compare'


def make_version(gray0, staff_arg, curl_px, curl_strength, shade_strength, shade_radius, label):
    level_cfg = copy.deepcopy(ds.NOISE_LEVELS[2])
    level_cfg['p_curl'] = 1.0
    level_cfg['p_bump_curl'] = 0.0  # 램프형으로 고정(비교 단순화)
    level_cfg['curl_px'] = curl_px
    level_cfg['p_rotate'] = 1.0
    level_cfg['p_persp'] = 0.0
    level_cfg['p_curl_shade'] = 1.0
    ds.NOISE_LEVELS[2] = level_cfg

    # apply_curl_shade의 strength/radius_frac은 augment_image() 내부에서 고정값으로
    # 호출되므로, 비교를 위해 여기서는 별도로 직접 호출해 강도를 바꿔본다.
    tile = ds.page_noise_and_redetect(gray0, staff_arg, 2)
    if tile is None:
        return None
    # page_noise_and_redetect가 이미 augment_image(기본 그림자 강도 포함)를 거쳤으므로,
    # "강한 그림자" 버전은 추가로 한 번 더 강하게 얹어서 비교치를 명확히 한다.
    tile = ds.apply_curl_shade(tile, corner=None, strength=shade_strength,
                                radius_frac=shade_radius, darken=True)
    return tile


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gray0 = ds.load_preprocessed(str(SRC))
    staffs, gray = ds.best_effort_staff_detection(gray0)
    staff_arg = staffs[0] if len(staffs) == 1 else staffs[:2]

    weak = make_version(gray0, staff_arg, curl_px=(4, 14), curl_strength=None,
                         shade_strength=0.25, shade_radius=0.35, label='weak(L2 기본)')
    strong = make_version(gray0, staff_arg, curl_px=(10, 20), curl_strength=None,
                          shade_strength=0.65, shade_radius=0.3, label='strong(데모 수준)')

    if weak is None or strong is None:
        print('재검출 실패로 비교 이미지 생성 못함')
        return

    # 높이를 맞추고 위아래로 이어붙여 라벨과 함께 저장.
    h = max(weak.shape[0], strong.shape[0])
    w = max(weak.shape[1], strong.shape[1])

    def pad(img):
        out = np.full((h, w), 255, dtype=np.uint8)
        out[:img.shape[0], :img.shape[1]] = img
        return out

    weak_p = pad(weak)
    strong_p = pad(strong)
    gap = np.full((30, w), 255, dtype=np.uint8)
    combined = np.vstack([weak_p, gap, strong_p])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    cv2.putText(combined_bgr, 'WEAK (L2 default: shade 0.25, curl 4-14px)', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(combined_bgr, 'STRONG (demo-level: shade 0.65, curl 10-20px)', (10, h + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = OUT_DIR / 'compare_weak_vs_strong.png'
    cv2.imwrite(str(out_path), combined_bgr)
    print(f'저장: {out_path.resolve()}')


if __name__ == '__main__':
    main()
