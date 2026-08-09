"""특정 구역에 더 강한 그림자를 의도적으로 준 버전 + 흐릿함(블러) 버전 시각 확인용.
apply_curl_shade()는 이미 corner/strength/radius_frac로 국소 그림자를 지원하고,
_apply_blur()/_motion_blur()도 이미 구현돼 있음(2026-07-31 확인) -- 기본 강도보다
훨씬 강하게 줘서 효과를 명확히 보이게 함.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as ds

SRC_DIR = Path(__file__).resolve().parent / 'data' / 'local_pools' / 'r3_density_register_clef'
OUT_DIR = Path(__file__).resolve().parent / 'data' / 'local_pools' / 'shadow_blur_test'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pngs = sorted(p for p in SRC_DIR.glob('*.png') if '_ms_tmp' not in p.stem)
    picked = all_pngs[100:103]  # 앞서 쓴 5장과 겹치지 않게 다른 샘플로

    for png_path in picked:
        stem = png_path.stem
        gray0 = ds.load_preprocessed(str(png_path))
        staffs, gray = ds.best_effort_staff_detection(gray0)
        if not staffs:
            print(f"[{stem}] 오선 검출 실패 -- 스킵")
            continue
        n = len(staffs)
        if n >= 2 and n % 2 == 0:
            tile = ds.extract_system_canvas(gray, staffs[:2])
        else:
            tile = ds.extract_staff_canvas(gray, staffs[0])

        cv2.imwrite(str(OUT_DIR / f"{stem}_clean.png"), tile)
        H, W = tile.shape

        # 1) 그림자를 "가운데 + 초반부(왼쪽)"에 골고루 -- corner 대신 center를 직접
        #    지정(2026-07-31 dataset.py에 center 파라미터 추가). 세로 중앙(H/2),
        #    가로는 왼쪽 1/4 지점을 중심으로 반경을 넓게(radius_frac 0.45) 잡아서
        #    중앙~초반 구간에 고르게 퍼지게 함.
        strong_shadow = ds.apply_curl_shade(tile, center=(H / 2, W * 0.25),
                                             strength=0.6, radius_frac=0.45, darken=True)
        cv2.imwrite(str(OUT_DIR / f"{stem}_strong_shadow.png"), strong_shadow)

        # 2) 흐릿함 -- 이전보다 약하게(모션 블러 커널 9~13 -> 5~7, 추가 가우시안 생략)
        blurred = ds._motion_blur(tile, ksize_range=(5, 7))
        cv2.imwrite(str(OUT_DIR / f"{stem}_blur.png"), blurred)

        # 3) 책 제본선(머릿부분)이 볼록 솟아오른 대칭 굴곡(신규 apply_page_curl_bump) --
        #    가로 중앙(peak_frac=0.5)이 가장 볼록, 양 끝은 평평.
        bump = ds.apply_page_curl_bump(tile, strength_px=10.0, peak_frac=0.5, axis='x')
        cv2.imwrite(str(OUT_DIR / f"{stem}_bump_curl.png"), bump)

        # 4) 참고: 중앙 그림자 + 완화된 블러 + 범프 굴곡 동시 적용
        combined = ds.apply_page_curl_bump(tile, strength_px=8.0, peak_frac=0.5, axis='x')
        combined = ds.apply_curl_shade(combined, center=(H / 2, W * 0.25), strength=0.5, radius_frac=0.45)
        combined = ds._motion_blur(combined, ksize_range=(5, 7))
        cv2.imwrite(str(OUT_DIR / f"{stem}_combined.png"), combined)

        print(f"[{stem}] 완료")

    print(f"\n출력: {OUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
