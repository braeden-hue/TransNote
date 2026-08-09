"""designKit/scoped_test/의 실사 종이 사진(화면촬영 제외)에서 재사용 가능한 왜곡
"재료"를 뽑아 round3train/real_texture_bank/에 저장한다.

뽑는 것 2가지:
  1. grain_*.png  -- 고주파 잡티 텍스처(종이결/인쇄 잡티/센서 노이즈). 저주파(조명)
     성분을 큰 커널 가우시안으로 빼고 남은 잔차. 합성 이미지에 덧셈으로 얹으면
     "디테일이 늘어나는" 실사 특유의 질감이 재현됨(2026-07-28 실측: 실사가 클린
     합성본보다 오히려 블러 지표가 높았던 이유 -- 텍스처가 국소 분산을 높임).
  2. lighting_*.png -- 저주파 조명 그라디언트 맵(정규화, 0~1). 합성 이미지에
     곱셈으로 얹으면 실사 특유의 조명 불균일이 재현됨.

analyze_real_photos.py에서 확인된 화면촬영 3장(114838227_06/_08/_09, 블러 지표
이상치)은 제외 -- 실제 종이 촬영이 아니라 무아레가 섞여 목적에 안 맞음.
"""
import glob
import os

import cv2
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'designKit', 'scoped_test')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'real_texture_bank')
EXCLUDE = {
    'KakaoTalk_20260728_114838227_06.jpg',
    'KakaoTalk_20260728_114838227_08.jpg',
    'KakaoTalk_20260728_114838227_09.jpg',
}
TARGET_SIZE = (960, 720)  # 저장 크기(가로x세로) -- 합성 캔버스에 타일링해서 쓸 수 있을 정도


def _find_blank_patch(gray, patch_h, patch_w):
    """음표 잉크가 없는(거의 균일한) 여백 영역을 슬라이딩 윈도우로 찾는다.
    노트/오선 잉크가 섞이면 로컬 표준편차가 커지므로, 표준편차가 가장 작은
    영역을 "순수 종이/조명" 패치로 채택 -- 블러로 신호 분리하는 대신 애초에
    콘텐츠가 없는 영역만 쓰는 방식(2026-07-28: 블러 분리 시도했더니 음표
    잉크가 잔차에 그대로 남는 문제 확인해서 이 방식으로 변경)."""
    h, w = gray.shape
    best_std = None
    best_patch = None
    step_y = max(1, (h - patch_h) // 24)
    step_x = max(1, (w - patch_w) // 24)
    for y in range(0, max(1, h - patch_h), step_y):
        for x in range(0, max(1, w - patch_w), step_x):
            patch = gray[y:y + patch_h, x:x + patch_w]
            if patch.shape != (patch_h, patch_w):
                continue
            std = float(patch.std())
            if best_std is None or std < best_std:
                best_std = std
                best_patch = patch.copy()
    return best_patch, best_std


def extract_from_photo(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    scale = TARGET_SIZE[0] / w
    gray = cv2.resize(gray, (TARGET_SIZE[0], int(h * scale)), interpolation=cv2.INTER_AREA)

    patch_h, patch_w = 90, 130
    patch, std = _find_blank_patch(gray, patch_h, patch_w)
    if patch is None:
        return None, None, None

    patch_f = patch.astype(np.float32)
    lighting = cv2.GaussianBlur(patch_f, (0, 0), sigmaX=25)
    grain = patch_f - lighting  # 고주파 잔차(잡티 텍스처), 평균 0 근처 -- 이제 음표 잉크가
                                 # 섞일 걱정 없이 순수 종이결/조명 성분만 분리됨

    lighting_norm = (lighting - lighting.min()) / (lighting.max() - lighting.min() + 1e-6)
    grain_vis = np.clip(grain + 128, 0, 255).astype(np.uint8)
    lighting_vis = (lighting_norm * 255).astype(np.uint8)
    return grain_vis, lighting_vis, std


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(SRC_DIR, 'KakaoTalk_*.jpg')))
    paths = [p for p in paths if os.path.basename(p) not in EXCLUDE]
    print(f"텍스처 추출 대상: {len(paths)}장 (화면촬영 {len(EXCLUDE)}장 제외)")

    ok = 0
    for i, p in enumerate(paths):
        grain_vis, lighting_vis, std = extract_from_photo(p)
        if grain_vis is None:
            print(f"  건너뜀(패치 탐색 실패): {os.path.basename(p)}")
            continue
        cv2.imwrite(os.path.join(OUT_DIR, f'grain_{ok:03d}.png'), grain_vis)
        cv2.imwrite(os.path.join(OUT_DIR, f'lighting_{ok:03d}.png'), lighting_vis)
        print(f"  {os.path.basename(p):40s} best_patch_std={std:.1f}")
        ok += 1

    print(f"완료: {OUT_DIR}에 grain_*.png {ok}개, lighting_*.png {ok}개 저장")


if __name__ == '__main__':
    main()
