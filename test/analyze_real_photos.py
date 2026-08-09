"""designKit/scoped_test/의 실사 촬영 40장을 분석해서 조명/블러/기울기/질감 특성을
정량적으로 뽑아낸다. NOISE_LEVELS 파라미터 실측 보정 + 조명 마스크 추출용 1차 스캔.
"""
import glob
import os
import sys

import cv2
import numpy as np

TARGET_W = 1200  # 분석용으로만 축소(원본 4032x3024는 무거움) -- 실제 마스크 추출 시엔 별도 처리


def load_gray_resized(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    scale = TARGET_W / w
    im = cv2.resize(im, (TARGET_W, int(h * scale)), interpolation=cv2.INTER_AREA)
    return im


def blur_score(gray):
    """라플라시안 분산 -- 낮을수록 흐릿함(블러 심함)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_stats(gray):
    return float(gray.mean()), float(gray.std())


def lighting_nonuniformity(gray):
    """저주파(조명) 성분만 남기고 전체적인 밝기 불균일도(그라데이션 강도) 측정."""
    small = cv2.resize(gray, (64, 64))
    blurred = cv2.GaussianBlur(small.astype(np.float32), (0, 0), sigmaX=8)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return float(grad_mag.mean()), float(blurred.max() - blurred.min())


def moire_score(gray):
    """FFT 고주파 대역 에너지 비율 -- 스크린 촬영 특유의 모아레(주기적 패턴) 탐지용."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r_outer = min(h, w) // 3
    r_inner = min(h, w) // 6
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    band = (dist >= r_inner) & (dist <= r_outer)
    total = mag.sum()
    band_energy = mag[band].sum()
    return float(band_energy / (total + 1e-9))


def skew_angle_estimate(gray):
    """오선(수평선) 각도 추정 -- Hough 변환으로 지배적인 직선 각도 찾기."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=int(gray.shape[1] * 0.3))
    if lines is None:
        return None
    angles = []
    for rho_theta in lines[:50]:
        rho, theta = rho_theta[0]
        deg = (theta * 180 / np.pi) - 90
        if -20 <= deg <= 20:
            angles.append(deg)
    if not angles:
        return None
    return float(np.median(angles))


def main():
    test_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', 'designKit', 'scoped_test')
    paths = sorted(glob.glob(os.path.join(test_dir, 'KakaoTalk_*.jpg')))
    print(f"분석 대상: {len(paths)}장\n")

    rows = []
    for p in paths:
        gray = load_gray_resized(p)
        blur = blur_score(gray)
        mean_b, std_b = brightness_stats(gray)
        grad_mean, grad_range = lighting_nonuniformity(gray)
        moire = moire_score(gray)
        skew = skew_angle_estimate(gray)
        rows.append(dict(name=os.path.basename(p), blur=blur, mean=mean_b, std=std_b,
                          grad_mean=grad_mean, grad_range=grad_range, moire=moire, skew=skew))
        flag = " <== 모아레 의심(스크린 촬영?)" if moire > 0.08 else ""
        print(f"{os.path.basename(p):40s} blur={blur:8.1f} mean={mean_b:5.1f} std={std_b:5.1f} "
              f"grad={grad_mean:5.2f} range={grad_range:5.1f} moire={moire:.4f} "
              f"skew={skew if skew is not None else 'N/A'}{flag}")

    blurs = [r['blur'] for r in rows]
    means = [r['mean'] for r in rows]
    grads = [r['grad_mean'] for r in rows]
    skews = [r['skew'] for r in rows if r['skew'] is not None]
    moires = [r['moire'] for r in rows]

    print(f"\n=== 집계 (n={len(rows)}) ===")
    print(f"블러(라플라시안 분산): min={min(blurs):.1f} max={max(blurs):.1f} 중앙값={np.median(blurs):.1f}")
    print(f"평균 밝기: min={min(means):.1f} max={max(means):.1f} 중앙값={np.median(means):.1f}")
    print(f"조명 그라데이션 강도: min={min(grads):.2f} max={max(grads):.2f} 중앙값={np.median(grads):.2f}")
    if skews:
        print(f"기울기(도): min={min(skews):.2f} max={max(skews):.2f} 중앙값={np.median(skews):.2f} (검출 {len(skews)}/{len(rows)})")
    print(f"모아레 지수: min={min(moires):.4f} max={max(moires):.4f} 중앙값={np.median(moires):.4f}")
    suspects = [r['name'] for r in rows if r['moire'] > 0.08]
    print(f"모아레(스크린 촬영 의심) {len(suspects)}장: {suspects}")


if __name__ == '__main__':
    main()
