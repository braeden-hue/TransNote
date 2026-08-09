"""real_texture_bank/(실사 37장에서 추출한 grain/lighting 패치)를 합성 이미지에
적용하는 증강 함수. analyze_real_photos.py 실측치에 맞춰 밝기/대비/조명 불균일을
보정한다(2026-07-28: 기존 NOISE_LEVELS는 밝기/대비/조명범위가 전부 실사와 반대
방향이었음 -- 노이즈를 세게 줄수록 실사에서 더 멀어지고 있었음).

실측 목표(analyze_real_photos.py, 화면촬영 제외 37장 기준):
  밝기 평균   : 150~181 (중앙값 174.5)  -- 합성 클린본은 240.8로 너무 밝음
  대비(std)   : 52~66   (중앙값 58.5)   -- 합성 클린본은 48.0으로 너무 낮고,
                                          기존 노이즈는 더 낮춤(반대 방향)
  조명 범위   : 34~117  (중앙값 73.5)   -- 합성 클린본은 31.4로 너무 균일

2026-07-28 추가: 파이프라인이 그레이스케일 전용이라(dataset.preprocess가 BGR->GRAY로
변환) 실사 중 옅은 황색 종이처럼 "색상"은 그대로 재현할 수 없다. 대신 그런 사진은 대체로
그레이스케일 밝기 자체가 더 어둡게 찍히므로, target_mean_range를 실측 하한(150)보다 더
아래(140)까지 넓혀 어둡고 따뜻한 톤의 사진도 일부 커버한다.
"""
import glob
import os
import random

import cv2
import numpy as np

BANK_DIR = os.path.join(os.path.dirname(__file__), 'real_texture_bank')

_GRAIN_CACHE = None
_LIGHTING_CACHE = None


def load_texture_bank():
    global _GRAIN_CACHE, _LIGHTING_CACHE
    if _GRAIN_CACHE is None:
        _GRAIN_CACHE = [cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                        for f in sorted(glob.glob(os.path.join(BANK_DIR, 'grain_*.png')))]
        _LIGHTING_CACHE = [cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                           for f in sorted(glob.glob(os.path.join(BANK_DIR, 'lighting_*.png')))]
    return _GRAIN_CACHE, _LIGHTING_CACHE


def _mosaic_to_size(patches, h, w):
    """그레인 패치 여러 장을 셀마다 무작위로 골라 이어붙인다. 같은 패치 하나를
    np.tile로 반복하면 패치 크기 간격(90x130)으로 뚜렷한 격자무늬가 보이는 문제가
    있었다(2026-07-28 실사 비교 피드백: "종이 곳곳에 점이 박힌 양상 같다" -- 실제로는
    타일 반복 아티팩트). 셀마다 다른 사진 출처의 패치를 무작위로 배치하면 완전한
    주기성이 깨져 격자무늬가 사라진다."""
    ph, pw = patches[0].shape
    rows = h // ph + 2
    cols = w // pw + 2
    canvas = np.zeros((rows * ph, cols * pw), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            canvas[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = random.choice(patches).astype(np.float32)
    oy = random.randint(0, ph - 1)
    ox = random.randint(0, pw - 1)
    mosaic = canvas[oy:oy + h, ox:ox + w]
    return cv2.GaussianBlur(mosaic, (0, 0), sigmaX=1.2)  # 타일 경계 이음매를 살짝 부드럽게


def _stretch_to_size(patch, h, w):
    """조명 패치는 타일링하지 않고 캔버스 전체 크기로 늘려서(1장짜리 연속 그라디언트)
    적용한다 -- 실제 사진 한 장은 광원 하나가 만드는 그라디언트가 페이지 전체에 걸쳐
    한 번만 나타나기 때문. 좌우/상하 뒤집기로 그라디언트 방향에 다양성을 준다."""
    if random.random() < 0.5:
        patch = patch[:, ::-1]
    if random.random() < 0.5:
        patch = patch[::-1, :]
    return cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_real_texture(gray: np.ndarray,
                        target_mean_range=(140, 186),
                        contrast_boost_range=(1.05, 1.20),
                        lighting_strength_range=(0.14, 0.32),
                        grain_strength_range=(0.35, 0.75)) -> np.ndarray:
    """합성 클린 이미지에 실사 조명/텍스처를 입혀 실측 통계에 맞춘다."""
    grains, lightings = load_texture_bank()
    if not grains or not lightings:
        return gray  # 텍스처 뱅크 없으면 원본 그대로(pod에 아직 안 올라간 경우 등)

    h, w = gray.shape
    grain = _mosaic_to_size(grains, h, w) - 128.0
    lighting = _stretch_to_size(random.choice(lightings), h, w).astype(np.float32) / 255.0

    out = gray.astype(np.float32)

    # 1) 조명 불균일 적용(곱셈) -- lighting을 [1-strength, 1] 범위로 리스케일해서
    #    strength가 클수록 그림자가 더 진해짐(실측 조명범위 73.5 근처를 목표)
    strength = random.uniform(*lighting_strength_range)
    lighting_scaled = (1.0 - strength) + strength * lighting
    out = out * lighting_scaled

    # 2) 잡티 텍스처(그레인) 덧셈 -- 종이결/인쇄잡티/센서노이즈 재현
    grain_strength = random.uniform(*grain_strength_range)
    out = out + grain * grain_strength

    # 3) 대비 보정 -- 실사가 클린 합성본보다 오히려 대비가 높으므로(48.0 -> 58.5) 살짝 키움.
    #    조명/그레인을 이미 입힌 뒤 현재 평균 기준으로 적용해야 대비가 왜곡되지 않는다.
    cur_mean = out.mean()
    contrast_boost = random.uniform(*contrast_boost_range)
    out = (out - cur_mean) * contrast_boost + cur_mean

    # 4) 밝기 최종 보정 -- 실측 중앙값(174.5) 근처로 타겟팅. 앞 단계들이 평균을 흔들어
    #    놓으므로 반드시 마지막에 해야 목표 밝기가 정확히 맞는다.
    target_mean = random.uniform(*target_mean_range)
    out = out - (out.mean() - target_mean)

    return np.clip(out, 0, 255).astype(np.uint8)
