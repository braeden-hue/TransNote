"""
round3train/dataset.py  –  Grand staff (대보표) 지원 OMR 데이터셋.

Round 3 핵심 변경:
1. 대보표 이미지: 오선 2개가 한 쌍(treble + bass)
   - 짝수 인덱스 오선 (0, 2, ...) = treble stave
   - 홀수 인덱스 오선 (1, 3, ...) = bass stave
2. 토큰 시퀀스를 staff-bass 기준으로 분리:
   - treble stave → treble 토큰만
   - bass stave   → bass 토큰만
3. 단일 오선 이미지(Round 1/2 누적 데이터)도 그대로 지원

토큰 구조 (대보표 1 system, 2마디 예시):
  [SOS, clef-G, key-C, time-4/4]
  [treble_m1] staff-bass clef-F [bass_m1] barline
  [treble_m2] staff-bass [bass_m2] barline-final

→ treble 샘플: header + treble_m1 + barline + treble_m2 + barline-final
→ bass   샘플: clef-F + bass_m1 + barline + bass_m2 + barline-final
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from real_texture_augment import apply_real_texture

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

PAD_ID       = 0
SOS_ID       = 1
EOS_ID       = 2

PATCH_SIZE   = 320
CANVAS_H        = 320   # MARGIN_UNITS=3.0 기준 (4+2*3=10 units); 256은 margin=2.0(8 units) 시절 값,
                         # 그대로 두면 오선 한 칸당 픽셀 밀도가 20% 줄어 음높이 구분 해상도 저하
CANVAS_W        = 1280
SYSTEM_CANVAS_H = 480   # 위와 동일 이유로 384(half=192, margin=2.0 기준)에서 비례 확대(half=240)
MARGIN_UNITS    = 3.0  # 2.0이면 최고/최저음(B5/C2 등) 음표머리·임시표가 크롭 경계에서 잘림
MARGIN_UNITS_CAP = 10.0  # 옥타브(ottava) 브래킷+텍스트처럼 MARGIN_UNITS보다 훨씬 멀리 그려지는
                         # 기호가 있으면 이 상한까지 콘텐츠 기준으로 마진을 늘림(_content_y_extent)

# 2026-08-03: CANVAS_H/MARGIN_UNITS 설계 의도("4+2*3=10 units")를 실제 크롭 스케일/배치
# 로직에 그대로 반영 -- 오선이 항상 이 픽셀당유닛으로, 항상 이 y좌표에 오도록 고정한다
# (기존엔 이 상수들이 캔버스 "크기"만 정하고, 실제 스케일/배치는 마디마다 다른 콘텐츠
# 높이 기준이라 오선의 절대 위치/크기가 흔들렸음 -- 3도(선/칸) 오독 근본원인 대응).
STAFF_UNIT_PX        = CANVAS_H / (4 + 2 * MARGIN_UNITS)         # = 32.0
STAFF_ANCHOR_TOP_PX  = int(round(MARGIN_UNITS * STAFF_UNIT_PX))  # = 96 (오선 첫 줄 고정 y)
SYSTEM_STAFF_UNIT_PX       = (SYSTEM_CANVAS_H / 2) / (4 + 2 * MARGIN_UNITS)  # = 24.0
SYSTEM_STAFF_ANCHOR_TOP_PX = int(round(MARGIN_UNITS * SYSTEM_STAFF_UNIT_PX))  # = 72
TARGET_W     = 1920
IMG_MEAN     = 0.7931
IMG_STD      = 0.1738


def make_model_input(canvas: np.ndarray, in_ch: int = 1) -> torch.Tensor:
    """정규화된 그레이스케일 캔버스([H,W], IMG_MEAN/IMG_STD 적용 완료)를 모델 입력 텐서로
    변환. in_ch=1(기본)이면 기존과 동일하게 [1,H,W]. in_ch=2면 CoordConv 실험용 -- 세로
    좌표를 [-1,1]로 정규화한 채널을 하나 더 붙여 [2,H,W]로 만든다(2026-07-31, 단3도/옥타브
    오독이 모델에 세로 위치 정보가 전혀 없는 구조적 한계 때문일 수 있다는 가설 검증).
    학습(dataset.py OMRDataset)과 추론(inference.py) 양쪽에서 반드시 같은 함수를 써야
    두 경로의 입력이 어긋나지 않는다."""
    canvas_t = torch.from_numpy(canvas).unsqueeze(0)
    if in_ch == 1:
        return canvas_t
    H, W = canvas.shape
    coord = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)
    coord_t = torch.from_numpy(coord).unsqueeze(0)
    return torch.cat([canvas_t, coord_t], dim=0)


SEG_BG         = 0
SEG_STEM_REST  = 1
SEG_NOTEHEAD   = 2
SEG_CLEF_KEY   = 3
SEG_STAFF_LINE = 4
SEG_SYMBOL     = 5
SEG_NUM_CLS    = 6

MIN_UNIT = 8.0  # 2026-07-31: 11.0은 Step1의 새 동적 와이드 페이지(마디당 2.5in, 최대 16in
# 폭)에서 preprocess()가 고정 TARGET_W=1920으로 리사이즈하면 오선 간격이 9px대까지 내려가는
# 케이스를 대량으로 걸러내버렸다(실측: 60장 샘플 기준 11.0에서 18/60만 검출 -> 8.0에서
# 59/60). 렌더링/콘텐츠 문제가 아니라 이 임계값 자체가 새 레이아웃엔 너무 엄격했던 것.
MAX_UNIT = 150.0  # 마디 수가 적은 페이지는 MuseScore가 오선을 크게 렌더링해 간격이
                  # ~113~116px까지 벌어짐 (기존 60은 다마디 페이지 기준) -- 여유있게 확장

_HEADER_PREFIXES = ('clef-', 'key-', 'time-')
_HEADER_SPECIALS = {'<SOS>', '<EOS>', '<PAD>', '<UNK>'}
_BARLINE_STRS    = frozenset({
    'barline', 'barline-final', 'barline-end-repeat',
    'barline-start-repeat', 'barline-double',
})


# ─────────────────────────────────────────────────────────────────────────────
#  전처리 (C++ 미러)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(bgr: np.ndarray, is_real_photo: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 여러 시스템(줄)이 있으면 각 줄이 서로 떨어진 별개 컨투어로 잡힌다. 예전엔 가장 큰
        # 컨투어 "하나"만 남겼는데, 그러면 한 줄짜리 페이지에서는 제목 텍스트만 잘 배제됐지만
        # 여러 줄 페이지에서는 제일 큰 줄 하나만 남고 나머지 줄이 통째로 잘려나갔음(마디 수를
        # 늘려 여러 줄로 자동 줄바꿈되는 경우에 뒤늦게 발견된 버그). 아주 작은 노이즈(안티에일리어싱
        # 잡티 등)만 제외하고 유의미한 컨투어 전부를 합친 바운딩박스를 쓴다.
        min_area = max(4.0, 0.00005 * gray.shape[0] * gray.shape[1])
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
        if boxes:
            x  = min(b[0] for b in boxes)
            y  = min(b[1] for b in boxes)
            x2 = max(b[0] + b[2] for b in boxes)
            y2 = max(b[1] + b[3] for b in boxes)
            w, h = x2 - x, y2 - y
            if w * h < 0.80 * gray.shape[0] * gray.shape[1]:
                # py를 1%->2%로 상향 -- 1%는 페이지 첫 시스템에 유난히 높은 음(레저선)이
                # 있으면 그 위쪽 실제 렌더 여백(MuseScore 기본 페이지 상단 여백)까지
                # 거의 다 잘라내버려서, 이후 _staff_y_bounds가 요구하는 MARGIN_UNITS
                # 여백을 확보 못해 오선이 캔버스 맨 위에 거의 붙어버리는 문제가 있었다
                # (2026-07-28 사용자 피드백: "악보가 너무 많이 잘렸음", num9 사례로 확인).
                px = max(2, gray.shape[1] // 100)
                py = max(2, gray.shape[0] // 50)
                gray = gray[max(0, y-py):y2+py, max(0, x-px):x2+px]
    H, W = gray.shape
    if W != TARGET_W:
        scale = TARGET_W / W
        # 2026-08-10: 폭 기준 축소만 쓰면 8마디급 와이드 대보표(측정 실폭이 TARGET_W보다
        # 훨씬 큼)에서 세로 해상도가 과도하게 눌려 오선 간격이 MIN_UNIT(8px, 위 주석의
        # 2026-07-31 완화 이후 기준)보다도 더 내려가 검출 자체가 실패하는 사례를 확인
        # (r18 held-out 합성 테스트, num21 등 -- 리사이즈 후 196x1920으로 붕괴, 오선 0개
        # 검출). MIN_UNIT을 더 내리는 대신(노이즈 오검출 위험 커짐), 세로 해상도가 이
        # 바닥 아래로 안 내려가게 축소율 자체를 완화 -- 폭이 TARGET_W를 넘어갈 수 있음
        # (detect_staffs()의 커널 폭(W//8)은 이미 W에 비례해서 별도 조정 불필요).
        MIN_RESULT_H = 280.0
        if H * scale < MIN_RESULT_H:
            scale = MIN_RESULT_H / H
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        gray  = cv2.resize(gray, (new_w, new_h),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    # 2026-08-02 1차 시도: 실사 사진에 CLAHE 강화(clipLimit 2.0)+bilateralFilter 생략+
    # 강한 언샤프 마스크(1.5/-0.5, sigma=3)를 적용해 임시표/화음 디테일을 살리려 했으나,
    # newage 51장 재검증(4곡 표본)에서 오히려 31.1%->58.7%로 나빠짐(카메라 잡음/JPEG
    # 압축 아티팩트까지 같이 증폭된 것으로 추정) -- 검증 없이 적용했던 실수, 되돌렸었음.
    #
    # 2차 시도(현재): newage 오류분석에서 "깨끗한 렌더링은 90%+ 인식하는데 실사에서만
    # 임시표(#,b)를 대량으로 놓친다"는 게 확인돼(콘텐츠 학습 문제가 아니라 촬영 열화
    # 문제) 재시도. 이번엔 순서를 바꿔 bilateralFilter(노이즈 제거)를 먼저 그대로 걸고,
    # 그 위에 아주 약한 언샤프 마스크(1.15/-0.15, sigma=1.0 -- 1차 시도의 1/3 수준)만
    # 얹어서 노이즈가 이미 걸러진 상태의 미세한 에지(임시표 획)만 살짝 강조한다.
    gray = cv2.bilateralFilter(gray, 9, 20.0, 7.0)
    if is_real_photo:
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv2.addWeighted(gray, 1.15, blurred, -0.15, 0)
    return gray


def _cached_npy(cache_path: str, compute_fn):
    """cache_path가 있으면 로드, 없으면 compute_fn()으로 계산 후 원자적으로 저장."""
    if os.path.isfile(cache_path):
        return np.load(cache_path)
    result = compute_fn()
    tmp_path = f'{cache_path}.tmp{os.getpid()}.npy'
    np.save(tmp_path, result)
    os.replace(tmp_path, cache_path)
    return result


def _read_image_composited(img_path: str) -> np.ndarray:
    """RGBA(투명 배경) PNG를 흰 배경 위에 합성해서 BGR로 반환. MuseScore CLI PNG export는
    기본적으로 투명 배경 + alpha에 실제 잉크가 들어있는 방식이라, 3채널로만 읽으면
    RGB가 전부 (0,0,0)인 새까만 이미지가 된다."""
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        bgr   = im[:, :, :3].astype(np.float32)
        alpha = im[:, :, 3:4].astype(np.float32) / 255.0
        bgr   = bgr * alpha + 255.0 * (1.0 - alpha)
        return bgr.astype(np.uint8)
    return im


def load_preprocessed(img_path: str) -> np.ndarray:
    """이미지 경로 기준 preprocess() 결과를 캐싱 (Phase1/Phase2 공유, 에폭 간 재사용)."""
    cache_path = os.path.splitext(img_path)[0] + '_pre.npy'
    is_real_photo = img_path.lower().endswith(('.jpg', '.jpeg'))
    return _cached_npy(cache_path, lambda: preprocess(_read_image_composited(img_path), is_real_photo))


def load_staffs_cached(img_path: str, gray: np.ndarray) -> List[Dict]:
    """detect_staffs() 결과를 캐싱. OMRDataset.__init__이 매번 재생성될 때마다(에폭마다가
    아니라 학습 재시작마다) 16000장 전체를 다시 스캔하는 비용을 없앤다.

    2026-08-02: detect_staffs()(직선 전용) 단독으로는 사용자가 대보표/단일오선 하나만
    남게 편집한 실사 사진(가로세로 비율 7~8:1의 초박형 스트립, 합성 렌더링의 1.5~3:1과
    판이하게 다름)에서 검출 실패율이 높았음(40장 표본 중 5장 완전실패). detect_staffs_curved
    (곡률 대응, 원본 이미지 그대로 사용 -- correct_perspective를 타지 않아 좌표계가
    gray와 어긋날 위험이 없음)를 폴백으로 추가해 40장 중 완전실패 0건, 정상범위(1~2개)
    검출 비율 67.5%->80%로 개선 확인(verify_staff_detection_fix.py). best_effort_staff_
    detection()은 보정된 이미지 기준 좌표를 반환할 수 있어 원본 gray와 좌표계가 어긋날
    위험이 있으므로 의도적으로 안 씀 -- 원본 이미지에서만 검출 방식 두 가지를 비교."""
    cache_path = os.path.splitext(img_path)[0] + '_staffs.json'
    if os.path.isfile(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            return json.load(f)
    staffs_straight = detect_staffs(gray)
    staffs_curved = detect_staffs_curved(gray)
    staffs = staffs_curved if len(staffs_curved) > len(staffs_straight) else staffs_straight
    tmp_path = f'{cache_path}.tmp{os.getpid()}'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(staffs, f)
    os.replace(tmp_path, cache_path)
    return staffs


# ─────────────────────────────────────────────────────────────────────────────
#  원근/기울기 보정 (ml/omr/engine/src/perspective_corrector.cpp 포팅)
#
#  detect_staffs()는 수평 투영 프로파일 기반이라 회전에 전혀 강건하지 않다(2026-07-24,
#  L4 노이즈 99장 테스트에서 60.6% 오선 검출 실패로 확인). C++ 엔진에는 이미 이 문제에
#  대응하는 2단계 보정 로직이 있었지만 Python 학습/평가 파이프라인에는 연결돼 있지
#  않았다 -- 그걸 그대로 포팅.
# ─────────────────────────────────────────────────────────────────────────────

_MIN_PAGE_AREA_RATIO  = 0.20  # 후보 페이지 quad가 프레임 대비 차지해야 할 최소 면적 비율
_MAX_DESKEW_ANGLE_DEG = 15.0  # 2단계 deskew가 보정하는 최대 각도(그 이상은 카메라 기울기로 보기 어려움)
_MIN_LINE_FRAC        = 0.30  # HoughLinesP가 인정하는 최소 정규화 직선 길이


def _order_points(pts: np.ndarray) -> np.ndarray:
    """4점을 [TL, TR, BR, BL] 순서로 정렬."""
    pts = sorted(pts.tolist(), key=lambda p: p[0] + p[1])
    tl, br = pts[0], pts[3]
    mid1, mid2 = pts[1], pts[2]
    if (mid1[0] - mid1[1]) > (mid2[0] - mid2[1]):
        tr, bl = mid1, mid2
    else:
        tr, bl = mid2, mid1
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_page_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    """페이지 경계를 4각형으로 검출. 실패 시 None."""
    scale = min(1.0, 640.0 / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    inv = cv2.bitwise_not(blurred)
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kside = max(11, int(small.shape[1] * 0.03)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kside, kside))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    img_area = small.shape[0] * small.shape[1]
    if cv2.contourArea(largest) / img_area < _MIN_PAGE_AREA_RATIO:
        return None
    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps_f in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        cand = cv2.approxPolyDP(hull, eps_f * peri, True)
        if len(cand) == 4:
            approx = cand
            break
    if approx is None:
        return None
    pts = approx.reshape(-1, 2).astype(np.float32) / scale

    # 면적비만 보면 "가로는 꽉 차지만 세로는 페이지의 절반만 덮는" 조각 윤곽(예: 위쪽 시스템
    # 하나만 감싸는 잘못된 사각형)도 통과해버린다(2026-07-26 실측: designKit 신선한 로컬
    # 렌더에서 대보표 2단 중 아래쪽 절반만 감싸는 quad가 검출돼 워프 후 오선 재검출이
    # 완전히 틀어짐). 실제 페이지 사진이라면 종이가 프레임의 가로/세로 모두 상당 부분을
    # 차지해야 정상이므로, 가로/세로 각각의 폭도 최소 비율을 넘는지 별도로 확인.
    H, W = gray.shape
    x_span = pts[:, 0].max() - pts[:, 0].min()
    y_span = pts[:, 1].max() - pts[:, 1].min()
    if x_span / W < 0.5 or y_span / H < 0.5:
        return None

    return _order_points(pts)


def _warp_quad(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    w_top, w_bot   = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    h_left, h_right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    dst_w = int(round(max(w_top, w_bot)))
    dst_h = int(round(max(h_left, h_right)))
    if dst_w < 16 or dst_h < 16:
        return gray
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(gray, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _deskew(gray: np.ndarray) -> np.ndarray:
    scale = min(1.0, 1024.0 / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    edges = cv2.Canny(small, 50, 150)
    W = small.shape[1]
    lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180.0, threshold=60,
                            minLineLength=W * _MIN_LINE_FRAC, maxLineGap=W * 0.05)
    if lines is None:
        return gray
    angles = []
    # cv2 버전에 따라 HoughLinesP 반환 shape가 (N,1,4) 또는 (N,4)로 달라서(로컬 4.13.0 vs
    # pod 5.0.0에서 실제로 크래시 확인, 2026-07-25) reshape로 통일 후 순회.
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < 1.0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) <= _MAX_DESKEW_ANGLE_DEG:
            angles.append(ang)
    if not angles:
        return gray
    angles.sort()
    median_angle = angles[len(angles) // 2]
    if abs(median_angle) < 0.1:
        return gray
    H, W2 = gray.shape
    R = cv2.getRotationMatrix2D((W2 / 2.0, H / 2.0), median_angle, 1.0)
    return cv2.warpAffine(gray, R, (W2, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def correct_perspective(gray: np.ndarray) -> np.ndarray:
    """1단계: 페이지 4꼭짓점 검출 후 warpPerspective. 실패 시 2단계: Hough 기반 deskew.
    detect_staffs() 호출 전에 적용한다."""
    corners = _detect_page_quad(gray)
    if corners is not None:
        return _warp_quad(gray, corners)
    return _deskew(gray)


def _peaks_from_binary(binary: np.ndarray, h_kernel_width: int) -> List[float]:
    """이진화된 이미지(또는 스트립)에서 수평 스트로크 피크 y좌표들을 찾는다.
    detect_staffs()/detect_staffs_curved() 공용."""
    H = binary.shape[0]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, h_kernel_width), 1))
    sb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    proj = sb.mean(axis=1) / 255.0
    proj = np.convolve(proj, np.ones(5) / 5, mode='same')  # 두꺼운 오선 내부 잡음 피크 스무딩
    THRESH = 0.05
    candidates = [(float(proj[i]), i) for i in range(1, H - 1)
                  if proj[i] > THRESH and proj[i] >= proj[i-1] and proj[i] >= proj[i+1]]
    candidates.sort(reverse=True)
    suppressed = np.zeros(H, dtype=bool)
    peaks = []
    sup_radius = max(MIN_UNIT / 2, 8.0)  # 두꺼운 오선 렌더링에서 생기는 근접 잡음 피크 흡수
    for val, row in candidates:
        if suppressed[row]: continue
        peaks.append(row)
        lo = max(0, int(row - sup_radius))
        hi = min(H - 1, int(row + sup_radius))
        suppressed[lo:hi+1] = True
    peaks.sort()
    return peaks


def _group_peaks_into_staffs(peaks: List[float]) -> List[Dict]:
    """정렬된 y 피크들을 5개씩 묶어 오선 그룹으로 변환(간격 일관성 체크).
    detect_staffs()/detect_staffs_curved() 공용."""
    if len(peaks) < 5:
        return []
    staffs = []
    i = 0
    while i + 4 < len(peaks):
        gaps = [peaks[i+k+1] - peaks[i+k] for k in range(4)]
        if all(MIN_UNIT <= g <= MAX_UNIT for g in gaps):
            unit   = sum(gaps) / 4.0
            stddev = (sum((g - unit)**2 for g in gaps) / 4.0) ** 0.5
            if stddev <= 0.6 * unit:
                staffs.append({'y_lines': [float(peaks[i+k]) for k in range(5)],
                                'unit_size': unit})
                i += 5
                continue
        i += 1
    return staffs


def detect_staffs(gray: np.ndarray) -> List[Dict]:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    W = binary.shape[1]
    peaks = _peaks_from_binary(binary, W // 8)
    return _group_peaks_into_staffs(peaks)


def detect_staffs_curved(gray: np.ndarray, strip_width: int = 150,
                         link_tol_px: float = 20.0) -> List[Dict]:
    """곡률(종이 휘어짐)에 강인한 오선 검출. detect_staffs()는 전체 폭에 걸친 단일 수평
    투영이라 조금만 휘어도 피크가 여러 행에 걸쳐 스미어져 실패한다(2026-07-25 실측:
    15px 휘어짐만으로 성공률 92%->22%로 붕괴, correct_perspective/dewarp_page도 못 살림).

    이미지를 좁은 세로 스트립으로 나눠 각각 독립적으로 오선 후보를 찾는다(국소적으로는
    곡률이 무시할 만큼 작아서 스트립 안에서는 detect_staffs와 같은 수평 투영이 여전히
    유효).

    2026-08-03 재설계: 기존에는 "낱개 줄"을 스트립 간에 개별 추적했는데, 오선 간격이
    좁은 사진(대보표 10줄이 300px대 높이에 들어간 경우 등)에서 약간의 촬영 기울기만
    있어도 인접한 다른 줄로 잘못 이어붙는 문제를 확인(newage15 held-out 실측, 치보표는
    살아남고 베이스보표 그룹이 못 만들어짐). 대신 스트립 안에서 먼저
    `_group_peaks_into_staffs()`로 5줄 그룹을 통째로 완성한 뒤(그룹 형태 자체가 이미
    간격 일관성 검증을 통과했으므로 줄끼리 섞일 여지가 없음), "그룹의 중앙값 위치"만
    스트립 간에 이어붙인다. 충분히 길게 이어진 그룹 트랙(스트립 3개 이상)만 신뢰해서
    줄 위치를 평균한다."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    H, W = binary.shape
    kernel_w = max(3, int(strip_width * 0.6))
    n_strips = max(1, W // strip_width)

    # 스트립 하나의 폭(150px) 안에서도 음표/빔이 오선 5줄 중 일부를 가려 그 스트립만
    # 그룹 형성에 실패하는 경우가 흔하다(2026-08-03, newage15 실측: 베이스보표가 12개
    # 스트립 중 4개에서만 그룹을 완성, 나머지는 음표에 가려 실패 -- 바로 인접한 스트립만
    # 잇는 방식이면 이 정도 공백에 트랙이 계속 끊겨 min_track_len을 못 채움). 마지막으로
    # 그룹이 보인 스트립부터 MAX_GAP_STRIPS 이내면 같은 트랙으로 계속 이어붙인다(트랙
    # 정체성은 그룹의 중앙값 근접도로만 판단하므로, 낱개 줄처럼 다른 줄로 잘못 이어붙는
    # 문제는 없음 -- 애초에 그룹 자체가 5줄 간격 일관성 검증을 통과한 것만 후보이기 때문).
    MAX_GAP_STRIPS = 8

    tracks: List[List[Dict]] = []
    last_median: List[float] = []
    last_seen: List[int] = []

    for s in range(n_strips):
        x0 = s * strip_width
        x1 = W if s == n_strips - 1 else (s + 1) * strip_width
        peaks = _peaks_from_binary(binary[:, x0:x1], kernel_w)
        groups = _group_peaks_into_staffs(peaks)

        matched = set()
        for g in groups:
            med = float(np.median(g['y_lines']))
            best_ti, best_dist, best_tol = None, None, None
            for ti in range(len(tracks)):
                gap = s - last_seen[ti]
                if ti in matched or gap > MAX_GAP_STRIPS:
                    continue
                # 스트립을 건너뛴 만큼 누적 기울기 드리프트도 커지므로, 건너뛴 스트립
                # 수에 비례해 허용 오차를 넓힌다(2026-08-03) -- 인접 스트립(gap=1)은
                # 기존 link_tol_px 그대로, 스트립을 건너뛸수록 완만하게 완화.
                tol = link_tol_px + 15.0 * max(0, gap - 1)
                d = abs(med - last_median[ti])
                if d <= tol and (best_dist is None or d < best_dist):
                    best_ti, best_dist, best_tol = ti, d, tol
            if best_ti is not None:
                tracks[best_ti].append(g)
                last_median[best_ti] = med
                last_seen[best_ti] = s
                matched.add(best_ti)
            else:
                tracks.append([g])
                last_median.append(med)
                last_seen.append(s)
                matched.add(len(tracks) - 1)

    def _finalize(min_track_len: int) -> List[Dict]:
        good_tracks = [t for t in tracks if len(t) >= min_track_len]
        if not good_tracks:
            return []
        result = [
            {'y_lines': [float(np.mean([g['y_lines'][k] for g in t])) for k in range(5)],
             'unit_size': float(np.mean([g['unit_size'] for g in t])),
             '_n': len(t)}
            for t in good_tracks
        ]
        result.sort(key=lambda s: s['y_lines'][0])

        # 드리프트가 스트립 경계에서 허용오차(tol)를 살짝 넘기면 같은 물리적 오선이 트랙
        # 둘로 쪼개져 y범위가 겹치는 중복 그룹이 나올 수 있다(2026-08-03, newage02 실측:
        # 21px 점프가 tol=20px를 1px 초과해 치보표가 84-153/100-173 두 그룹으로 분열).
        # y범위가 겹치면 같은 오선의 분열이므로, 더 많은 스트립에서 이어진(더 신뢰할 수
        # 있는) 쪽만 남긴다.
        merged: List[Dict] = []
        for r in result:
            if merged and r['y_lines'][0] < merged[-1]['y_lines'][-1]:
                if r['_n'] > merged[-1]['_n']:
                    merged[-1] = r
                continue
            merged.append(r)
        for r in merged:
            del r['_n']
        return merged

    # 실제 오선 폭이 페이지 폭의 절반에 못 미치는 경우가 흔함(마디 수가 적은 시스템 등) --
    # n_strips에 비례한 최소 길이를 쓰면 그런 짧은 오선을 통째로 걸러내 버린다(2026-07-25
    # 실측: num3500004 등에서 두번째 시스템이 x=0-600에만 잉크가 있어 12개 스트립 중
    # 4개만 커버, n_strips//2=6 기준을 못 넘겨 탈락). 고정값 3을 기본으로 쓴다.
    #
    # 2026-08-03: 온음표/2분음표(속이 빈 노트헤드)가 오선 5줄 중 하나 이상을 자주 가려
    # 베이스보표가 전체 12개 스트립 중 단 2개에서만 5줄을 완전히 만족하는 사례 확인
    # (newage19) -- 그런데 min_track_len을 3->2로 전역으로 낮추면, 다른 사진(newage20)의
    # 화음/빔이 밀집된 구간에서 우연히 2개 스트립만 근접한 노이즈 그룹이 여럿(최대 8개)
    # 살아남는 회귀가 발생함을 확인. 따라서 기본은 엄격한 3을 유지하고, 그 결과 그룹이
    # 2개 미만(대보표를 못 이룸)일 때만 완화된 2로 재시도한다 -- 이미 정상 검출되는
    # 절대다수의 사진은 엄격한 기준 그대로라 노이즈 그룹이 생길 여지가 없고, 진짜로
    # 부족한 사진만 구제한다.
    strict = _finalize(3)
    if len(strict) >= 2:
        return strict
    relaxed = _finalize(2)
    return relaxed if len(relaxed) > len(strict) else strict


# ─────────────────────────────────────────────────────────────────────────────
#  페이지 곡률(종이 휘어짐) 보정 (ml/omr/engine/src/page_dewarper.cpp 포팅)
#
#  C++판은 segnet의 픽셀 단위 오선 확률 마스크를 쓰지만, 이 Python 파이프라인엔 segnet이
#  연결돼 있지 않다(2026-07-25 확인, detect_staffs도 순수 고전 CV). detect_staffs가 이미
#  쓰는 수평 모폴로지 오프닝 결과를 마스크 대체로 재사용한다 -- correct_perspective(강체
#  회전/원근 보정)로는 못 잡는 비선형 왜곡(원통형 휘어짐, 손으로 들어 생기는 굴곡)에 대응하는
#  2차 보정. correct_perspective 다음 단계로 쓴다.
# ─────────────────────────────────────────────────────────────────────────────

_DEWARP_TRACE_STEP_PX = 16
_DEWARP_SEARCH_HALF   = 0.65
_DEWARP_MIN_MASK_VAL  = 0.05
_DEWARP_MIN_SAMPLES   = 8
_DEWARP_MARGIN_UNITS  = 2.0


def _build_staff_mask(gray: np.ndarray) -> np.ndarray:
    """detect_staffs와 동일한 수평 스트로크 강조 방식으로 근사 오선 마스크(0~1) 생성."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    W = binary.shape[1]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 8, 1))
    sb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    return sb.astype(np.float32) / 255.0


def _trace_staff_lines(mask: np.ndarray, staff: Dict) -> List[Tuple[int, List[float]]]:
    """열(column) 단위로 오선 5개의 실제 y위치를 추적. mask 값이 너무 낮으면(오선 흔적 없음)
    그 열은 버린다."""
    H = mask.shape[0]
    y_lines = staff['y_lines']
    half = staff['unit_size'] * _DEWARP_SEARCH_HALF
    samples = []
    for x in range(0, mask.shape[1], _DEWARP_TRACE_STEP_PX):
        ys = []
        valid = True
        for line in range(5):
            y_min = max(0, int(y_lines[line] - half))
            y_max = min(H - 1, int(y_lines[line] + half))
            col = mask[y_min:y_max + 1, x]
            if col.size == 0:
                valid = False
                break
            best_i = int(np.argmax(col))
            if col[best_i] < _DEWARP_MIN_MASK_VAL:
                valid = False
                break
            ys.append(float(y_min + best_i))
        if valid:
            samples.append((x, ys))
    return samples


def dewarp_page(gray: np.ndarray, staffs: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """오선별로 실제 곡선을 추적해서 이상적인(등간격) 위치로 펴는 열 단위 rubber-sheet warp.
    추적 샘플이 부족한 오선은 그대로 둔다. 보정된 오선이 하나도 없으면 원본 반환."""
    if not staffs:
        return gray, staffs

    H, W = gray.shape
    mask = _build_staff_mask(gray)
    map_x, map_y = np.meshgrid(np.arange(W, dtype=np.float32),
                               np.arange(H, dtype=np.float32))
    map_y = map_y.copy()

    any_corrected = False
    new_staffs = []
    for staff in staffs:
        samples = _trace_staff_lines(mask, staff)
        if len(samples) < _DEWARP_MIN_SAMPLES:
            new_staffs.append(staff)
            continue

        y_lines = staff['y_lines']
        unit = staff['unit_size']
        y_ideal = [y_lines[0] + i * unit for i in range(5)]
        y_ideal_arr = np.array(y_ideal, dtype=np.float32)
        margin = unit * _DEWARP_MARGIN_UNITS
        y_top = max(0, int(y_ideal[0] - margin))
        y_bot = min(H - 1, int(y_ideal[4] + margin))
        y_dst = np.arange(y_top, y_bot + 1, dtype=np.float32)

        xs = [s[0] for s in samples]
        x_lo, x_hi = xs[0], xs[-1]
        lo_idx = 0
        for x in range(x_lo, x_hi + 1):
            while lo_idx + 1 < len(samples) - 1 and samples[lo_idx + 1][0] <= x:
                lo_idx += 1
            hi_idx = lo_idx + 1
            if hi_idx >= len(samples):
                break
            dx = samples[hi_idx][0] - samples[lo_idx][0]
            t = 0.0 if dx <= 0 else (x - samples[lo_idx][0]) / dx
            yt = np.array([samples[lo_idx][1][i] * (1 - t) + samples[hi_idx][1][i] * t
                           for i in range(5)], dtype=np.float32)

            src_y = np.empty_like(y_dst)
            above = y_dst <= y_ideal[0]
            below = y_dst >= y_ideal[4]
            between = ~above & ~below

            denom0 = max(1.0, y_ideal[1] - y_ideal[0])
            slope0 = (yt[1] - yt[0]) / denom0
            src_y[above] = yt[0] + slope0 * (y_dst[above] - y_ideal[0])

            denom4 = max(1.0, y_ideal[4] - y_ideal[3])
            slope4 = (yt[4] - yt[3]) / denom4
            src_y[below] = yt[4] + slope4 * (y_dst[below] - y_ideal[4])

            if np.any(between):
                idx = np.clip(np.searchsorted(y_ideal_arr, y_dst[between], side='right') - 1, 0, 3)
                denom = np.maximum(1.0, y_ideal_arr[idx + 1] - y_ideal_arr[idx])
                alpha = (y_dst[between] - y_ideal_arr[idx]) / denom
                src_y[between] = yt[idx] + alpha * (yt[idx + 1] - yt[idx])

            src_y = np.clip(src_y, 0, H - 1)
            map_y[y_top:y_bot + 1, x] = src_y

        flat_staff = dict(staff)
        flat_staff['y_lines'] = [y_lines[0] + i * unit for i in range(5)]
        new_staffs.append(flat_staff)
        any_corrected = True

    if not any_corrected:
        return gray, staffs

    dewarped = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return dewarped, new_staffs


def _content_x_range(strip: np.ndarray, margin_px: int = 20) -> tuple:
    """스트립에서 실제 콘텐츠(오선+기호)가 있는 가로 범위를 찾는다.
    페이지 폭을 그대로 쓰면 마디 수가 적은 이미지(예: 고립 음표 1개)는
    캔버스 대부분이 빈 배경으로 낭비되어, 마디 수가 많은 데이터와 콘텐츠
    밀도가 크게 달라지는 문제가 생긴다 -- 콘텐츠 경계 기준으로 크롭해 완화."""
    binary = strip < 250
    cols = np.where(binary.any(axis=0))[0]
    if len(cols) == 0:
        return 0, strip.shape[1]
    x_min = max(0, int(cols[0]) - margin_px)
    x_max = min(strip.shape[1], int(cols[-1]) + margin_px)
    if x_max <= x_min:
        return 0, strip.shape[1]
    return x_min, x_max


def _content_y_extent(gray: np.ndarray, edge: int, direction: int,
                       base_margin_px: float, cap_px: float) -> int:
    """staff 위/아래로 base_margin_px만큼은 항상 포함하고, 그 밖에도 잉크(옥타브
    브래킷+텍스트처럼 MARGIN_UNITS보다 멀리 그려지는 기호)가 이어지면 cap_px까지
    마진을 늘린다. 점선(대시) 사이 공백에 대응하려면 짧은 공백은 무시하고
    gap_tol_px 이상 잉크가 없을 때만 콘텐츠가 끝났다고 판단한다."""
    H = gray.shape[0]
    gap_tol_px = 30  # 옥타브(ottava) 점선의 대시 간 공백을 건너뛰기 충분한 크기
    extent = base_margin_px
    blank_run = 0
    y = edge + direction * int(base_margin_px)
    step = int(base_margin_px)
    while step < cap_px:
        if y < 0 or y >= H:
            break
        if bool((gray[y, :] < 220).any()):
            extent = step
            blank_run = 0
        else:
            blank_run += 1
            if blank_run > gap_tol_px:
                break
        y += direction
        step += 1
    return int(extent) + 4  # 소량 여유


def _staff_y_bounds(gray: np.ndarray, staff: Dict,
                     hard_top: Optional[int] = None, hard_bot: Optional[int] = None) -> tuple:
    """오선의 y_top/y_bot 크롭 경계. MARGIN_UNITS를 기본으로 하되, 그 밖에
    콘텐츠(옥타브 브래킷, 셋잇단음표 숫자 등)가 이어지면 MARGIN_UNITS_CAP까지 확장한다.
    hard_top/hard_bot이 주어지면(대보표에서 인접 오선까지의 실제 거리) 그 경계를
    넘지 않도록 강제한다 -- 안 그러면 치/베이스 간격이 좁을 때 옆 오선 내용까지
    끌어와서(2026-07-21 확인: 셋잇단음표 진단 중 발견) 실제 콘텐츠가 희석된다."""
    H = gray.shape[0]
    base = staff['unit_size'] * MARGIN_UNITS
    cap  = staff['unit_size'] * MARGIN_UNITS_CAP
    top_edge = int(staff['y_lines'][0])
    bot_edge = int(staff['y_lines'][4])
    top_ext = _content_y_extent(gray, top_edge, -1, base, cap)
    bot_ext = _content_y_extent(gray, bot_edge, +1, base, cap)
    y_top = max(0, top_edge - top_ext)
    y_bot = min(H - 1, bot_edge + bot_ext)
    if hard_top is not None:
        y_top = max(y_top, hard_top)
    if hard_bot is not None:
        y_bot = min(y_bot, hard_bot)
    return y_top, y_bot


def extract_staff_canvas(gray: np.ndarray, staff: Dict) -> np.ndarray:
    H, W = gray.shape
    y_top, y_bot = _staff_y_bounds(gray, staff)
    strip  = gray[y_top:y_bot+1, :]
    if strip.size == 0:
        return np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    x_min, x_max = _content_x_range(strip)
    strip = strip[:, x_min:x_max]

    # 2026-08-03: 스케일을 크롭 전체 높이(sh, 레저선/임시표 등 콘텐츠 양에 따라
    # 마디마다 달라짐) 기준이 아니라 오선 자체의 unit_size(줄 간격) 기준 고정값으로
    # 계산한다 -- CANVAS_H/MARGIN_UNITS 설계 의도(주석 "4+2*3=10 units")대로 오선이
    # 항상 STAFF_UNIT_PX(=CANVAS_H/10)로 스케일되게 하고, 오선 첫 줄을 캔버스의 고정
    # 위치(STAFF_ANCHOR_TOP_PX)에 앵커링한다. 기존 방식(크롭 전체를 캔버스에 맞춰
    # 스케일+중앙정렬)은 부수 콘텐츠(레저선·임시표·핑거링 숫자 등) 양에 따라 크롭
    # 높이가 들쭉날쭉해져서, 같은 음이 마디마다 캔버스 안 다른 절대 위치/크기로
    # 나타났다(실측: 오선 상단 위치가 56~89px로 오선 한 줄 간격의 약 2배만큼 편차) --
    # pool_h=1 구조에서 위치->음이름 매핑을 안정적으로 학습하기 어려웠던 근본 원인으로
    # 추정, held-out 3도(선/칸) 오독의 대다수를 차지하는 문제. 폭이 STAFF_UNIT_PX
    # 기준으로 CANVAS_W를 넘는 드문 경우(마디당 음표가 매우 많음)에만 추가로 축소
    # (이 경우에 한해 오선 위치 고정이 깨짐 -- 흔치 않은 예외로 간주).
    sh, sw = strip.shape
    scale = STAFF_UNIT_PX / max(staff['unit_size'], 1e-6)
    if sw * scale > CANVAS_W:
        scale = CANVAS_W / sw
    new_w = max(1, int(round(sw * scale)))
    new_h = max(1, int(round(sh * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(strip, (new_w, new_h), interpolation=interp)

    # 오선 첫 줄(스케일 전 크롭 안에서 top_edge - y_top 위치)이 스케일 후 캔버스의
    # 고정 위치(STAFF_ANCHOR_TOP_PX)에 오도록 배치 -- 중앙정렬이 아니라 오선 기준 고정
    # 앵커. 가로는 기존처럼 중앙정렬 유지(회전 augmentation 시 잘림 방지, 2026-07-28).
    staff_top_in_strip = int(staff['y_lines'][0]) - y_top
    staff_top_scaled = int(round(staff_top_in_strip * scale))
    oy = STAFF_ANCHOR_TOP_PX - staff_top_scaled

    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    ox = (CANVAS_W - new_w) // 2
    src_y0 = max(0, -oy)
    dst_y0 = max(0, oy)
    copy_h = min(new_h - src_y0, CANVAS_H - dst_y0)
    if copy_h > 0:
        canvas[dst_y0:dst_y0 + copy_h, ox:ox + new_w] = resized[src_y0:src_y0 + copy_h]
    return canvas


def extract_system_canvas(gray: np.ndarray, staffs: List[Dict]) -> np.ndarray:
    """
    Grand staff: staffs[0](treble) + staffs[1](bass) 영역을 수직으로 이어붙여
    SYSTEM_CANVAS_H × CANVAS_W 캔버스를 반환한다.
    모델이 학습된 방식과 동일 (ml/omr/training/dataset.py 기준).
    두 오선의 가로 콘텐츠 범위를 공유해서 크롭한다 (같은 시간축이므로
    treble/bass가 서로 다른 가로 위치로 어긋나지 않도록).
    """
    H, W   = gray.shape
    half_h = SYSTEM_CANVAS_H // 2  # 192

    # treble(staffs[0])의 아래쪽/bass(staffs[1])의 위쪽 콘텐츠 기반 마진이 서로의
    # 오선까지 침범하지 않도록, 중간 지점을 강제 상한으로 둔다.
    mid = (int(staffs[0]['y_lines'][4]) + int(staffs[1]['y_lines'][0])) // 2

    def _raw_strip(staff: Dict, hard_top: Optional[int] = None, hard_bot: Optional[int] = None) -> tuple:
        y_top, y_bot = _staff_y_bounds(gray, staff, hard_top=hard_top, hard_bot=hard_bot)
        strip = gray[y_top:y_bot + 1, :]
        staff_top_in_strip = int(staff['y_lines'][0]) - y_top
        return strip, staff_top_in_strip

    strip0, top0 = _raw_strip(staffs[0], hard_bot=mid)
    strip1, top1 = _raw_strip(staffs[1], hard_top=mid)
    x0min, x0max = _content_x_range(strip0) if strip0.size else (0, W)
    x1min, x1max = _content_x_range(strip1) if strip1.size else (0, W)
    x_min, x_max = min(x0min, x1min), max(x0max, x1max)
    if x_max <= x_min:
        x_min, x_max = 0, W

    s0 = strip0[:, x_min:x_max] if strip0.size else strip0
    s1 = strip1[:, x_min:x_max] if strip1.size else strip1

    # 2026-08-03: 스케일을 콘텐츠 높이(s0/s1.shape[0], 마디마다 들쭉날쭉) 기준이 아니라
    # 오선 unit_size 기준 고정값(SYSTEM_STAFF_UNIT_PX)으로 계산 -- extract_staff_canvas와
    # 동일한 이유(3도 오독 근본원인 대응). treble/bass unit_size가 약간 다를 수 있어
    # (실측 1~3% 차이) 더 작은 쪽 기준으로 잡아 두 성부 모두 half_h 안에 들어오도록
    # 보수적으로 계산 -- 가로 스케일은 여전히 공유(시간축 정렬 유지, 기존 이유 그대로).
    unit = min(staffs[0]['unit_size'], staffs[1]['unit_size'])
    scale = SYSTEM_STAFF_UNIT_PX / max(unit, 1e-6)
    # 음표 밀도가 높아 unit_size 기준 스케일로도 CANVAS_W를 넘으면(예: 한 마디에 8분음표
    # 8개) 추가로 축소 -- 공통 스케일이므로 treble/bass에 동일하게 적용된다(이 경우에
    # 한해 오선 위치 고정이 깨짐 -- 흔치 않은 예외).
    max_scaled_w = max(s0.shape[1], s1.shape[1]) * scale
    if max_scaled_w > CANVAS_W:
        scale *= CANVAS_W / max_scaled_w

    def _resize(s: np.ndarray, staff_top_in_strip: int) -> np.ndarray:
        new_w  = max(1, int(round(s.shape[1] * scale)))
        new_h  = max(1, int(round(s.shape[0] * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(s, (new_w, new_h), interpolation=interp)
        tile = np.full((half_h, new_w), 255, dtype=np.uint8)
        # 오선 첫 줄이 half-tile 안의 고정 위치(SYSTEM_STAFF_ANCHOR_TOP_PX)에 오도록
        # 배치(중앙정렬 대신 오선 기준 고정 앵커, extract_staff_canvas와 동일 방식).
        staff_top_scaled = int(round(staff_top_in_strip * scale))
        oy = SYSTEM_STAFF_ANCHOR_TOP_PX - staff_top_scaled
        src_y0 = max(0, -oy)
        dst_y0 = max(0, oy)
        copy_h = min(new_h - src_y0, half_h - dst_y0)
        if copy_h > 0:
            tile[dst_y0:dst_y0 + copy_h, :] = resized[src_y0:src_y0 + copy_h]
        return tile

    # treble/bass 모두 같은 가로 오프셋으로 중앙 정렬해야 시간축 정렬이 유지된다(각자
    # 다른 오프셋을 쓰면 같은 시점의 마디가 서로 다른 x에 놓임). 왼쪽에 붙이면(구 버전)
    # 왼쪽 여백이 0이 되어 캔버스 레벨 회전 시 왼쪽 내용이 잘리는 문제도 있었다
    # (2026-07-28 사용자 피드백: "악보가 너무 많이 잘림").
    r0 = _resize(s0, top0)
    r1 = _resize(s1, top1)  # r0.shape[1] == r1.shape[1] (공통 scale이므로) -- 항상 성립
    ox = (CANVAS_W - max(r0.shape[1], r1.shape[1])) // 2
    ox = max(0, ox)

    def _pad(strip: np.ndarray) -> np.ndarray:
        tile = np.full((half_h, CANVAS_W), 255, dtype=np.uint8)
        cw = min(CANVAS_W - ox, strip.shape[1])
        tile[:, ox:ox + cw] = strip[:, :cw]
        return tile

    return np.vstack([_pad(r0), _pad(r1)])


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


def _redetect_with_fallback(noisy: np.ndarray, corrected: np.ndarray,
                             expected_n: int) -> Tuple[List[Dict], np.ndarray]:
    """오선 개수가 이미 알려진 상황(학습 중 재검출)에서 여러 검출 조합을 순서대로 시도해
    기대 개수와 일치하는 첫 결과를 채택한다. 실측(2026-07-25, 회전<=6도+curl 합성):
    correct_perspective 단독 + 전역 투영은 curl 15px에서 성공률이 92%->23%로 붕괴하지만,
    스트립 기반 검출(detect_staffs_curved)을 조합하면 81%까지 회복되고, 여기에 폴백
    체인(보정+스트립 -> 무보정+스트립 -> 보정+전역)까지 더하면 86%까지 오른다.
    dewarp_page는 스트립 검출을 부트스트랩으로 써도 오히려 성공률을 떨어뜨려(재추적 시
    보간 잡음이 이득보다 큼) 이 체인에서 제외했다.

    반환된 staffs의 좌표가 어느 이미지(corrected/noisy) 기준인지가 호출부의 크롭 대상과
    어긋나면 안 되므로, staffs와 함께 그 좌표계의 원본 이미지를 같이 반환한다.

    L3(밝기/블러/jpeg 열화)에서는 위 세 조합도 실패하는 경우가 있는데, CLAHE로 국소
    대비를 복원한 버전을 마지막으로 한 번 더 시도하면 추가로 건진다(2026-07-25 실측:
    60개 샘플 기준 84.4%->90.0%)."""
    cand = detect_staffs_curved(corrected)
    if len(cand) == expected_n:
        return cand, corrected
    cand = detect_staffs_curved(noisy)
    if len(cand) == expected_n:
        return cand, noisy
    cand = detect_staffs(corrected)
    if len(cand) == expected_n:
        return cand, corrected

    corrected_c = _CLAHE.apply(corrected)
    cand = detect_staffs_curved(corrected_c)
    if len(cand) == expected_n:
        return cand, corrected
    noisy_c = _CLAHE.apply(noisy)
    cand = detect_staffs_curved(noisy_c)
    if len(cand) == expected_n:
        return cand, noisy
    return cand, noisy


def best_effort_staff_detection(gray: np.ndarray, use_full_warp: bool = True) -> Tuple[List[Dict], np.ndarray]:
    """실전 추론(inference.py의 run_image())용 오선 검출. 학습 때(_redetect_with_fallback)와
    달리 정답 오선 개수를 모르므로, 여러 조합(보정+전역, 원본+전역, 보정+스트립, 원본+스트립)
    중 가장 많은 오선을 찾은 조합을 채택한다 -- correct_perspective가 오히려 망가뜨리는
    경우(예: 이미 오선 위주로 타이트하게 촬영된 사진에서 페이지 경계 오검출로 워프가
    틀어짐, 2026-07-27 실측: 대보표 2개가 제대로 잡히던 원본이 보정 후 0개로 붕괴)
    원본으로 안전하게 폴백한다.

    오선 개수가 동점이면 항상 원본(gray)을 우선한다(2026-07-28 실측: 이미 평평한 합성
    렌더링에도 correct_perspective가 불필요한 원근 워프를 걸어 페이지 세로 크기를
    565px->339px로, 오선 상단 여백을 94px->4px로 깎아버린 사례 확인 -- 오선 개수는
    보정 전후로 동일(2개)했는데 candidates 순서상 corrected가 항상 먼저 채택되고
    있었음. 이후 캔버스 추출 시 MARGIN_UNITS 여백을 확보 못해 오선이 캔버스 맨 위에
    거의 붙어버림). 동점일 땐 보정에 득이 없으므로 여백이 더 건강한 원본을 쓴다.

    2026-08-02 근본 원인 확정: 넓은 음역(레저선 다수)의 화음이 있는 합성 이미지에서
    correct_perspective가 실제로 무엇을 망가뜨리는지 픽셀 단위로 추적한 결과, 문제는
    "레저선을 오선으로 오검출"이 아니라 _detect_page_quad()가 페이지 경계가 아예 없는
    (이미 시스템 하나만 잘라낸) 이 이미지에서도 면적 조건(x_span/y_span >= 0.5)을 우연히
    만족하는 윤곽을 "페이지"로 오인해 _warp_quad()로 콘텐츠 자체를 압축/왜곡시키는
    것이었음(가짜 오선 5개가 전부 폭 100%를 차지 -- 레저선이 아니라 압축된 콘텐츠가
    만든 허상 줄무늬). 이 앱의 실제 촬영 방식(오버레이로 오선 1~2개 폭만큼만 맞춰
    촬영, project.md/세션 기록)과 학습 시 노이즈 경로(page_noise_and_redetect, 이미
    시스템 하나만 크롭된 입력엔 1단계(페이지 모서리 검출+워프)를 건너뛰고 2단계
    (Hough 기반 _deskew)만 적용)를 감안하면, 추론 경로도 애초에 "페이지 통째" 워프가
    아니라 회전 보정만 필요하다.

    다만 실사 사진 51장으로 재검증한 결과 _deskew만으로 완전히 바꾸면 실사 쪽도
    오히려 나빠짐(원래 76.0% -> 70.0%) -- 실사는 촬영 시 주변 배경/여백이 어느 정도
    같이 잡혀 correct_perspective의 1단계(페이지 검출+워프)가 실제로 도움되는
    경우가 다수 있는 것으로 보임(반면 완전히 깨끗한 합성 렌더링은 그런 경계가 아예
    없어서 오검출만 함). 따라서 use_full_warp로 분기 -- run_image()가 확장자로
    실사(jpg/jpeg)는 True(기존 correct_perspective 그대로), 합성(png)은 False
    (_deskew만)로 넘겨준다."""
    corrected = correct_perspective(gray) if use_full_warp else _deskew(gray)
    candidates = [
        (detect_staffs(gray), gray),
        (detect_staffs(corrected), corrected),
        (detect_staffs_curved(gray), gray),
        (detect_staffs_curved(corrected), corrected),
    ]

    # 2026-08-03: held-out 실사 오류 분석에서, run_image()가 개별 후보 중 오선 개수가
    # 가장 많은 것만 보고 골라 홀수(1개/3개)를 채택하는 사례를 다수 확인 -- 이 프로젝트의
    # 실제 콘텐츠는 전부 대보표(치+베이스 짝, 항상 짝수)이므로 홀수 채택은 곧 오검출이다.
    # 홀수를 채택하면 run_image()가 "대보표" 분기 대신 "개별 오선 순차 처리" 분기로 빠져,
    # 오선마다 헤더(clef/key/time)를 통째로 반복 생성해 이어붙이는 심각한 과잉생성을
    # 일으킴(GT 대비 2~3배 길이). 짝수(>=2) 후보가 하나라도 있으면 그 중에서만 최댓값을
    # 고르고, 짝수 후보가 전혀 없을 때만(=오선 자체를 거의 못 찾은 완전 실패) 기존처럼
    # 전체 후보 중 최댓값(홀수 포함)으로 폴백한다.
    even_candidates = [c for c in candidates if len(c[0]) >= 2 and len(c[0]) % 2 == 0]
    pool = even_candidates if even_candidates else candidates

    # 2026-08-03: "짝수 중 최댓값"만으로 고르면, correct_perspective가 콘텐츠를
    # 압축/왜곡시켜 곡선검출이 스퓨리어스하게 부풀린 개수(예: 정상 2개인데 보정본에서만
    # 8개)를 오히려 우선시하는 회귀를 확인(newage20 실측). 후보 4개 중 다수가 동의하는
    # 개수(최빈값)를 우선하고, 최댓값은 동표일 때만 보조 기준으로 쓴다 -- 여러 검출기가
    # 독립적으로 같은 개수에 도달했다면 그게 실제 오선 개수일 가능성이 높고, 혼자만 크게
    # 다른 값을 내는 후보는 오검출로 부풀려졌을 위험이 크다.
    from collections import Counter
    freq = Counter(len(c[0]) for c in pool)
    best_count = max(freq.items(), key=lambda kv: (kv[1], kv[0]))[0]
    consensus_pool = [c for c in pool if len(c[0]) == best_count]
    best_staffs, best_img = consensus_pool[0]
    return best_staffs, best_img


def page_noise_and_redetect(gray0: np.ndarray, staff, level: int) -> Optional[np.ndarray]:
    """5n5 전용: 실제 추론 경로(노이즈 낀 사진 -> correct_perspective -> 오선 재검출)를
    학습 때도 그대로 거치게 한다 -- 캔버스에 직접 약한 기하 노이즈만 주던 기존 방식은
    "이미 정확히 잘라낸 캔버스"만 봤을 뿐, deskew가 남기는 잔여 정렬 오차·이중 보간
    흐림·재검출 좌표 오차는 한 번도 학습하지 못했다(2026-07-24 100장 테스트에서 확인).

    staff 주변만 넉넉히 크롭해서 노이즈+보정+재검출을 거치므로, 페이지 내 다른 시스템과
    섞이지 않는다(대보표 샘플이 시스템 단위로 저장돼 있어 어느 시스템인지 재추적할 필요가
    없음). 재검출 결과 구조(오선 개수)가 기대와 다르면 None -- 호출부에서 기존 캔버스
    레벨 경로로 폴백해야 한다.
    """
    is_grand = isinstance(staff, list)
    staffs_clean = staff if is_grand else [staff]
    y_top = int(min(s['y_lines'][0] for s in staffs_clean))
    y_bot = int(max(s['y_lines'][4] for s in staffs_clean))
    # 350px 고정 -- 실측(2026-07-24, 100장): 500px 이상으로 늘리면 인접 시스템 오선이
    # 크롭에 섞여 들어와 재검출이 오히려 더 자주 틀어짐(패딩 넓을수록 정확도 하락).
    pad   = 350
    H, W  = gray0.shape
    crop  = gray0[max(0, y_top - pad):min(H, y_bot + pad), :]
    if crop.shape[0] < 50:
        return None

    lvl = NOISE_LEVELS[level]
    # 굴곡을 회전보다 먼저 적용한다 -- 종이 자체의 물리적 휘어짐(굴곡)이 먼저 있고, 그
    # 다음에 카메라가 그 종이를 어떤 각도로 보느냐(회전)가 정해지는 게 실제 순서다.
    # 반대로 하면(회전 먼저) 이미 기울어진 마디선이 굴곡의 열(column) 기준과 어긋나며
    # treble/bass가 서로 다른 굴곡을 받아 위아래 정합이 깨진다(apply_page_curl 주석 참고).
    noisy = crop
    if random.random() < lvl['p_curl']:
        lo, hi = lvl['curl_px']
        if random.random() < lvl.get('p_bump_curl', 0.0):
            # 책 머릿부분(제본선)이 볼록 솟아오른 대칭 굴곡(2026-07-31 추가) -- 기존
            # 램프형(한쪽만 평평, 반대쪽으로 단조증가)과 별개 시나리오라 확률적으로 섞음.
            noisy = apply_page_curl_bump(noisy, random.uniform(lo, hi),
                                          peak_frac=random.uniform(0.4, 0.6), axis='x')
        else:
            noisy = apply_page_curl(noisy, random.uniform(lo, hi),
                                    flat_side=random.choice(['left', 'right']),
                                    direction=random.choice([1.0, -1.0]))
    noisy, _ = geometric_augment(noisy, max_angle_deg=lvl['angle_page'],
                                 persp_margin_frac=lvl['persp_page'],
                                 p_rotate=lvl['p_rotate'], p_persp=lvl['p_persp'])
    noisy = augment_image(noisy, level=level)
    # correct_perspective()의 1단계(페이지 4꼭짓점 검출)는 진짜 촬영된 "전체 페이지"를
    # 가정한다. 여기서는 이미 시스템 하나 주변만 350px 패딩으로 잘라낸 크롭이라 페이지
    # 경계가 프레임 안에 없는데도, 크롭 내부의 큰 윤곽을 페이지로 오인해 잘못된 quad로
    # warp해버리는 경우가 실측됨(2026-07-25: 5도 회전만 있는 샘플이 quad 오검출로 전혀
    # 안 펴진 채 재검출 실패). 크롭 입력에는 처음부터 2단계(Hough 기반 deskew)만 적용.
    corrected = _deskew(noisy)

    expected_n = 2 if is_grand else 1
    staffs_new, src_img = _redetect_with_fallback(noisy, corrected, expected_n)
    if is_grand:
        if len(staffs_new) != 2:
            return None
        return extract_system_canvas(src_img, staffs_new)
    if len(staffs_new) != 1:
        return None
    return extract_staff_canvas(src_img, staffs_new[0])


def generate_weak_seg_labels(gray: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    labels = np.zeros((H, W), dtype=np.int64)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h_len  = max(W // 8, 60)
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    staff  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kern)
    d_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    staff  = cv2.dilate(staff, d_kern)
    labels[staff > 0] = SEG_STAFF_LINE
    symbols = cv2.subtract(binary, staff)
    v_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    stems   = cv2.morphologyEx(symbols, cv2.MORPH_OPEN, v_kern)
    labels[(stems > 0) & (labels == SEG_BG)] = SEG_STEM_REST
    no_stems = cv2.subtract(symbols, stems)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(no_stems, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw   = int(stats[i, cv2.CC_STAT_WIDTH])
        bh   = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bh < 1: continue
        ratio = bw / bh
        if 15 <= area <= 600 and 0.4 <= ratio <= 2.5:
            labels[comp == i] = SEG_NOTEHEAD
    labels[(binary > 0) & (labels == SEG_BG)] = SEG_SYMBOL
    return labels


# 노이즈 커리큘럼 단계별 강도 프리셋. "촬영 시뮬레이션" 목적이므로 4단계 모두 동일한
# 노이즈 종류를 쓰되 강도(확률/범위)만 올라간다. 페이지 전체용(angle_page/persp_page)과
# 폭이 넓어 훨씬 예민한 캔버스 타일용(angle_canvas/persp_canvas)을 분리해서 정의한다
# (SegnetDataset은 페이지, OMRDataset은 캔버스 타일에 기하 왜곡을 적용하기 때문).
# max_concurrent: brightness/gauss_noise/blur/motion_blur/lighting/jpeg 6종 중 한 샘플에
# 동시에 적용될 수 있는 최대 개수. 각 확률을 독립적으로 굴리면 L3/L4에서 6종이 한꺼번에
# 걸려 실제 촬영에서는 나올 수 없을 만큼 정보가 소실된 이미지가 섞여 들어간다(회전+원근처럼
# 실제로 동시에 발생하는 게 자연스러운 기하 왜곡과 달리, 이 6종은 서로 다른 열화 경로라
# 전부 겹칠 이유가 없음) -- augment_image가 이 값으로 동시 적용 개수를 제한한다.
# p_curl/curl_px: 종이 볼록/오목 곡률(apply_page_curl) 시뮬레이션 -- page_noise_and_redetect
# 에서만 사용(캔버스 레벨 augment_image에는 적용하지 않음). 실측(2026-07-25, 회전<=6도 +
# correct_perspective + detect_staffs_curved 폴백체인 기준 100장): curl 15px에서 재검출
# 성공률 86%, 25px 70%, 40px 52% -- 90% 게이트를 안정적으로 넘기는 건 15px 부근까지라
# curl_px 범위를 그 이하로 보수적으로 잡는다(L4도 최댓값 20px로 제한).
# p_real_texture: real_texture_bank/(실사 37장에서 추출한 grain/lighting)를 캔버스에
# 입히는 확률(real_texture_augment.apply_real_texture) -- 2026-07-28 실측(analyze_real_photos.py)
# 결과 기존 합성 밝기/대비/조명범위가 전부 실사와 반대 방향으로 어긋나 있었음이 드러나서
# (밝기 240.8 vs 실측 174.5, 대비 48.0 vs 58.5, 조명범위 31.4 vs 73.5) 추가. 발동 시 아래
# p_brightness/p_lighting 합성 조정은 건너뛴다(실사 텍스처가 이미 둘 다 포함하므로 중복 왜곡 방지).
# angle_canvas/persp_canvas: 2026-07-28 extract_staff_canvas를 비율유지+패딩 방식으로
# 바꾸면서(위 주석 참고) 콘텐츠 오른쪽에 흰 여백이 생겨 회전 시 잘려나갈 위험이 줄었으므로,
# 기존 L3=L4 상한(3.5도)을 풀고 전 레벨 각도/원근을 상향(사용자 피드백: "기울기랑 각도
# 조금 더 강하게").
#
# p_curl/curl_px 2026-07-28 상향: 실사 40장 재비교에서 "종이 굴곡"이 부족하다는 피드백.
# 다만 angle_page/persp_page/p_curl은 page_noise_and_redetect(5n5 전용, --page_level_noise
# 플래그로만 켜짐) 경로에서만 실제로 적용되는데, 정작 최근 두 라운드(chopin_style, noise2)는
# 전부 이 플래그 없이 돌아서 curl이 코드상 존재만 하고 학습에는 한 번도 반영된 적이 없었다
# -- 다음 라운드부터는 반드시 --page_level_noise를 켜야 아래 값이 의미가 있다.
# p_blur/p_motion_blur/downscale 2026-07-28 하향: 실사 블러지표(Laplacian 분산) 중앙값이
# 2318인데 반해 위 그레인 텍스처까지 적용한 합성본이 103으로 20배 이상 더 흐릿했다(육안으로도
# 글자가 잘 안 보이는 수준). 실사는 애초에 그렇게 블러가 심하지 않고 그레인/조명 변화가
# "디테일이 많아 보이는" 느낌을 만드는 것이었는데, blur/motion_blur/jpeg다운스케일이 예전
# (그레인 텍스처 도입 전) 기준으로 세게 잡혀 있어 그 위에 또 뭉갠 것 -- 전 레벨에서 약화.
# used_real_texture일 때는 augment_image()에서 blur/motion_blur를 아예 끄고 jpeg도 압축만
# 남기므로(다운스케일 생략), 아래 값은 주로 real_texture 미발동 케이스에 적용된다.
NOISE_LEVELS = {
    1: dict(angle_page=2.0, persp_page=0.015, angle_canvas=1.5, persp_canvas=0.012,
            p_rotate=0.3, p_persp=0.2, max_concurrent=2,
            p_brightness=0.5, p_gauss_noise=0.3, p_blur=0.12,
            p_motion_blur=0.06, motion_ksize=(3, 5),
            p_lighting=0.2, lighting_strength=(0.05, 0.15),
            p_jpeg=0.15, jpeg_quality=(70, 90), downscale=(0.9, 1.0),
            p_curl=0.15, curl_px=(2, 8), p_real_texture=0.8),
    2: dict(angle_page=4.0, persp_page=0.03, angle_canvas=3.0, persp_canvas=0.02,
            p_rotate=0.5, p_persp=0.4, max_concurrent=2,
            p_brightness=0.6, p_gauss_noise=0.4, p_blur=0.15,
            p_motion_blur=0.1, motion_ksize=(3, 7),
            p_lighting=0.35, lighting_strength=(0.1, 0.25),
            p_jpeg=0.3, jpeg_quality=(50, 80), downscale=(0.85, 1.0),
            p_curl=0.45, curl_px=(10, 20), p_real_texture=0.85),  # 2026-07-31: 4-14->10-20 강화(사용자 확인)
    3: dict(angle_page=7.0, persp_page=0.05, angle_canvas=4.5, persp_canvas=0.032,
            p_rotate=0.6, p_persp=0.5, max_concurrent=3,
            p_brightness=0.7, p_gauss_noise=0.5, p_blur=0.2,
            p_motion_blur=0.15, motion_ksize=(5, 9),
            p_lighting=0.5, lighting_strength=(0.2, 0.4),
            p_jpeg=0.45, jpeg_quality=(30, 65), downscale=(0.75, 0.95),
            p_curl=0.55, curl_px=(6, 18), p_real_texture=0.9),
    4: dict(angle_page=12.0, persp_page=0.08, angle_canvas=6.0, persp_canvas=0.04,
            p_rotate=0.7, p_persp=0.6, max_concurrent=3,
            p_brightness=0.8, p_gauss_noise=0.6, p_blur=0.25,
            p_motion_blur=0.2, motion_ksize=(7, 11),
            p_lighting=0.65, lighting_strength=(0.3, 0.5),
            p_jpeg=0.6, jpeg_quality=(20, 50), downscale=(0.65, 0.85),
            p_curl=0.65, curl_px=(10, 25), p_real_texture=0.95),
}

# p_curl_shade: 실사 재검토(2026-07-28, IMG_8784) 결과 "종이 굴곡"의 실제 지배적인
# 인상은 오선 자체가 휘는 게 아니라 모서리가 살짝 들려서 생기는 부드러운 그림자/
# 하이라이트였다(apply_page_curl의 기하 왜곡은 오히려 회전과 결합했을 때 딱딱한
# 대각선 잘림을 만들어 실사와 다르게 보인다는 피드백). apply_curl_shade는 기하를
# 전혀 건드리지 않는 순수 셰이딩이라 위/아래 정합이 깨질 걱정이 없어 canvas-level
# augment_image()에서 항상 안전하게 쓸 수 있다(page_level_noise 플래그 불필요).
for _lvl_idx, _p in NOISE_LEVELS.items():
    _p['p_curl_shade'] = {1: 0.2, 2: 0.35, 3: 0.5, 4: 0.6}[_lvl_idx]
    # p_bump_curl: 굴곡이 걸렸을 때(p_curl) 그중 이 비율만큼은 기존 램프형 대신 책
    # 제본선이 볼록 솟아오른 대칭 범프형(apply_page_curl_bump)을 씀(2026-07-31 추가,
    # 사용자 요청 -- "책 머릿부분이 볼록 튀어나온" 케이스도 굴곡 종류에 포함).
    _p['p_bump_curl'] = 0.3
    # p_local_blur: 이미지 전체가 아니라 한쪽 구역만 흐릿한 경우(초점/손떨림) --
    # apply_curl_shade와 같은 타원 마스크 구조, L1/L2는 0(끔), L3부터 도입해서 L4에서
    # 강해짐(2026-08-02, "특정 구역이 흐릿함"을 L4 특성으로 명시 요청 반영).
    _p['p_local_blur'] = {1: 0.0, 2: 0.0, 3: 0.15, 4: 0.25}[_lvl_idx]
    # p_unsharp_adapt: 실사 사진 전용 전처리(preprocess()의 언샤프 마스크)와 동일한
    # 스타일을 합성 이미지에도 증강으로 노출(2026-08-02) -- 실사 도입 시 "실사 도메인"과
    # "언샤프 마스크 특유의 질감" 두 가지 새로운 축이 동시에 들어가 적응이 느려지는
    # 문제(r8_1_r9 샤프닝 재학습에서 epoch1 Acc가 기존 대비 더 낮게 시작하는 것으로 확인)
    # 발견 -- 실사 도입 전에 합성만으로 이 질감에 먼저 익숙해지게 하기 위함.
    _p['p_unsharp_adapt'] = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}[_lvl_idx]


def _motion_blur(gray: np.ndarray, ksize_range: Tuple[int, int]) -> np.ndarray:
    lo, hi = ksize_range
    choices = [k for k in range(lo, hi + 1) if k % 2 == 1] or [lo | 1]
    ksize = random.choice(choices)
    angle = random.uniform(0, 180)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    s = kernel.sum()
    kernel /= s if s > 1e-6 else 1.0
    return cv2.filter2D(gray, -1, kernel)


def _uneven_lighting(gray: np.ndarray, strength_range: Tuple[float, float]) -> np.ndarray:
    H, W = gray.shape
    angle    = random.uniform(0, 2 * np.pi)
    strength = random.uniform(*strength_range)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = xx * np.cos(angle) + yy * np.sin(angle)
    grad -= grad.min()
    denom = grad.max()
    if denom > 1e-6:
        grad /= denom
    mask = 1.0 - strength * grad
    return np.clip(gray.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def _jpeg_and_resize_degrade(gray: np.ndarray, quality_range: Tuple[int, int],
                              downscale_range: Tuple[float, float]) -> np.ndarray:
    scale = random.uniform(*downscale_range)
    if scale < 0.999:
        H, W = gray.shape
        small = cv2.resize(gray, (max(1, int(W * scale)), max(1, int(H * scale))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    quality = random.randint(*quality_range)
    ok, enc = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        gray = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return gray


def _apply_brightness(out):
    beta  = random.uniform(-25, 25)
    alpha = random.uniform(0.85, 1.15)
    return np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _apply_gauss_noise(out):
    sigma = random.uniform(2.0, 8.0)
    return np.clip(out.astype(np.float32) + np.random.normal(0, sigma, out.shape),
                   0, 255).astype(np.uint8)


def _apply_blur(out):
    ksize = random.choice([3, 5])
    return cv2.GaussianBlur(out, (ksize, ksize), 0)


def apply_local_blur(gray: np.ndarray, radius_frac: float = 0.35,
                      strength: float = 1.0, aspect_x: float = 1.0,
                      angle_deg: float = 0.0, ksize: int = 9) -> np.ndarray:
    """이미지 전체가 아니라 타원형 영역 하나만 흐릿하게(초점 나간 렌즈/손떨림처럼 한쪽만
    아웃포커스) -- apply_curl_shade와 동일한 타원 마스크(중심/종횡비/기울기 랜덤) 구조를
    그대로 재사용하되, 밝기 대신 블러 강도를 섞는다(2026-08-02, L4 "특정 구역 흐릿함"
    요구사항 반영). strength=1.0이면 영역 중심은 완전히 블러 버전, 가장자리는 원본으로
    부드럽게 전환.
    """
    H, W = gray.shape
    cy = random.uniform(-0.2, 1.2) * H
    cx = random.uniform(-0.2, 1.2) * W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dy = yy - cy
    dx = xx - cx
    if angle_deg != 0.0:
        theta = np.deg2rad(angle_deg)
        dx, dy = (dx * np.cos(theta) + dy * np.sin(theta),
                  -dx * np.sin(theta) + dy * np.cos(theta))
    dist = np.sqrt(dy ** 2 + (dx / max(aspect_x, 1e-6)) ** 2)
    max_dist = radius_frac * np.sqrt(H ** 2 + W ** 2)
    falloff = np.clip(1.0 - dist / max_dist, 0.0, 1.0) ** 2
    alpha = (strength * falloff).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0).astype(np.float32)
    out = gray.astype(np.float32) * (1.0 - alpha) + blurred * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_unsharp_like_real(gray: np.ndarray) -> np.ndarray:
    """preprocess()가 실사 사진에 적용하는 언샤프 마스크와 동일한 파라미터(sigma=1.0,
    1.15/-0.15)를 합성 캔버스에도 적용 -- 실사 도입 전에 합성만으로 이 특유의 질감에
    미리 적응시키기 위한 증강(2026-08-02)."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.15, blurred, -0.15, 0)


def augment_image(gray: np.ndarray, level: int = 2) -> np.ndarray:
    p = NOISE_LEVELS[level]
    out = gray.copy()

    # 실사 텍스처(그레인+조명)를 먼저 입힌다 -- 이미 밝기/대비/조명을 실측값에 맞춰
    # 조정하므로, 발동 시 아래 candidates의 brightness/lighting은 건너뛰어 중복 왜곡을 막는다.
    used_real_texture = random.random() < p.get('p_real_texture', 0.0)
    if used_real_texture:
        out = apply_real_texture(out)

    # 종이 굴곡을 코너 그림자/하이라이트로 표현(기하 왜곡 없음, 위/아래 정합 걱정 없이
    # 항상 안전하게 적용 가능) -- apply_page_curl(기하 remap)보다 이쪽이 실사와 더 가깝다는
    # 피드백(2026-07-28)에 따라 canvas-level에서 기본으로 사용.
    if random.random() < p.get('p_curl_shade', 0.0):
        # 2026-07-31 사용자 확인(비교 이미지 검토) -- 기존 0.18~0.32는 약해 보인다는
        # 피드백으로 데모에서 쓴 강한 쪽(strength~0.65) 기준으로 상향. 오선은 가로로
        # 길어서 그림자도 원형보다 타원형이 더 자연스럽다는 요청으로 aspect_x를
        # 1.0(원형)~2.5(가로로 넓은 타원) 범위에서 무작위 적용. 추가로(2026-07-31):
        # 타원 중심 위치와 장축 기울기(angle_deg)도 매번 다르게 -- 중심은 이미지 안팎
        # (-0.3~1.3 배율)에서 균등 추첨해 코너/가장자리/중앙 어디든 나올 수 있게 함.
        _sh_H, _sh_W = out.shape
        _sh_cy = random.uniform(-0.3, 1.3) * _sh_H
        _sh_cx = random.uniform(-0.3, 1.3) * _sh_W
        out = apply_curl_shade(out, center=(_sh_cy, _sh_cx),
                               strength=random.uniform(0.45, 0.7),
                               radius_frac=random.uniform(0.28, 0.42),
                               aspect_x=random.uniform(1.0, 2.5),
                               angle_deg=random.uniform(-25.0, 25.0))

    # 국소 블러(초점/손떨림으로 한쪽 구역만 흐릿한 경우) -- curl_shade와 별개로 독립
    # 추첨(같은 이미지에 그림자+국소블러가 동시에 걸릴 수도 있음, 둘 다 실사에서 흔한
    # 독립적 열화 경로라 배타적으로 만들 이유 없음). L3부터 도입, L1/L2는 확률 0.
    if random.random() < p.get('p_local_blur', 0.0):
        out = apply_local_blur(out,
                                radius_frac=random.uniform(0.25, 0.4),
                                strength=random.uniform(0.7, 1.0),
                                aspect_x=random.uniform(1.0, 2.0),
                                angle_deg=random.uniform(-25.0, 25.0),
                                ksize=random.choice([7, 9, 11]))

    # 실사 전처리(preprocess())의 언샤프 마스크 질감에 합성 단계에서 미리 노출(2026-08-02).
    if random.random() < p.get('p_unsharp_adapt', 0.0):
        out = apply_unsharp_like_real(out)

    # 각 노이즈 종류를 독립적으로 굴려 "이 종류가 이번에 등장할 자격이 있는지"만 정하고,
    # 실제 적용은 그중 max_concurrent개까지만 무작위로 골라 실행한다 -- 안 그러면 6종이
    # 한 샘플에 전부 겹쳐 실제 촬영에서는 나올 수 없는 수준으로 정보가 소실될 수 있다.
    # 그레인(real_texture)이 이미 실사 특유의 디테일/질감을 담당하므로, 발동 시
    # blur/motion_blur/downscale까지 겹치면 실사보다 훨씬 뭉개진다(2026-07-28 실측:
    # 실사 블러지표 중앙값 2318 vs 합성 103 -- 약 20배 차이, 육안으로도 글자를 알아보기
    # 힘든 수준). 그레인이 켜졌을 때는 blur/motion_blur를 끄고 jpeg는 다운스케일 없이
    # 압축 아티팩트만 남긴다.
    blur_prob  = 0.0 if used_real_texture else p['p_blur']
    mblur_prob = 0.0 if used_real_texture else p['p_motion_blur']
    jpeg_downscale = (1.0, 1.0) if used_real_texture else p['downscale']

    candidates = [
        ('brightness',  0.0 if used_real_texture else p['p_brightness'],  _apply_brightness),
        ('gauss_noise', p['p_gauss_noise'], _apply_gauss_noise),
        ('blur',        blur_prob,        _apply_blur),
        ('motion_blur', mblur_prob, lambda o: _motion_blur(o, p['motion_ksize'])),
        ('lighting',    0.0 if used_real_texture else p['p_lighting'],    lambda o: _uneven_lighting(o, p['lighting_strength'])),
        ('jpeg',        p['p_jpeg'],        lambda o: _jpeg_and_resize_degrade(o, p['jpeg_quality'], jpeg_downscale)),
    ]
    eligible_idx = [i for i, (_, prob, _) in enumerate(candidates) if random.random() < prob]
    max_concurrent = p['max_concurrent']
    if len(eligible_idx) > max_concurrent:
        eligible_idx = random.sample(eligible_idx, max_concurrent)
    for i in sorted(eligible_idx):  # 항상 같은 순서(광도->노이즈->블러->모션블러->조명->압축)로 적용
        out = candidates[i][2](out)
    return out


def apply_page_curl(gray: np.ndarray, strength_px: float, flat_side: str = 'left',
                     direction: float = 1.0) -> np.ndarray:
    """종이가 원통형으로 살짝 휘어지는 효과 시뮬레이션 -- correct_perspective(강체 변환)로는
    못 잡는 비선형 왜곡.

    2026-07-28 실사 40장 재검토로 모양을 다시 잡음: 기존엔 중앙은 평평하고 양쪽 가장자리가
    대칭으로 처지는 모양(코사인 범프)이었는데, 실사(특히 제본선 근처)를 보면 그런 대칭
    형태가 아니라 한쪽(제본선 쪽)은 거의 평평하다가 반대쪽으로 갈수록 단조증가로 휘어지는
    비대칭 곡선이었다 -- flat_side 쪽 끝은 변위 0, 반대쪽 끝이 strength_px. 이 함수는
    y_shift가 오직 x(열)에만 의존해 모든 행(오선 상단부터 하단까지)이 같은 열에서 항상
    같은 양만큼 이동하므로, 대보표에서 treble/bass가 항상 같은 만큼 같이 휘어져 위아래
    정합이 깨지지 않는다 -- 단, 이 성질은 회전보다 먼저 적용될 때만 유효하다(회전 후에
    적용하면 이미 기울어진 마디선이 서로 다른 열을 지나게 되어 treble/bass가 서로 다른
    굴곡을 받는다). 그래서 page_noise_and_redetect에서 굴곡을 회전보다 먼저 적용한다."""
    H, W = gray.shape
    x = np.arange(W, dtype=np.float32)
    xn = x / max(W - 1, 1)  # 0(왼쪽)..1(오른쪽)
    if flat_side == 'right':
        xn = 1.0 - xn
    y_shift = direction * strength_px * (1.0 - np.cos(xn * np.pi / 2.0))
    map_x, map_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_y = map_y + y_shift[np.newaxis, :]
    return cv2.remap(gray, map_x, map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def apply_curl_shade(gray: np.ndarray, corner: Optional[str] = None,
                      strength: float = 0.25, radius_frac: float = 0.35,
                      darken: bool = True,
                      center: Optional[Tuple[float, float]] = None,
                      aspect_x: float = 1.0,
                      angle_deg: float = 0.0) -> np.ndarray:
    """종이 굴곡을 기하 왜곡이 아니라 "이미지 편집" 스타일 부드러운 코너 그림자/
    하이라이트로 표현한다.

    2026-07-28 실사(IMG_8784) 재검토: 실제 사진에서 보이는 "종이 굴곡"은 오선 자체가
    휘어 보이는 게 아니라, 모서리가 살짝 들려서 생기는 완만한 음영 변화였다(오선 라인은
    거의 평평하게 유지됨). apply_page_curl(픽셀 remap)은 회전과 결합하면 대각선으로
    딱딱하게 잘려나가는 부자연스러운 경계가 생기고, 이 함수처럼 곱셈 마스크만 씌우는
    방식은 그런 경계가 없다. 기하를 전혀 바꾸지 않으므로(픽셀 좌표 이동 없음) 대보표의
    treble/bass 정합이 깨질 걱정이 원천적으로 없다.

    center: (cy, cx) 픽셀 좌표를 직접 주면 corner 대신 이 지점을 그림자 중심으로 씀
    (2026-07-31 추가 -- 모서리뿐 아니라 "가운데/초반부(왼쪽) 영역"처럼 임의 위치에도
    그림자를 주고 싶다는 요청 대응).
    aspect_x: 1.0=원형(기존). 1보다 크면 타원(오선은 가로로 길어서 그림자도 타원형으로
    퍼지는 게 더 자연스럽다는 요청 대응, 2026-07-31 추가) -- 장축 방향 거리를 aspect_x로
    나눠서 같은 반경이라도 그 방향으로 더 멀리까지 퍼지게 함.
    angle_deg: 타원의 장축 회전각(도, 2026-07-31 추가) -- 0이면 장축이 가로 방향
    (기존과 동일), 다른 값을 주면 타원 자체가 그만큼 기울어짐(그림자 방향도 매번
    달라지게 하려는 요청 대응).
    """
    H, W = gray.shape
    if center is not None:
        cy, cx = center
    else:
        if corner is None:
            corner = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
        cy, cx = {'top-left': (0, 0), 'top-right': (0, W),
                  'bottom-left': (H, 0), 'bottom-right': (H, W)}[corner]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dy = yy - cy
    dx = xx - cx
    if angle_deg != 0.0:
        theta = np.deg2rad(angle_deg)
        dx, dy = (dx * np.cos(theta) + dy * np.sin(theta),
                  -dx * np.sin(theta) + dy * np.cos(theta))
    dist = np.sqrt(dy ** 2 + (dx / max(aspect_x, 1e-6)) ** 2)
    max_dist = radius_frac * np.sqrt(H ** 2 + W ** 2)
    falloff = np.clip(1.0 - dist / max_dist, 0.0, 1.0) ** 2  # 모서리에서 멀어질수록 부드럽게 감쇠
    mask = 1.0 - strength * falloff if darken else 1.0 + strength * falloff
    out = np.clip(gray.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    return out


def apply_page_curl_bump(gray: np.ndarray, strength_px: float,
                          peak_frac: float = 0.5, axis: str = 'x') -> np.ndarray:
    """책을 펼쳤을 때 제본선(gutter) 근처가 위로 볼록 솟아오르는 형태의 대칭 굴곡
    (2026-07-31 추가). apply_page_curl()은 한쪽(제본선)은 평평하고 반대쪽으로 갈수록
    단조증가하는 "비대칭 램프" 모양인데, 이건 반대로 "가운데(또는 peak_frac 지점)가
    가장 볼록하고 양 끝은 평평한" 대칭 범프 모양 -- 책 머릿부분처럼 중앙이 튀어나온
    케이스 재현용. axis='x'면 가로 방향(열)로 범프, 'y'면 세로 방향(행)으로 범프
    (예: 페이지 상단이 말려 올라온 경우).
    """
    H, W = gray.shape
    n = W if axis == 'x' else H
    t = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    # peak_frac 지점에서 최대, 양 끝(0, 1)에서 0인 대칭 범프(사인 곡선 기반).
    bump = np.sin(np.clip(t / max(peak_frac, 1e-6), 0, 1) * np.pi / 2) ** 2
    bump2 = np.sin(np.clip((1 - t) / max(1 - peak_frac, 1e-6), 0, 1) * np.pi / 2) ** 2
    shift = strength_px * np.minimum(bump, bump2) / max(np.minimum(bump, bump2).max(), 1e-6)

    map_x, map_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    if axis == 'x':
        map_y = map_y - shift[np.newaxis, :]
    else:
        map_x = map_x - shift[:, np.newaxis]
    return cv2.remap(gray, map_x, map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def geometric_augment(gray: np.ndarray, label: Optional[np.ndarray] = None,
                       max_angle_deg: float = 4.0,
                       persp_margin_frac: float = 0.03,
                       p_rotate: float = 0.5,
                       p_persp: float = 0.4,
                       ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """프린트 후 카메라로 촬영할 때 남는 잔여 기울기(회전+원근)를 시뮬레이션.

    C++ 파이프라인은 segnet/encoder 입력 전에 perspective_corrector로 한 번
    보정하므로, 여기서 재현할 대상은 원본 촬영 각도가 아니라 그 보정 후에도
    남는 잔여 왜곡 수준이다. label을 같이 주면 동일한 변환행렬을 적용해
    정합을 유지한다(세그멘테이션 라벨용).
    """
    H, W = gray.shape
    out_gray  = gray
    out_label = label

    if random.random() < p_rotate:
        angle = random.uniform(-max_angle_deg, max_angle_deg)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        out_gray = cv2.warpAffine(np.ascontiguousarray(out_gray), M, (W, H),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        if out_label is not None:
            out_label = cv2.warpAffine(np.ascontiguousarray(out_label.astype(np.uint8)), M, (W, H),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=SEG_BG).astype(np.int64)

    if random.random() < p_persp:
        margin = max(1, int(min(H, W) * persp_margin_frac))
        src    = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        def r(): return random.randint(-margin, margin)
        dst = np.float32([[r(), r()], [W+r(), r()], [W+r(), H+r()], [r(), H+r()]])
        try:
            M = cv2.getPerspectiveTransform(src, dst)
            out_gray = cv2.warpPerspective(np.ascontiguousarray(out_gray), M, (W, H),
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            if out_label is not None:
                out_label = cv2.warpPerspective(np.ascontiguousarray(out_label.astype(np.uint8)), M, (W, H),
                                                flags=cv2.INTER_NEAREST,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=SEG_BG).astype(np.int64)
        except cv2.error:
            pass

    return out_gray, out_label


@contextmanager
def _frozen_rng(seed_key: int):
    """idx 기반 결정론적 시드로 전역 random/np.random 상태를 일시적으로 바꾼다.

    val 샘플은 매 epoch 같은 Dataset.__getitem__(idx)를 다시 호출하는데, 노이즈가 매번
    새로 무작위로 뽑히면 val_acc 곡선의 변동이 모델 개선인지 단순 측정 노이즈인지
    구분할 수 없다. val 인덱스에서만 idx로 시드를 고정해 매 epoch 동일한 노이즈가
    나오게 하고, 끝나면 원래 전역 RNG 상태로 복구해 train 샘플의 무작위성에는
    영향을 주지 않는다.
    """
    rand_state = random.getstate()
    np_state   = np.random.get_state()
    random.seed(seed_key)
    np.random.seed(seed_key % (2**32 - 1))
    try:
        yield
    finally:
        random.setstate(rand_state)
        np.random.set_state(np_state)


# ─────────────────────────────────────────────────────────────────────────────
#  대보표 토큰 분리: treble 토큰 / bass 토큰 per system
# ─────────────────────────────────────────────────────────────────────────────

def _has_grand_staff(token_strs: List[str]) -> bool:
    return 'staff-bass' in token_strs


def _split_grand_staff_interleaved(
    token_ids: List[int],
    n_systems: int,
    id2tok: Dict[int, str],
    tok2id: Dict[str, int],
    system_breaks: Optional[List[int]] = None,
) -> List[List[int]]:
    """
    대보표 인터리빙 시퀀스를 시스템별 전체 시퀀스로 분배.

    마디 구조: [treble_notes] staff-bass [bass_notes] barline
    → barline 기준으로 마디 분리 후 시스템별로 배분.

    system_breaks: generate_scores.py가 실제 렌더링 때 줄바꿈을 넣은 마디 인덱스
    목록(밀도 기반 등 마디 수가 균등하지 않은 경우). 길이가 n_systems-1과 정확히
    맞으면 이 경계를 그대로 쓰고, 없거나 안 맞으면(구버전 데이터 등) 예전처럼
    균등 분배로 fallback한다 -- 마디 수가 시스템 수로 안 나눠떨어지면 이미지와
    라벨이 어긋날 수 있으니 가능하면 항상 system_breaks를 넘겨줄 것.

    Returns:
        [sys0_ids, sys1_ids, ...]  길이 n_systems
    """
    if n_systems == 1:
        return [list(token_ids)]

    # 헤더 분리
    header_ids: List[int] = []
    body_start = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        s = id2tok.get(tok_id, '')
        is_hdr = (s in _HEADER_SPECIALS or
                  any(s.startswith(p) for p in _HEADER_PREFIXES))
        if is_hdr:
            header_ids.append(tok_id)
        else:
            body_start = i
            break

    body = token_ids[body_start:]

    # barline 기준 마디 분리 (staff-bass 포함 전체 인터리빙 유지)
    measures: List[List[int]] = []
    current: List[int] = []
    for tok_id in body:
        s = id2tok.get(tok_id, '')
        current.append(tok_id)
        if s in _BARLINE_STRS:
            measures.append(list(current))
            current = []
    if current:
        measures.append(current)

    total = len(measures)
    if total == 0:
        return [list(token_ids)] * n_systems

    if system_breaks is not None and len(system_breaks) == n_systems - 1 and \
            all(0 < b < total for b in system_breaks):
        bounds = [0] + list(system_breaks) + [total]
    else:
        bounds = [i * total // n_systems for i in range(n_systems + 1)]

    result: List[List[int]] = []
    for sys_i in range(n_systems):
        start, end = bounds[sys_i], bounds[sys_i + 1]
        sys_ids = list(header_ids)
        for m in measures[start:end]:
            sys_ids.extend(m)
        result.append(sys_ids if sys_ids else list(token_ids))
    return result


def _split_grand_staff_tokens(
    token_ids: List[int],
    n_systems: int,
    id2tok: Dict[int, str],
    tok2id: Dict[str, int],
) -> List[Tuple[List[int], List[int]]]:
    """
    대보표 토큰 시퀀스를 시스템별 (treble_ids, bass_ids) 쌍으로 분리.

    토큰 구조 (마디별 반복):
      [treble 음표들] staff-bass [clef-F] [bass 음표들] barline

    Returns:
        [(treble_ids_sys0, bass_ids_sys0), ...]  길이 n_systems
    """
    STAFF_BASS = 'staff-bass'
    SOS_ID_    = tok2id.get('<SOS>', 1)
    CLEF_F_ID  = tok2id.get('clef-F', tok2id.get('clef-G', 5))

    # 헤더 분리 (clef-G, key-*, time-* 계열)
    header_ids: List[int] = []
    body_start = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        s = id2tok.get(tok_id, '')
        is_hdr = (s in _HEADER_SPECIALS or
                  any(s.startswith(p) for p in _HEADER_PREFIXES))
        if is_hdr:
            header_ids.append(tok_id)
        else:
            body_start = i
            break

    body = token_ids[body_start:]

    # 마디 단위로 분리 (treble part + bass part + barline)
    # 구조: [treble 토큰들] staff-bass [bass 토큰들] barline
    MeasureEntry = Tuple[List[int], List[int], int]  # (treble, bass, barline_id)
    measures: List[MeasureEntry] = []

    treble_buf: List[int] = []
    bass_buf:   List[int] = []
    in_bass = False

    for tok_id in body:
        s = id2tok.get(tok_id, '')
        if s == STAFF_BASS:
            in_bass = True
            continue   # staff-bass 자체는 토큰 배열에 추가하지 않음
        if s in _BARLINE_STRS:
            measures.append((list(treble_buf), list(bass_buf), tok_id))
            treble_buf = []
            bass_buf   = []
            in_bass    = False
        elif in_bass:
            bass_buf.append(tok_id)
        else:
            treble_buf.append(tok_id)

    # 마지막 마디에 barline이 없는 경우
    if treble_buf or bass_buf:
        final_bar = tok2id.get('barline-final', tok2id.get('barline', PAD_ID))
        measures.append((treble_buf, bass_buf, final_bar))

    total = len(measures)
    if total == 0:
        # 대보표 파싱 실패 → 샘플 스킵 (staff-bass 포함 전체 시퀀스를 타겟으로 넣지 않음)
        return [([], [])] * n_systems

    # 시스템별로 마디를 균등 배분
    result: List[Tuple[List[int], List[int]]] = []
    for sys_i in range(n_systems):
        start = sys_i * total // n_systems
        end   = (sys_i + 1) * total // n_systems

        treble_ids = list(header_ids)

        # bass 헤더: clef-G → clef-F 교체, key/time은 treble과 동일하게 유지
        bass_header: List[int] = []
        clef_g_id = tok2id.get('clef-G', -1)
        for hid in header_ids:
            bass_header.append(CLEF_F_ID if hid == clef_g_id else hid)
        bass_ids   = [SOS_ID_] + bass_header
        first_bass = True

        for t_toks, b_toks, bar_id in measures[start:end]:
            treble_ids.extend(t_toks)
            treble_ids.append(bar_id)

            if b_toks:
                if first_bass:
                    # clef-F가 bass 토큰 앞에 이미 있으면 헤더와 중복 → 제거
                    if b_toks[0] == CLEF_F_ID:
                        b_toks = b_toks[1:]
                    first_bass = False
                bass_ids.extend(b_toks)
                bass_ids.append(bar_id)

        if not treble_ids:
            treble_ids = []   # 분리 실패 → 스킵 (staff-bass 포함 전체 할당 제거)
        if not bass_ids or bass_ids == [SOS_ID_]:
            bass_ids = []     # 분리 실패 → 스킵

        result.append((treble_ids, bass_ids))

    return result


def _split_token_ids_by_rows(
    token_ids: List[int],
    n_rows: int,
    id2tok: Dict[int, str],
) -> List[List[int]]:
    """단일 오선 다중 행: round2train/dataset.py와 동일한 마디 분배 로직."""
    if n_rows <= 1:
        return [list(token_ids)]
    header_ids: List[int] = []
    body_start = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        s = id2tok.get(tok_id, '')
        if s in _HEADER_SPECIALS or any(s.startswith(p) for p in _HEADER_PREFIXES):
            header_ids.append(tok_id)
        else:
            body_start = i
            break
    body = token_ids[body_start:]
    measures: List[List[int]] = []
    current: List[int] = []
    for tok_id in body:
        s = id2tok.get(tok_id, '')
        current.append(tok_id)
        if s in _BARLINE_STRS:
            measures.append(list(current))
            current = []
    if current:
        measures.append(current)
    total = len(measures)
    if total == 0:
        return [list(token_ids)] * n_rows
    result: List[List[int]] = []
    for row_i in range(n_rows):
        start   = row_i * total // n_rows
        end     = (row_i + 1) * total // n_rows
        row_ids = list(header_ids)
        for m in measures[start:end]:
            row_ids.extend(m)
        if not row_ids:
            row_ids = list(token_ids)
        result.append(row_ids)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  SegnetDataset
# ─────────────────────────────────────────────────────────────────────────────

class SegnetDataset(Dataset):
    def __init__(self, data_dir: str, patches_per_image: int = 8,
                 augment: bool = True, patch_size: int = PATCH_SIZE,
                 noise_level: int = 2):
        self.data_dir     = data_dir
        self.n_patches    = patches_per_image
        self.augment      = augment
        self.patch_size   = patch_size
        self.noise_level  = noise_level
        self.val_indices: set = set()   # split_dataset()이 채움 -- 비어있으면 전부 train 취급
        self.image_paths: List[str] = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and not fname.endswith('_seg.png'):
                self.image_paths.append(os.path.join(data_dir, fname))
        self._len = len(self.image_paths) * patches_per_image
        print(f"[SegnetDataset] {len(self.image_paths)} × {patches_per_image} = {self._len}")

    def __len__(self): return self._len

    def __getitem__(self, idx):
        img_path = self.image_paths[idx // self.n_patches]
        stem  = os.path.splitext(img_path)[0]
        gray0 = load_preprocessed(img_path)

        lab_path = stem + '_seg.png'
        if os.path.isfile(lab_path):
            label0 = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
        else:
            weak_cache = stem + '_weak.npy'
            label0 = _cached_npy(
                weak_cache,
                lambda: generate_weak_seg_labels(gray0).astype(np.uint8)
            ).astype(np.int64)
        if label0.shape != gray0.shape:
            label0 = cv2.resize(label0.astype(np.uint8),
                                (gray0.shape[1], gray0.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(np.int64)

        is_val = self.augment and idx in self.val_indices
        with (_frozen_rng(idx) if is_val else nullcontext()):
            if self.augment:
                lvl = NOISE_LEVELS[self.noise_level]
                gray, label = geometric_augment(gray0, label0,
                                                max_angle_deg=lvl['angle_page'],
                                                persp_margin_frac=lvl['persp_page'],
                                                p_rotate=lvl['p_rotate'], p_persp=lvl['p_persp'])
                gray = augment_image(gray, level=self.noise_level)
            else:
                gray, label = gray0.copy(), label0
            P  = self.patch_size
            H, W = gray.shape
            if H < P or W < P:
                gray  = cv2.copyMakeBorder(gray,  0, max(0,P-H), 0, max(0,P-W), cv2.BORDER_CONSTANT, value=255)
                label = cv2.copyMakeBorder(label.astype(np.uint8), 0, max(0,P-H), 0, max(0,P-W),
                                            cv2.BORDER_CONSTANT, value=0).astype(np.int64)
                H, W  = gray.shape
            y0 = random.randint(0, H - P)
            x0 = random.randint(0, W - P)
        img_t = torch.from_numpy(gray[y0:y0+P, x0:x0+P].astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        lbl_t = torch.from_numpy(label[y0:y0+P, x0:x0+P].copy())
        return img_t, lbl_t


# ─────────────────────────────────────────────────────────────────────────────
#  OMRDataset (Round 3: grand staff 지원)
# ─────────────────────────────────────────────────────────────────────────────

class OMRDataset(Dataset):
    """
    Round 3 전용 데이터셋.

    대보표 이미지 처리:
    - N_STAVES = 짝수 → N_STAVES/2 시스템 (treble + bass 쌍)
    - treble 오선 → treble 토큰 (staff-bass 이전)
    - bass   오선 → bass 토큰   (staff-bass 이후)

    단일 오선 이미지 (Round 1/2 누적 데이터):
    - 기존과 동일: 행별 마디 분배
    """

    def __init__(self, data_dir: str, tokenizer: Dict[str, int],
                 max_seq: int = 512, augment: bool = True,
                 replay_dir: str = None, replay_count: int = 0, replay_seed: int = 42,
                 noise_level: int = 2, noise_level_max: Optional[int] = None,
                 p_level_max: float = 0.5, page_level_noise: bool = False,
                 in_ch: int = 1):
        """
        in_ch=2: CoordConv 실험(2026-07-31) -- 캔버스 그레이스케일 채널에 정규화된 세로
        좌표 채널을 하나 더 붙여 [2,H,W]로 만든다. 모델(model.py Encoder)이 세로(음높이)
        위치를 명시적으로 못 받고 CNN이 학습으로만 추론해야 하는 게 단3도/옥타브 오독의
        구조적 원인일 수 있다는 가설 검증용 -- 기본값 1(기존과 동일)이면 아무 영향 없음.

        replay_dir/replay_count: 지정 시 이전 단계 학습 디렉토리에서 파일을 복사하지 않고
        원본 경로 그대로 무작위 replay_count개를 섞어 읽는다. 복사/symlink가 필요 없어
        네트워크 마운트 상의 대량 파일 복사 지연이 사라지고, 원본 디렉토리에 이미 계산된
        전처리 캐시(_pre.npy/_staffs.json)를 그대로 재사용해 중복 계산도 없앤다.
        (replay 데이터는 이전 단계 학습에 이미 성공적으로 쓰여 검증된 것으로 간주 -- 재검증 불필요)

        page_level_noise: 5n5+ 전용. True면 캔버스 대신 페이지 레벨에서 노이즈+
        correct_perspective+오선 재검출을 거쳐 실제 추론 경로를 재현한다(page_noise_and_
        redetect 참고). 재검출 실패 시에만 기존 캔버스 레벨 경로로 폴백.

        noise_level_max: 지정하면 [noise_level, noise_level_max] 범위에서 샘플마다 레벨을
        뽑는다(2026-07-24, 목표 촬영 조건이 "약~중간 기울기(L2~L3)"로 확정돼 단일 레벨 대신
        그 구간 전체를 노출시키려는 목적).

        p_level_max: noise_level_max를 뽑을 확률(기본 0.5=균등, 나머지는 noise_level).
        2026-07-25, 5n5 사후분석: L2/L3 균등 랜덤(0.5)으로 돌렸더니 재검출 성공률 격차
        (L2 92% vs L3 56%)때문에 "진짜" page_level_noise 학습 신호가 L2 쪽에 훨씬 많이
        쏠렸고, L2는 개선(76.7%->80.8%)됐지만 L3는 오히려 악화(41.5%->35.3%)됐다. 성공률
        역수 비율(0.92/0.56≈1.64)만큼 상위 레벨을 더 자주 뽑아야 실효 노출량이 균형을
        맞춘다 -- 0.5보다 높게(예: 0.65) 주면 noise_level_max(예: 3) 쪽으로 편향.
        """
        self.max_seq          = max_seq
        self.augment          = augment
        self.noise_level      = noise_level
        self.noise_level_max  = noise_level_max
        self.p_level_max      = p_level_max
        self.page_level_noise = page_level_noise
        self.in_ch            = in_ch
        self.val_indices: set = set()   # split_dataset()이 채움 -- 비어있으면 전부 train 취급
        tok2id = tokenizer
        id2tok = {v: k for k, v in tok2id.items()}
        self._tok2id = tok2id
        self._id2tok = id2tok

        self.samples: List[Tuple[str, List[int], Dict]] = []

        dir_fnames = [(data_dir, f) for f in sorted(os.listdir(data_dir))
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if replay_dir and replay_count > 0:
            replay_fnames = sorted(f for f in os.listdir(replay_dir)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            chosen = random.Random(replay_seed).sample(
                replay_fnames, min(replay_count, len(replay_fnames)))
            dir_fnames += [(replay_dir, f) for f in chosen]
            print(f"  Replay: {len(chosen)}/{replay_count} from {replay_dir}")

        def _load_one(dir_fname):
            """파일 1개 분량의 라벨 파싱 + 캐시 로드/계산 (I/O-bound, 스레드로 병렬화 대상)."""
            src_dir, fname = dir_fname
            stem     = os.path.splitext(fname)[0]
            img_path = os.path.join(src_dir, fname)
            gt_path  = os.path.join(src_dir, stem + '.json')
            if not os.path.isfile(gt_path):
                return None
            try:
                with open(gt_path, encoding='utf-8') as f:
                    data = json.load(f)
                token_strs = [t for t in data.get('tokens', [])
                              if t not in ('<SOS>', '<EOS>', '<PAD>')]
                system_breaks = data.get('system_breaks')  # None이면 예전 데이터(균등분배 fallback)
            except Exception:
                return None
            if not token_strs:
                return None

            ids = [tok2id.get(t, tok2id.get('<UNK>', 3)) for t in token_strs]
            ids = ids[:max_seq]
            is_grand = _has_grand_staff(token_strs)

            try:
                gray   = load_preprocessed(img_path)
                staffs = load_staffs_cached(img_path, gray)
            except Exception:
                return None
            if not staffs:
                return None

            return img_path, ids, is_grand, staffs, system_breaks

        skipped = 0
        # 네트워크 마운트에서 파일당 open/stat 지연이 커서 스레드풀로 병렬화
        # (I/O 대기 중 GIL이 풀리므로 스레드만으로 충분 — round3train/prewarm_cache.py와 동일 방식)
        # 2026-07-30: 예전 pod 컨테이너 cgroup 메모리 한도(~46.5GB)에서 32스레드 동시 전처리
        # (cv2 연산 + 캐시 미스 시 CLAHE/bilateralFilter 등 무거운 연산)가 메모리 스파이크로
        # OOM-kill을 유발해 6으로 낮췄었음. 2026-07-31: `free -h`가 보여준 호스트 전체
        # 메모리(440GB)만 보고 32로 다시 올렸다가 동일하게 OOM-kill(exit 137) 재현됨 --
        # 실제 컨테이너 cgroup 한도(/sys/fs/cgroup/memory.max)는 여전히 약 50.3GB로 예전
        # 포드와 비슷한 수준이었음(호스트 메모리 표시와 실제 사용 가능량은 별개). 6과 32
        # 사이 절충값 12로 재시도.
        with ThreadPoolExecutor(max_workers=12) as ex:
            for result in ex.map(_load_one, dir_fnames):
                if result is None:
                    skipped += 1
                    continue
                img_path, ids, is_grand, staffs, system_breaks = result
                n_staffs = len(staffs)

                if is_grand and n_staffs >= 2 and n_staffs % 2 == 0:
                    # 대보표: 시스템당 1샘플, 전체 인터리빙 시퀀스 + system canvas(SYSTEM_CANVAS_H×CANVAS_W)
                    n_systems = n_staffs // 2
                    sys_token_lists = _split_grand_staff_interleaved(
                        ids, n_systems, id2tok, tok2id, system_breaks)

                    for sys_i, sys_ids in enumerate(sys_token_lists):
                        if not sys_ids:
                            continue
                        treble_staff = staffs[sys_i * 2]
                        bass_staff   = staffs[sys_i * 2 + 1]
                        self.samples.append(
                            (img_path, sys_ids[:max_seq], [treble_staff, bass_staff])
                        )
                else:
                    # 단일 오선 또는 홀수 오선: Round 2 방식 (행별 마디 분배)
                    row_token_lists = _split_token_ids_by_rows(ids, n_staffs, id2tok)
                    for staff, row_ids in zip(staffs, row_token_lists):
                        self.samples.append((img_path, row_ids, staff))

        print(f"[OMRDataset R3] {len(self.samples)} samples "
              f"({skipped} skipped)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, ids, staff = self.samples[idx]
        gray0 = load_preprocessed(img_path)

        # 실사 촬영 사진(jpg/jpeg)은 이미 실제 노이즈를 담고 있으므로 합성 증강을 추가로
        # 얹지 않는다(2026-08-02 사용자 지시) -- 합성 렌더링(png)에만 augment 적용.
        is_real_photo = img_path.lower().endswith(('.jpg', '.jpeg'))
        aug = self.augment and not is_real_photo

        is_val = aug and idx in self.val_indices
        with (_frozen_rng(idx) if is_val else nullcontext()):
            lvl_idx = self.noise_level
            if self.noise_level_max:
                lvl_idx = (self.noise_level_max if random.random() < self.p_level_max
                          else self.noise_level)
            tile = None
            if aug and self.page_level_noise:
                tile = page_noise_and_redetect(gray0, staff, lvl_idx)

            if tile is None:
                # page_level_noise=False, 또는 재검출 실패 시 폴백: 기존 캔버스 레벨 경로.
                gray = augment_image(gray0, level=lvl_idx) if aug else gray0
                if isinstance(staff, list):
                    tile = extract_system_canvas(gray, staff)   # 대보표: SYSTEM_CANVAS_H×CANVAS_W
                else:
                    tile = extract_staff_canvas(gray, staff)    # 단일 오선: CANVAS_H×CANVAS_W
                if aug:
                    # 캔버스는 폭 대비 높이가 매우 작아(예: 1280×320) 전체 페이지 기준 각도를
                    # 그대로 쓰면 가장자리 콘텐츠가 잘려나간다. 훨씬 약한 잔여 기울기만 적용.
                    lvl = NOISE_LEVELS[lvl_idx]
                    tile, _ = geometric_augment(tile, max_angle_deg=lvl['angle_canvas'],
                                                persp_margin_frac=lvl['persp_canvas'],
                                                p_rotate=lvl['p_rotate'], p_persp=lvl['p_persp'])
        canvas   = (tile.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
        canvas_t = make_model_input(canvas, self.in_ch)

        tgt_in  = [SOS_ID] + ids
        tgt_out = ids + [EOS_ID]
        return canvas_t, torch.tensor(tgt_in, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  Collate / helpers
# ─────────────────────────────────────────────────────────────────────────────

def omr_collate(batch):
    canvases, tgt_ins, tgt_outs = zip(*batch)
    # 단일 오선(256px)과 대보표(384px)가 섞일 수 있으므로 max_H로 white-pad.
    # 채널 수는 하드코딩하지 않고 실제 입력에서 읽음(2026-07-31, CoordConv in_ch=2 대응) --
    # 그레이스케일(채널0)은 기존처럼 white로, 좌표 채널(채널1, 있는 경우)은 "콘텐츠 없음"을
    # 뜻하는 중립값 0으로 채운다(둘 다 padding=무시할 영역이라는 의도는 동일).
    C     = canvases[0].shape[0]
    max_H = max(c.shape[1] for c in canvases)
    max_W = max(c.shape[2] for c in canvases)
    white = (1.0 - IMG_MEAN) / IMG_STD
    padded = []
    for c in canvases:
        H, W = c.shape[1], c.shape[2]
        if H < max_H or W < max_W:
            p = torch.zeros((C, max_H, max_W), dtype=c.dtype)
            p[0].fill_(white)
            p[:, :H, :W] = c
            padded.append(p)
        else:
            padded.append(c)
    canvases = torch.stack(padded, dim=0)
    max_T    = max(t.size(0) for t in tgt_ins)
    B        = len(tgt_ins)
    tgt_in_p  = torch.full((B, max_T), PAD_ID, dtype=torch.long)
    tgt_out_p = torch.full((B, max_T), PAD_ID, dtype=torch.long)
    mask      = torch.ones(B, max_T, dtype=torch.bool)
    for i, (ti, to) in enumerate(zip(tgt_ins, tgt_outs)):
        L = ti.size(0)
        tgt_in_p[i, :L]  = ti
        tgt_out_p[i, :L] = to
        mask[i, :L]       = False
    return canvases, tgt_in_p, tgt_out_p, mask


def load_tokenizer(path: str):
    with open(path, encoding='utf-8') as f:
        tok2id = json.load(f)
    return tok2id, {v: k for k, v in tok2id.items()}


def split_dataset(dataset, val_ratio: float = 0.1, seed: int = 42):
    from torch.utils.data import Subset
    n     = len(dataset)
    n_val = max(1, int(n * val_ratio))
    perm    = np.random.default_rng(seed).permutation(n).tolist()
    val_idx = perm[:n_val]
    if hasattr(dataset, 'val_indices'):
        # val 샘플은 __getitem__에서 idx로 시드를 고정해 매 epoch 같은 노이즈로
        # 평가되게 한다(geometric_augment/_frozen_rng 참고) -- val_acc 곡선의 변동이
        # 모델 개선인지 측정 노이즈인지 구분하기 위함.
        dataset.val_indices = set(val_idx)
    return Subset(dataset, perm[n_val:]), Subset(dataset, val_idx)
