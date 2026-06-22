"""
round1train/dataset.py  –  Multi-row aware OMR dataset for Round 1.

핵심 수정: 이미지에 오선 행이 N개 검출될 때 전체 토큰 시퀀스를 행 수만큼
분할하여 각 행에 해당 마디의 토큰만 배정한다.

기존 코드의 버그:
  - 이미지에 2행 오선이 있으면 두 행 모두 동일한 전체 토큰 시퀀스를 GT로 받음
  - 행 1: 1~3마디 이미지 → GT: 1~6마디 전체 (불가능한 학습)
  - 행 2: 4~6마디 이미지 → GT: 1~6마디 전체 (완전히 틀린 매핑)

수정 후:
  - 행 1: 1~3마디 이미지 → GT: 1~3마디 토큰만
  - 행 2: 4~6마디 이미지 → GT: 4~6마디 토큰만

추가: extract_staff_canvas() 적용 (team_ml 방식).
  타일 분할 대신 오선 전체를 1280px에 압축 → 이미지-토큰 정렬 보장.
"""

import json
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
#  Constants (model.py, C++ 엔진과 일치해야 함)
# ─────────────────────────────────────────────────────────────────────────────

PAD_ID       = 0
SOS_ID       = 1
EOS_ID       = 2

PATCH_SIZE   = 320
CANVAS_H     = 256
CANVAS_W     = 1280
TILE_OVERLAP = 64
MARGIN_UNITS = 2.0
TARGET_W     = 1920
IMG_MEAN     = 0.7931
IMG_STD      = 0.1738

SEG_BG         = 0
SEG_STEM_REST  = 1
SEG_NOTEHEAD   = 2
SEG_CLEF_KEY   = 3
SEG_STAFF_LINE = 4
SEG_SYMBOL     = 5
SEG_NUM_CLS    = 6

MIN_UNIT = 11.0
MAX_UNIT = 60.0

# 헤더 토큰 식별용 (note/rest/chord/barline 직전까지)
_HEADER_PREFIXES = ('clef-', 'key-', 'time-')
_HEADER_SPECIALS = {'<SOS>', '<EOS>', '<PAD>', '<UNK>'}
_BARLINE_STRS    = frozenset({
    'barline', 'barline-final', 'barline-end-repeat',
    'barline-start-repeat', 'barline-double',
})


# ─────────────────────────────────────────────────────────────────────────────
#  전처리 (C++ Preprocessor 미러)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if w * h < 0.80 * gray.shape[0] * gray.shape[1]:
            px = max(2, gray.shape[1] // 100)
            py = max(2, gray.shape[0] // 100)
            gray = gray[max(0, y - py):y + h + py, max(0, x - px):x + w + px]
    H, W = gray.shape
    if W != TARGET_W:
        scale = TARGET_W / W
        gray = cv2.resize(gray, (TARGET_W, int(round(H * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.bilateralFilter(gray, 9, 20.0, 7.0)
    return gray


def detect_staffs(gray: np.ndarray) -> List[Dict]:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    H, W = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 8, 1))
    staff_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    proj = staff_binary.mean(axis=1) / 255.0
    THRESH = 0.05
    candidates = [(float(proj[i]), i) for i in range(1, H - 1)
                  if proj[i] > THRESH and proj[i] >= proj[i - 1] and proj[i] >= proj[i + 1]]
    candidates.sort(reverse=True)
    suppressed = np.zeros(H, dtype=bool)
    peaks = []
    for val, row in candidates:
        if suppressed[row]:
            continue
        peaks.append(row)
        lo = max(0, int(row - MIN_UNIT / 2))
        hi = min(H - 1, int(row + MIN_UNIT / 2))
        suppressed[lo:hi + 1] = True
    peaks.sort()
    if len(peaks) < 5:
        return []
    staffs = []
    i = 0
    while i + 4 < len(peaks):
        gaps = [peaks[i + k + 1] - peaks[i + k] for k in range(4)]
        if all(MIN_UNIT <= g <= MAX_UNIT for g in gaps):
            unit = sum(gaps) / 4.0
            stddev = (sum((g - unit) ** 2 for g in gaps) / 4.0) ** 0.5
            if stddev <= 0.6 * unit:
                staffs.append({'y_lines': [float(peaks[i + k]) for k in range(5)],
                                'unit_size': unit})
                i += 5
                continue
        i += 1
    return staffs


def extract_staff_canvas(gray: np.ndarray, staff: Dict) -> np.ndarray:
    """오선 행 전체를 CANVAS_H×CANVAS_W 한 장으로 압축. (team_ml 방식)"""
    H, W = gray.shape
    margin = staff['unit_size'] * MARGIN_UNITS
    y_top = max(0, int(staff['y_lines'][0] - margin))
    y_bot = min(H - 1, int(staff['y_lines'][4] + margin))
    strip = gray[y_top:y_bot + 1, :]
    if strip.size == 0:
        return np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    interp = cv2.INTER_AREA if strip.shape[1] > CANVAS_W else cv2.INTER_LINEAR
    return cv2.resize(strip, (CANVAS_W, CANVAS_H), interpolation=interp)


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
        if bh < 1:
            continue
        ratio = bw / bh
        if 15 <= area <= 600 and 0.4 <= ratio <= 2.5:
            labels[comp == i] = SEG_NOTEHEAD
    labels[(binary > 0) & (labels == SEG_BG)] = SEG_SYMBOL
    return labels


def augment_image(gray: np.ndarray, perspective: bool = False) -> np.ndarray:
    out = gray.copy()
    if random.random() < 0.6:
        beta  = random.uniform(-25, 25)
        alpha = random.uniform(0.85, 1.15)
        out   = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.4:
        sigma = random.uniform(2.0, 8.0)
        out   = np.clip(out.astype(np.float32) + np.random.normal(0, sigma, out.shape),
                        0, 255).astype(np.uint8)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        out   = cv2.GaussianBlur(out, (ksize, ksize), 0)
    if perspective and random.random() < 0.3:
        H, W   = out.shape
        margin = int(min(H, W) * 0.03)
        src    = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        def r(): return random.randint(-margin, margin)
        dst = np.float32([[r(), r()], [W + r(), r()], [W + r(), H + r()], [r(), H + r()]])
        try:
            M   = cv2.getPerspectiveTransform(src, dst)
            out = cv2.warpPerspective(np.ascontiguousarray(out), M, (W, H),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        except cv2.error:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  핵심: 다중 행 토큰 분할
# ─────────────────────────────────────────────────────────────────────────────

def _split_token_ids_by_rows(
    token_ids: List[int],
    n_rows: int,
    id2tok: Dict[int, str],
) -> List[List[int]]:
    """
    전체 토큰 시퀀스를 n_rows개의 행에 분배한다.
    각 행은 헤더(clef/key/time) + 해당 행의 마디 토큰을 받는다.

    Args:
        token_ids : 전체 토큰 ID 리스트 (SOS 포함, EOS 미포함 권장)
        n_rows    : 이미지에서 검출된 오선 행 수
        id2tok    : ID → 토큰 문자열 사전

    Returns:
        길이 n_rows의 리스트. 각 원소는 해당 행의 토큰 ID 리스트.
    """
    if n_rows <= 1:
        return [list(token_ids)]

    # 헤더 분리: clef-*, key-*, time-*, <SOS> 계열
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

    # 마디 단위로 분리
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

    # 행별로 마디를 균등 배분
    result: List[List[int]] = []
    for row_i in range(n_rows):
        start = row_i * total // n_rows
        end   = (row_i + 1) * total // n_rows
        row_ids = list(header_ids)
        for m in measures[start:end]:
            row_ids.extend(m)
        if not row_ids:
            row_ids = list(token_ids)  # fallback
        result.append(row_ids)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  SegnetDataset
# ─────────────────────────────────────────────────────────────────────────────

class SegnetDataset(Dataset):
    def __init__(self, data_dir: str, patches_per_image: int = 8,
                 augment: bool = True, patch_size: int = PATCH_SIZE):
        self.data_dir  = data_dir
        self.n_patches = patches_per_image
        self.augment   = augment
        self.patch_size = patch_size
        self.image_paths: List[str] = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and not fname.endswith('_seg.png'):
                self.image_paths.append(os.path.join(data_dir, fname))
        self._len = len(self.image_paths) * patches_per_image
        print(f"[SegnetDataset] {len(self.image_paths)} images × {patches_per_image} patches = {self._len}")

    def __len__(self): return self._len

    def __getitem__(self, idx):
        img_path = self.image_paths[idx // self.n_patches]
        bgr  = cv2.imread(img_path)
        gray = preprocess(bgr)
        if self.augment:
            gray = augment_image(gray, perspective=False)
        stem     = os.path.splitext(img_path)[0]
        lab_path = stem + '_seg.png'
        if os.path.isfile(lab_path):
            label = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
        else:
            label = generate_weak_seg_labels(gray)
        if label.shape != gray.shape:
            label = cv2.resize(label.astype(np.uint8),
                               (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(np.int64)
        P  = self.patch_size
        H, W = gray.shape
        if H < P or W < P:
            gray  = cv2.copyMakeBorder(gray,  0, max(0, P-H), 0, max(0, P-W), cv2.BORDER_CONSTANT, value=255)
            label = cv2.copyMakeBorder(label.astype(np.uint8), 0, max(0, P-H), 0, max(0, P-W),
                                        cv2.BORDER_CONSTANT, value=0).astype(np.int64)
            H, W  = gray.shape
        y0 = random.randint(0, H - P)
        x0 = random.randint(0, W - P)
        img_t = torch.from_numpy(gray[y0:y0+P, x0:x0+P].astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        lbl_t = torch.from_numpy(label[y0:y0+P, x0:x0+P].copy())
        return img_t, lbl_t


# ─────────────────────────────────────────────────────────────────────────────
#  OMRDataset (Round 2: 다중 행 지원)
# ─────────────────────────────────────────────────────────────────────────────

class OMRDataset(Dataset):
    """
    Round 2 전용 OMR 데이터셋.

    변경점:
    1. 다중 오선 행 처리: 행 수만큼 토큰 시퀀스를 마디 단위로 분할
    2. extract_staff_canvas() 사용: 오선 전체를 1장 캔버스로 압축
    """

    def __init__(self, data_dir: str, tokenizer: Dict[str, int],
                 max_seq: int = 512, augment: bool = True):
        self.max_seq = max_seq
        self.augment = augment
        tok2id = tokenizer
        id2tok = {v: k for k, v in tok2id.items()}
        self._id2tok = id2tok

        # samples: (img_path, row_token_ids, staff_dict)
        self.samples: List[Tuple[str, List[int], Dict]] = []

        skipped = 0
        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            stem     = os.path.splitext(fname)[0]
            img_path = os.path.join(data_dir, fname)
            gt_path  = os.path.join(data_dir, stem + '.json')
            if not os.path.isfile(gt_path):
                skipped += 1
                continue
            try:
                with open(gt_path, encoding='utf-8') as f:
                    data = json.load(f)
                token_strs = [t for t in data.get('tokens', [])
                              if t not in ('<SOS>', '<EOS>', '<PAD>')]
            except Exception:
                skipped += 1
                continue
            if not token_strs:
                skipped += 1
                continue

            ids = [tok2id.get(t, tok2id.get('<UNK>', 3)) for t in token_strs]
            ids = ids[:max_seq]

            try:
                bgr    = cv2.imread(img_path)
                gray   = preprocess(bgr)
                staffs = detect_staffs(gray)
            except Exception:
                skipped += 1
                continue
            if not staffs:
                skipped += 1
                continue

            n_rows = len(staffs)
            # 행별 토큰 분할 (핵심 수정)
            row_token_lists = _split_token_ids_by_rows(ids, n_rows, id2tok)

            for staff, row_ids in zip(staffs, row_token_lists):
                self.samples.append((img_path, row_ids, staff))

        print(f"[OMRDataset R2] {len(self.samples)} samples "
              f"({skipped} skipped — no staff or label)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, ids, staff = self.samples[idx]
        bgr  = cv2.imread(img_path)
        gray = preprocess(bgr)
        if self.augment:
            gray = augment_image(gray)

        tile = extract_staff_canvas(gray, staff)

        canvas = (tile.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
        canvas_t = torch.from_numpy(canvas).unsqueeze(0)

        tgt_in  = [SOS_ID] + ids
        tgt_out = ids + [EOS_ID]
        return canvas_t, torch.tensor(tgt_in, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  Collate / split helpers
# ─────────────────────────────────────────────────────────────────────────────

def omr_collate(batch):
    canvases, tgt_ins, tgt_outs = zip(*batch)
    canvases = torch.stack(canvases, dim=0)
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
    perm  = np.random.default_rng(seed).permutation(n).tolist()
    return Subset(dataset, perm[n_val:]), Subset(dataset, perm[:n_val])
