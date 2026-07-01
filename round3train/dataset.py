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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

PAD_ID       = 0
SOS_ID       = 1
EOS_ID       = 2

PATCH_SIZE   = 320
CANVAS_H        = 256
CANVAS_W        = 1280
SYSTEM_CANVAS_H = 384   # grand staff system canvas (treble+bass 수직 결합)
MARGIN_UNITS    = 2.0
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

_HEADER_PREFIXES = ('clef-', 'key-', 'time-')
_HEADER_SPECIALS = {'<SOS>', '<EOS>', '<PAD>', '<UNK>'}
_BARLINE_STRS    = frozenset({
    'barline', 'barline-final', 'barline-end-repeat',
    'barline-start-repeat', 'barline-double',
})


# ─────────────────────────────────────────────────────────────────────────────
#  전처리 (C++ 미러)
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
            gray = gray[max(0, y-py):y+h+py, max(0, x-px):x+w+px]
    H, W = gray.shape
    if W != TARGET_W:
        scale = TARGET_W / W
        gray  = cv2.resize(gray, (TARGET_W, int(round(H * scale))),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.bilateralFilter(gray, 9, 20.0, 7.0)
    return gray


def detect_staffs(gray: np.ndarray) -> List[Dict]:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    H, W = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 8, 1))
    sb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    proj = sb.mean(axis=1) / 255.0
    THRESH = 0.05
    candidates = [(float(proj[i]), i) for i in range(1, H - 1)
                  if proj[i] > THRESH and proj[i] >= proj[i-1] and proj[i] >= proj[i+1]]
    candidates.sort(reverse=True)
    suppressed = np.zeros(H, dtype=bool)
    peaks = []
    for val, row in candidates:
        if suppressed[row]: continue
        peaks.append(row)
        lo = max(0, int(row - MIN_UNIT / 2))
        hi = min(H - 1, int(row + MIN_UNIT / 2))
        suppressed[lo:hi+1] = True
    peaks.sort()
    if len(peaks) < 5: return []
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


def extract_staff_canvas(gray: np.ndarray, staff: Dict) -> np.ndarray:
    H, W = gray.shape
    margin = staff['unit_size'] * MARGIN_UNITS
    y_top  = max(0, int(staff['y_lines'][0] - margin))
    y_bot  = min(H - 1, int(staff['y_lines'][4] + margin))
    strip  = gray[y_top:y_bot+1, :]
    if strip.size == 0:
        return np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    interp = cv2.INTER_AREA if strip.shape[1] > CANVAS_W else cv2.INTER_LINEAR
    return cv2.resize(strip, (CANVAS_W, CANVAS_H), interpolation=interp)


def extract_system_canvas(gray: np.ndarray, staffs: List[Dict]) -> np.ndarray:
    """
    Grand staff: staffs[0](treble) + staffs[1](bass) 영역을 수직으로 이어붙여
    SYSTEM_CANVAS_H × CANVAS_W 캔버스를 반환한다.
    모델이 학습된 방식과 동일 (ml/omr/training/dataset.py 기준).
    """
    H, W   = gray.shape
    half_h = SYSTEM_CANVAS_H // 2  # 192

    def _strip(staff: Dict) -> np.ndarray:
        margin = staff['unit_size'] * MARGIN_UNITS
        y_top  = max(0, int(staff['y_lines'][0] - margin))
        y_bot  = min(H - 1, int(staff['y_lines'][4] + margin))
        s      = gray[y_top:y_bot + 1, :]
        scale  = half_h / max(s.shape[0], 1)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(s, (int(round(s.shape[1] * scale)), half_h),
                          interpolation=interp)

    def _pad(strip: np.ndarray) -> np.ndarray:
        tile = np.full((half_h, CANVAS_W), 255, dtype=np.uint8)
        cw = min(CANVAS_W, strip.shape[1])
        tile[:, :cw] = strip[:, :cw]
        return tile

    return np.vstack([_pad(_strip(staffs[0])), _pad(_strip(staffs[1]))])


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
        dst = np.float32([[r(), r()], [W+r(), r()], [W+r(), H+r()], [r(), H+r()]])
        try:
            M   = cv2.getPerspectiveTransform(src, dst)
            out = cv2.warpPerspective(np.ascontiguousarray(out), M, (W, H),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        except cv2.error:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  대보표 토큰 분리: treble 토큰 / bass 토큰 per system
# ─────────────────────────────────────────────────────────────────────────────

def _has_grand_staff(token_strs: List[str]) -> bool:
    return 'staff-bass' in token_strs


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
                 augment: bool = True, patch_size: int = PATCH_SIZE):
        self.data_dir   = data_dir
        self.n_patches  = patches_per_image
        self.augment    = augment
        self.patch_size = patch_size
        self.image_paths: List[str] = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and not fname.endswith('_seg.png'):
                self.image_paths.append(os.path.join(data_dir, fname))
        self._len = len(self.image_paths) * patches_per_image
        print(f"[SegnetDataset] {len(self.image_paths)} × {patches_per_image} = {self._len}")

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
                 max_seq: int = 512, augment: bool = True):
        self.max_seq = max_seq
        self.augment = augment
        tok2id = tokenizer
        id2tok = {v: k for k, v in tok2id.items()}
        self._tok2id = tok2id
        self._id2tok = id2tok

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
            is_grand = _has_grand_staff(token_strs)

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

            n_staffs = len(staffs)

            if is_grand and n_staffs >= 2 and n_staffs % 2 == 0:
                # 대보표: 짝수 오선 → 쌍으로 분리
                n_systems = n_staffs // 2
                system_tokens = _split_grand_staff_tokens(ids, n_systems, id2tok, tok2id)

                for sys_i, (treble_ids, bass_ids) in enumerate(system_tokens):
                    treble_staff = staffs[sys_i * 2]
                    bass_staff   = staffs[sys_i * 2 + 1]
                    if treble_ids:   # 분리 실패 샘플 제외
                        self.samples.append((img_path, treble_ids[:max_seq], treble_staff))
                    if bass_ids:
                        self.samples.append((img_path, bass_ids[:max_seq], bass_staff))
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
        bgr  = cv2.imread(img_path)
        gray = preprocess(bgr)
        if self.augment:
            gray = augment_image(gray)

        tile     = extract_staff_canvas(gray, staff)
        canvas   = (tile.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
        canvas_t = torch.from_numpy(canvas).unsqueeze(0)

        tgt_in  = [SOS_ID] + ids
        tgt_out = ids + [EOS_ID]
        return canvas_t, torch.tensor(tgt_in, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  Collate / helpers
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
