"""
dataset.py  –  PyTorch datasets for OMR model training.

Two dataset classes:

1. SegnetDataset
   Input  : random 320×320 patches from training images
   Target : 6-class pixel segmentation labels (weak, image-processing based)
   Used for Phase 1 (SegNet pre-training).

   Label generation (no manual annotation required):
     - Class 4 (SEG_STAFF_LINE): thin horizontal dark bands detected via
       morphological opening with a wide horizontal kernel.
     - Class 2 (SEG_NOTEHEAD): small compact blobs after staff-line removal,
       filtered by area and aspect ratio.
     - Class 1 (SEG_STEM_REST): thin vertical strokes.
     - Class 5 (SEG_SYMBOL): remaining dark foreground pixels.
     - Class 0 (SEG_BACKGROUND): white background.
   These are "weak labels" – noisy but sufficient to bootstrap SegNet.

2. OMRDataset
   Input  : 256×1280 staff-strip canvas tiles (uint8, normalised to float32)
   Target : token-ID sequences from data/train/*.json
   Used for Phase 2/3 (Encoder-Decoder training).

   Pipeline per sample:
     1. Load PNG → preprocess (autocrop, resize to 1920px, CLAHE, bilateral)
     2. Detect staff lines (horizontal-projection peak-finding)
     3. Crop staff region → scale to 256px height → tile at 1280px width
     4. Load JSON token sequence → convert to ID tensor
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
#  Shared constants (must match model.py and C++ engine)
# ─────────────────────────────────────────────────────────────────────────────

PAD_ID      = 0
SOS_ID      = 1
EOS_ID      = 2

PATCH_SIZE  = 320       # SegNet input patch (square)
CANVAS_H    = 256       # Encoder canvas height
CANVAS_W    = 1280      # Encoder canvas width
TILE_OVERLAP = 64       # Horizontal tile overlap (px)
MARGIN_UNITS = 2.0      # Staff-region vertical margin (in unit_size units)
TARGET_W    = 1920      # Preprocessor target width
IMG_MEAN    = 0.7931    # Encoder normalisation mean
IMG_STD     = 0.1738    # Encoder normalisation std

# SegNet class indices
SEG_BG         = 0
SEG_STEM_REST  = 1
SEG_NOTEHEAD   = 2
SEG_CLEF_KEY   = 3
SEG_STAFF_LINE = 4
SEG_SYMBOL     = 5
SEG_NUM_CLS    = 6

# Staff detection
MIN_UNIT   = 11.0
MAX_UNIT   = 60.0


# ─────────────────────────────────────────────────────────────────────────────
#  Shared preprocessing (mirrors C++ Preprocessor)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR → preprocessed grayscale (same pipeline as C++ Preprocessor)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()

    # autocrop
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

    # resize to TARGET_W
    H, W = gray.shape
    if W != TARGET_W:
        scale = TARGET_W / W
        gray = cv2.resize(gray, (TARGET_W, int(round(H * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # bilateral denoising
    gray = cv2.bilateralFilter(gray, 9, 20.0, 7.0)
    return gray


def detect_staffs(gray: np.ndarray) -> List[Dict]:
    """
    Rule-based staff detection via horizontal projection.
    Returns list of {'y_lines': [y0..y4], 'unit_size': float}.
    Mirrors C++ StaffDetector::detect() (no TFLite required).
    """
    # Binarise: dark pixels = likely music content.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Horizontal projection: keep only long horizontal runs (staff lines).
    H, W = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 8, 1))
    staff_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    proj = staff_binary.mean(axis=1) / 255.0  # [H], normalised to [0,1]

    # Peak finding (local maxima > threshold, NMS with min_dist = MIN_UNIT/2).
    THRESH  = 0.05
    candidates = [(float(proj[i]), i) for i in range(1, H - 1)
                  if proj[i] > THRESH
                  and proj[i] >= proj[i - 1]
                  and proj[i] >= proj[i + 1]]
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

    # Group 5 consecutive peaks with consistent spacing.
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


def extract_canvas_tiles(gray: np.ndarray, staff: Dict) -> List[np.ndarray]:
    """
    Crop staff region → scale to CANVAS_H → tile into CANVAS_W strips.
    Returns list of uint8 [CANVAS_H × CANVAS_W] arrays.
    """
    H, W = gray.shape
    margin = staff['unit_size'] * MARGIN_UNITS
    y_top  = max(0, int(staff['y_lines'][0] - margin))
    y_bot  = min(H - 1, int(staff['y_lines'][4] + margin))

    strip = gray[y_top:y_bot + 1, :]
    scale = CANVAS_H / strip.shape[0]
    new_w = int(round(strip.shape[1] * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    strip  = cv2.resize(strip, (new_w, CANVAS_H), interpolation=interp)

    tiles = []
    step  = CANVAS_W - TILE_OVERLAP
    x     = 0
    SW    = strip.shape[1]
    while True:
        tile = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        cw   = min(CANVAS_W, SW - x)
        tile[:, :cw] = strip[:, x:x + cw]
        tiles.append(tile)
        if x + CANVAS_W >= SW:
            break
        x += step

    if not tiles:
        tile = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        cw   = min(CANVAS_W, SW)
        tile[:, :cw] = strip[:, :cw]
        tiles.append(tile)

    return tiles


# ─────────────────────────────────────────────────────────────────────────────
#  Segnet weak-label generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_weak_seg_labels(gray: np.ndarray) -> np.ndarray:
    """
    Generate approximate 6-class pixel segmentation labels from a grayscale
    sheet-music image using image processing only (no manual annotation).

    Quality: sufficient for Phase 1 pre-training to bootstrap SegNet.
    For higher accuracy, replace with manually annotated or SVG-derived labels.

    Returns: int64 ndarray [H, W] with values in {0, 1, 2, 4, 5}.
    """
    H, W = gray.shape
    labels = np.zeros((H, W), dtype=np.int64)

    # Binarise (dark ink on white paper).
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # ── Class 4: Staff lines ──────────────────────────────────────────────────
    # Morphological opening with a wide horizontal kernel isolates long thin
    # horizontal structures (staff lines) while removing noteheads/stems.
    h_len   = max(W // 8, 60)
    h_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    staff   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kern)
    # Dilate to cover the full staff line width (≈2-4 px).
    d_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    staff   = cv2.dilate(staff, d_kern)
    labels[staff > 0] = SEG_STAFF_LINE

    # Symbols = all dark pixels minus staff lines.
    symbols = cv2.subtract(binary, staff)

    # ── Class 1: Stems ────────────────────────────────────────────────────────
    # Thin, tall vertical strokes.
    v_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    stems   = cv2.morphologyEx(symbols, cv2.MORPH_OPEN, v_kern)
    labels[(stems > 0) & (labels == SEG_BG)] = SEG_STEM_REST

    # ── Class 2: Noteheads ────────────────────────────────────────────────────
    # Compact, roughly circular blobs.  Detect via connected components
    # after removing stems.
    no_stems = cv2.subtract(symbols, stems)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(no_stems, connectivity=8)
    for i in range(1, n):
        area  = int(stats[i, cv2.CC_STAT_AREA])
        bw    = int(stats[i, cv2.CC_STAT_WIDTH])
        bh    = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bh < 1:
            continue
        ratio = bw / bh
        # Noteheads: area 15..600 px, aspect ratio close to 1.
        if 15 <= area <= 600 and 0.4 <= ratio <= 2.5:
            labels[comp == i] = SEG_NOTEHEAD

    # ── Class 5: Other symbols ────────────────────────────────────────────────
    # Anything dark that is not yet labelled (clefs, accidentals, bar lines, etc.).
    unlabelled = (binary > 0) & (labels == SEG_BG)
    labels[unlabelled] = SEG_SYMBOL

    return labels  # [H, W]


# ─────────────────────────────────────────────────────────────────────────────
#  Data augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(gray: np.ndarray,
                  brightness:  bool = True,
                  noise:       bool = True,
                  blur:        bool = True,
                  perspective: bool = False) -> np.ndarray:
    """Light augmentations that preserve music content structure."""
    out = gray.copy()

    if brightness and random.random() < 0.6:
        beta   = random.uniform(-25, 25)          # brightness offset
        alpha  = random.uniform(0.85, 1.15)       # contrast scale
        out    = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if noise and random.random() < 0.4:
        sigma = random.uniform(2.0, 8.0)
        noise_arr = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise_arr, 0, 255).astype(np.uint8)

    if blur and random.random() < 0.3:
        ksize = random.choice([3, 5])
        out   = cv2.GaussianBlur(out, (ksize, ksize), 0)

    if perspective and random.random() < 0.3:
        H, W = out.shape
        margin = int(min(H, W) * 0.03)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        def r(): return random.randint(-margin, margin)
        dst = np.float32([[r(), r()], [W + r(), r()], [W + r(), H + r()], [r(), H + r()]])
        M   = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M, (W, H),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  1. SegnetDataset
# ─────────────────────────────────────────────────────────────────────────────

class SegnetDataset(Dataset):
    """
    Random 320×320 patch crops from training images with weak pixel labels.

    Each call to __getitem__ returns:
      image : float32 tensor [1, 320, 320]  normalised to [-1, 1]
      label : int64  tensor [320, 320]      class index {0,1,2,3,4,5}

    Label files: if '<stem>_seg.png' exists in data_dir, it is loaded
    directly (each pixel = class index 0-5).  Otherwise, labels are
    generated on-the-fly using generate_weak_seg_labels().
    """

    def __init__(self,
                 data_dir:    str,
                 patches_per_image: int = 8,
                 augment:     bool = True,
                 patch_size:  int  = PATCH_SIZE):
        self.data_dir   = data_dir
        self.n_patches  = patches_per_image
        self.augment    = augment
        self.patch_size = patch_size

        # Collect all image paths.
        self.image_paths: List[str] = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                if not fname.endswith('_seg.png'):
                    self.image_paths.append(os.path.join(data_dir, fname))

        # Each image contributes n_patches samples.
        self._len = len(self.image_paths) * patches_per_image
        print(f"[SegnetDataset] {len(self.image_paths)} images × "
              f"{patches_per_image} patches = {self._len} samples")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx  = idx // self.n_patches
        img_path = self.image_paths[img_idx]

        # Load and preprocess.
        bgr  = cv2.imread(img_path)
        gray = preprocess(bgr)

        if self.augment:
            gray = augment_image(gray, perspective=True)

        # Load or generate segmentation labels.
        stem    = os.path.splitext(img_path)[0]
        lab_path = stem + '_seg.png'
        if os.path.isfile(lab_path):
            label = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
        else:
            label = generate_weak_seg_labels(gray)

        # Ensure label and image have the same shape.
        if label.shape != gray.shape:
            label = cv2.resize(label.astype(np.uint8),
                               (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(np.int64)

        # Extract a random PATCH_SIZE×PATCH_SIZE crop.
        P  = self.patch_size
        H, W = gray.shape
        if H < P or W < P:
            gray  = cv2.copyMakeBorder(gray,  0, max(0, P - H), 0, max(0, P - W),
                                        cv2.BORDER_CONSTANT, value=255)
            label = cv2.copyMakeBorder(label.astype(np.uint8),
                                        0, max(0, P - H), 0, max(0, P - W),
                                        cv2.BORDER_CONSTANT, value=0).astype(np.int64)
            H, W = gray.shape

        y0 = random.randint(0, H - P)
        x0 = random.randint(0, W - P)
        patch = gray[y0:y0 + P, x0:x0 + P]
        lbl   = label[y0:y0 + P, x0:x0 + P]

        # Normalise image to [-1, 1].
        img_t = torch.from_numpy(patch.astype(np.float32) / 127.5 - 1.0)
        img_t = img_t.unsqueeze(0)          # [1, P, P]
        lbl_t = torch.from_numpy(lbl.copy())  # [P, P]  int64

        return img_t, lbl_t


# ─────────────────────────────────────────────────────────────────────────────
#  2. OMRDataset (Encoder-Decoder training)
# ─────────────────────────────────────────────────────────────────────────────

class OMRDataset(Dataset):
    """
    Staff-strip canvas tiles paired with ground-truth token sequences.

    Each call to __getitem__ returns:
      canvas  : float32 tensor [1, CANVAS_H, CANVAS_W]  (normalised)
      tgt_in  : int64  tensor [T]   decoder input  = [SOS, t1, ..., t_{n-1}]
      tgt_out : int64  tensor [T]   decoder target = [t1,  ..., t_n,  EOS ]

    If an image has multiple staff groups, each group is registered as a
    separate sample (all sharing the same token sequence for simplicity).
    Samples with no detectable staff are skipped during dataset construction.
    """

    def __init__(self,
                 data_dir:     str,
                 tokenizer:    Dict[str, int],
                 max_seq:      int  = 512,
                 augment:      bool = True,
                 tiles_per_staff: int = 1):
        """
        Args:
            data_dir       : directory containing *.png + *.json pairs
            tokenizer      : dict mapping token string → int ID
            max_seq        : maximum token sequence length (longer = truncated)
            augment        : apply photometric augmentation to canvas tiles
            tiles_per_staff: how many tiles per staff to include (1 = first only)
        """
        self.tokenizer    = tokenizer
        self.max_seq      = max_seq
        self.augment      = augment
        self.tiles_per_staff = tiles_per_staff

        self.samples: List[Tuple[str, List[int], int, int]] = []
        # Each entry: (image_path, token_ids, staff_index, tile_index)

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

            # Load token IDs.
            try:
                with open(gt_path, encoding='utf-8') as f:
                    data = json.load(f)
                token_strs = data.get('tokens', [])
            except (json.JSONDecodeError, KeyError):
                skipped += 1
                continue

            if not token_strs:
                skipped += 1
                continue

            ids = [tokenizer.get(t, tokenizer.get('<UNK>', 3))
                   for t in token_strs]
            ids = ids[:max_seq]   # truncate if needed

            # Pre-detect staffs to count samples; actual extraction is in __getitem__.
            try:
                bgr   = cv2.imread(img_path)
                gray  = preprocess(bgr)
                staffs = detect_staffs(gray)
            except Exception:
                skipped += 1
                continue

            if not staffs:
                skipped += 1
                continue

            for si in range(len(staffs)):
                for ti in range(tiles_per_staff):
                    self.samples.append((img_path, ids, si, ti))

        print(f"[OMRDataset] {len(self.samples)} canvas-tile samples "
              f"({skipped} images skipped — no staff or label)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, ids, staff_idx, tile_idx = self.samples[idx]

        # Load + preprocess image.
        bgr   = cv2.imread(img_path)
        gray  = preprocess(bgr)

        if self.augment:
            gray = augment_image(gray, perspective=False)

        # Extract staff tiles.
        staffs = detect_staffs(gray)
        if staff_idx >= len(staffs):
            staff_idx = 0   # fallback

        tiles = extract_canvas_tiles(gray, staffs[staff_idx])
        tile_idx = min(tile_idx, len(tiles) - 1)
        tile = tiles[tile_idx]  # uint8 [CANVAS_H, CANVAS_W]

        # Normalise to float32: (pixel/255 − mean) / std
        canvas = ((tile.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD)
        canvas_t = torch.from_numpy(canvas).unsqueeze(0)  # [1, H, W]

        # Build decoder input / target.
        # tgt_in  = [SOS] + ids
        # tgt_out = ids   + [EOS]
        tgt_in  = [SOS_ID] + ids
        tgt_out = ids + [EOS_ID]

        tgt_in_t  = torch.tensor(tgt_in,  dtype=torch.long)
        tgt_out_t = torch.tensor(tgt_out, dtype=torch.long)

        return canvas_t, tgt_in_t, tgt_out_t


# ─────────────────────────────────────────────────────────────────────────────
#  Collate function (pads variable-length sequences to max in batch)
# ─────────────────────────────────────────────────────────────────────────────

def omr_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Pads tgt_in / tgt_out to the same length within the batch.

    Returns:
      canvases         : float32 [B, 1, H, W]
      tgt_in_padded    : int64   [B, T]   (padded with PAD_ID)
      tgt_out_padded   : int64   [B, T]   (padded with PAD_ID)
      tgt_key_pad_mask : bool    [B, T]   True where padding
    """
    canvases, tgt_ins, tgt_outs = zip(*batch)

    canvases = torch.stack(canvases, dim=0)  # [B, 1, H, W]

    max_T = max(t.size(0) for t in tgt_ins)

    tgt_in_padded  = torch.full((len(tgt_ins),  max_T), PAD_ID, dtype=torch.long)
    tgt_out_padded = torch.full((len(tgt_outs), max_T), PAD_ID, dtype=torch.long)
    tgt_key_pad_mask = torch.ones(len(tgt_ins), max_T, dtype=torch.bool)  # True = padding

    for i, (ti, to) in enumerate(zip(tgt_ins, tgt_outs)):
        L = ti.size(0)
        tgt_in_padded[i, :L]   = ti
        tgt_out_padded[i, :L]  = to
        tgt_key_pad_mask[i, :L] = False   # real tokens → False (not masked)

    return canvases, tgt_in_padded, tgt_out_padded, tgt_key_pad_mask


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer loader helper
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load tokenizer.json and return (token→id, id→token) dicts."""
    with open(path, encoding='utf-8') as f:
        tok2id: Dict[str, int] = json.load(f)
    id2tok = {v: k for k, v in tok2id.items()}
    return tok2id, id2tok


# ─────────────────────────────────────────────────────────────────────────────
#  Train / validation split utility
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(dataset: Dataset,
                  val_ratio: float = 0.1,
                  seed: int = 42
                  ) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and validation subsets."""
    from torch.utils.data import Subset
    n     = len(dataset)
    n_val = max(1, int(n * val_ratio))
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n).tolist()
    return Subset(dataset, perm[n_val:]), Subset(dataset, perm[:n_val])
