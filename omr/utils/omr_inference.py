"""
omr_inference.py  –  Python TFLite inference pipeline for evaluation.

Mirrors the C++ engine pipeline so that model accuracy can be measured
during training (before the C++ binary is built).

Pipeline stages:
  1. Preprocess  : BGR -> gray -> autocrop -> resize(1920) -> CLAHE -> bilateral
  2. SegNet      : sliding-window TFLite inference -> 6-class probability maps
  3. StaffDetect : horizontal projection -> peak NMS -> 5-line grouping
  4. Canvas      : crop + scale -> 256×1280 tiles with 64px overlap
  5. Encoder     : tile -> [seq_len × 512] float32 (TFLite)
  6. Decoder     : greedy autoregressive with KV-cache (TFLite)
  7. Tokens      : ID sequence -> token strings (via tokenizer.json)

Decoder TFLite input tensor order (must match the trained model):
  [0]    token_in   : int32  [1, 1]
  [1]    context    : float32 [1, seq_len, 512]  (seq_len=1 for steps 1+)
  [2]    cache_len  : int32  [1]
  [3..34] kv_in_i  : float32 [1, NUM_HEADS, step, HEAD_DIM]

Decoder TFLite output tensor order:
  [0]    logits     : float32 [1, 1, vocab_size]
  [1..32] kv_out_i  : float32 [1, NUM_HEADS, step+1, HEAD_DIM]
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  TFLite interpreter import (supports both tflite-runtime and tensorflow)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as _tflite
    Interpreter = _tflite.Interpreter
except ImportError:
    import tensorflow.lite as _tflite
    Interpreter = _tflite.Interpreter


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StaffGroup:
    y_lines:   List[float]   # [y0, y1, y2, y3, y4] top→bottom
    unit_size: float         # mean inter-line spacing (px)


# ─────────────────────────────────────────────────────────────────────────────
#  Main inference class
# ─────────────────────────────────────────────────────────────────────────────

class OmrInference:
    # ── Segnet constants (match C++ SegnetRunner) ─────────────────────────────
    PATCH_SIZE    = 320
    STRIDE        = 160       # 50 % overlap
    NUM_CLASSES   = 6
    SEG_STAFF_IDX = 3         # SEG_STAFF_LINE - 1 = index 3 in seg_maps list
    IMG_MEAN      = 0.7931
    IMG_STD       = 0.1738

    # ── Staff detection constants (match C++ StaffDetector) ───────────────────
    MIN_UNIT      = 11.0
    MAX_UNIT      = 60.0
    PEAK_THRESH   = 0.05

    # ── Canvas constants (match C++ StaffCanvas) ──────────────────────────────
    CANVAS_H      = 256
    CANVAS_W      = 1280
    TILE_OVERLAP  = 64
    MARGIN_UNITS  = 2.0

    # ── Encoder normalisation (match C++ EncoderRunner) ───────────────────────
    ENC_MEAN      = 0.7931
    ENC_STD       = 0.1738

    # ── Decoder constants (match C++ DecoderRunner) ───────────────────────────
    PAD_ID        = 0
    SOS_ID        = 1
    EOS_ID        = 2
    MAX_SEQ       = 608
    NUM_KV        = 32        # 8 Transformer layers × 2 (key + value) × 2 directions
    NUM_HEADS     = 8
    HEAD_DIM      = 64

    # ── Preprocessor ─────────────────────────────────────────────────────────
    TARGET_WIDTH  = 1920

    def __init__(self,
                 segnet_path:   str,
                 encoder_path:  str,
                 decoder_path:  str,
                 tokenizer_path: str,
                 num_threads:   int = 4):
        """Load all three TFLite models and the tokenizer vocabulary."""

        def _load(path: str) -> Interpreter:
            interp = Interpreter(model_path=path,
                                 num_threads=num_threads)
            interp.allocate_tensors()
            return interp

        self._seg_interp = _load(segnet_path)
        self._enc_interp = _load(encoder_path)
        # Decoder does NOT allocate here – shapes change each step.
        self._dec_interp = Interpreter(model_path=decoder_path,
                                       num_threads=num_threads)

        with open(tokenizer_path, encoding='utf-8') as f:
            tok: dict = json.load(f)
        self._token_to_id: dict[str, int] = tok
        self._id_to_token: dict[int, str] = {v: k for k, v in tok.items()}

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, image_path: str) -> List[str]:
        """
        Run the full OMR pipeline on one image.

        Returns a list of token strings, e.g.:
          ['clef-G', 'key-C', 'time-4/4', 'note-C4-1/4', 'barline', ...]
        Returns an empty list if no staff lines are found.
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        gray      = self._preprocess(bgr)
        seg_maps  = self._run_segnet(gray)
        staffs    = self._detect_staffs(seg_maps[self.SEG_STAFF_IDX])

        if not staffs:
            return []

        tokens: List[str] = []
        for idx, staff in enumerate(staffs):
            tiles = self._build_tiles(gray, staff)
            for tile in tiles:
                enc_out = self._run_encoder(tile)
                ids     = self._run_decoder(enc_out)
                for tok_id in ids:
                    if tok_id not in (self.PAD_ID, self.SOS_ID, self.EOS_ID):
                        tokens.append(self._id_to_token.get(tok_id, '<UNK>'))

            if idx < len(staffs) - 1:
                tokens.append('barline')

        return tokens

    # ─────────────────────────────────────────────────────────────────────────
    #  Stage 1: preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """BGR -> preprocessed grayscale (mirrors C++ Preprocessor::process)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
        gray = self._autocrop(gray)
        gray = self._resize_to_width(gray)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray  = cv2.bilateralFilter(gray, 9, 20.0, 7.0)
        return gray

    def _autocrop(self, gray: np.ndarray) -> np.ndarray:
        inv = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if w * h > 0.80 * gray.shape[0] * gray.shape[1]:
            return gray
        pad_x = max(2, gray.shape[1] // 100)
        pad_y = max(2, gray.shape[0] // 100)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray.shape[1], x + w + pad_x)
        y2 = min(gray.shape[0], y + h + pad_y)
        return gray[y1:y2, x1:x2]

    def _resize_to_width(self, gray: np.ndarray, target: int = TARGET_WIDTH) -> np.ndarray:
        H, W = gray.shape
        if W == target:
            return gray
        scale = target / W
        new_h = int(round(H * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(gray, (target, new_h), interpolation=interp)

    # ─────────────────────────────────────────────────────────────────────────
    #  Stage 2: segmentation (sliding window)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_segnet(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Returns 6 probability maps (one per SegClass), each shape [H, W] float32.
        Uses linear-weight blending across overlapping patches.
        """
        H, W  = gray.shape
        P, S  = self.PATCH_SIZE, self.STRIDE

        accum      = np.zeros((self.NUM_CLASSES, H, W), dtype=np.float64)
        weight_map = np.zeros((H, W), dtype=np.float64)

        # 2-D triangular weight for one patch: linearly decreases from centre.
        wy = 1.0 - np.abs(np.arange(P) - P / 2.0) / (P / 2.0)
        wx = 1.0 - np.abs(np.arange(P) - P / 2.0) / (P / 2.0)
        patch_w = np.outer(wy, wx).astype(np.float32)  # [P, P]

        seg_in  = self._seg_interp.get_input_details()
        seg_out = self._seg_interp.get_output_details()

        for y0 in range(0, H, S):
            for x0 in range(0, W, S):
                y1 = min(y0 + P, H)
                x1 = min(x0 + P, W)
                ph = y1 - y0
                pw = x1 - x0

                # Extract patch; pad with white (255) on the right / bottom.
                patch = np.full((P, P), 255, dtype=np.uint8)
                patch[:ph, :pw] = gray[y0:y1, x0:x1]

                # Normalise: (pixel/255 - mean) / std
                patch_f = (patch.astype(np.float32) / 255.0
                           - self.IMG_MEAN) / self.IMG_STD

                # Build NCHW input [1, 3, P, P] (replicate grayscale 3×).
                inp = np.stack([patch_f, patch_f, patch_f], axis=0)[None]
                self._seg_interp.set_tensor(seg_in[0]['index'],
                                            inp.astype(np.float32))
                self._seg_interp.invoke()

                # Output: [1, NUM_CLASSES, P, P] logits.
                logits = self._seg_interp.get_tensor(
                    seg_out[0]['index'])[0].astype(np.float64)  # [C, P, P]

                # Softmax along class axis.
                logits -= logits.max(axis=0, keepdims=True)
                exp_l   = np.exp(logits)
                probs   = exp_l / exp_l.sum(axis=0, keepdims=True)  # [C, P, P]

                # Linear-weight blend into accumulator.
                w2d = patch_w[:ph, :pw]
                accum[:, y0:y1, x0:x1]  += probs[:, :ph, :pw] * w2d[None]
                weight_map[y0:y1, x0:x1] += w2d

        # Normalise by accumulated weights.
        mask = weight_map > 0
        for c in range(self.NUM_CLASSES):
            accum[c, mask] /= weight_map[mask]

        return [accum[c].astype(np.float32) for c in range(self.NUM_CLASSES)]

    # ─────────────────────────────────────────────────────────────────────────
    #  Stage 3: staff line detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_staffs(self, staff_mask: np.ndarray) -> List[StaffGroup]:
        """Mirrors C++ StaffDetector::detect()."""
        H, W = staff_mask.shape

        # Horizontal projection: mean across each row.
        proj = staff_mask.mean(axis=1)  # [H]

        # Local-maxima candidates above threshold.
        candidates: List[Tuple[float, int]] = []
        for i in range(1, H - 1):
            if (proj[i] > self.PEAK_THRESH
                    and proj[i] >= proj[i - 1]
                    and proj[i] >= proj[i + 1]):
                candidates.append((float(proj[i]), i))

        # Non-maximum suppression with min_dist = MIN_UNIT / 2.
        candidates.sort(reverse=True)
        suppressed = np.zeros(H, dtype=bool)
        peaks: List[int] = []
        min_dist = self.MIN_UNIT / 2.0
        for val, row in candidates:
            if suppressed[row]:
                continue
            peaks.append(row)
            lo = max(0, int(row - min_dist))
            hi = min(H - 1, int(row + min_dist))
            suppressed[lo:hi + 1] = True

        peaks.sort()
        if len(peaks) < 5:
            return []

        # Group consecutive peaks into sets of 5 with uniform spacing.
        groups: List[List[int]] = []
        i = 0
        while i + 4 < len(peaks):
            gaps = [peaks[i + k + 1] - peaks[i + k] for k in range(4)]
            if all(self.MIN_UNIT <= g <= self.MAX_UNIT for g in gaps):
                groups.append(peaks[i:i + 5])
                i += 5
            else:
                i += 1

        # Build StaffGroup objects; reject if std-dev of gaps is too large.
        staffs: List[StaffGroup] = []
        for g in groups:
            gaps = [float(g[k + 1] - g[k]) for k in range(4)]
            unit = sum(gaps) / 4.0
            var  = sum((d - unit) ** 2 for d in gaps) / 4.0
            if (var ** 0.5) > 0.6 * unit:
                continue
            staffs.append(StaffGroup(
                y_lines=[float(y) for y in g],
                unit_size=unit))

        return staffs

    # ─────────────────────────────────────────────────────────────────────────
    #  Stage 4: canvas tiling
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tiles(self, gray: np.ndarray,
                     staff: StaffGroup) -> List[np.ndarray]:
        """Mirrors C++ StaffCanvas::build_tiles()."""
        H, W = gray.shape
        margin = staff.unit_size * self.MARGIN_UNITS

        y_top = max(0, int(staff.y_lines[0] - margin))
        y_bot = min(H - 1, int(staff.y_lines[4] + margin))
        strip = gray[y_top:y_bot + 1, :]  # [strip_h, W]

        # Scale height to CANVAS_H.
        scale  = self.CANVAS_H / strip.shape[0]
        new_w  = int(round(strip.shape[1] * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        strip  = cv2.resize(strip, (new_w, self.CANVAS_H), interpolation=interp)

        # Tile horizontally with TILE_OVERLAP overlap.
        tiles: List[np.ndarray] = []
        step = self.CANVAS_W - self.TILE_OVERLAP
        x    = 0
        SW   = strip.shape[1]
        while True:
            tile = np.full((self.CANVAS_H, self.CANVAS_W), 255, dtype=np.uint8)
            copy_w = min(self.CANVAS_W, SW - x)
            tile[:, :copy_w] = strip[:, x:x + copy_w]
            tiles.append(tile)
            if x + self.CANVAS_W >= SW:
                break
            x += step

        if not tiles:   # degenerate short strip
            tile = np.full((self.CANVAS_H, self.CANVAS_W), 255, dtype=np.uint8)
            copy_w = min(self.CANVAS_W, SW)
            tile[:, :copy_w] = strip[:, :copy_w]
            tiles.append(tile)

        return tiles

    # ─────────────────────────────────────────────────────────────────────────
    #  Stage 5: encoder
    # ─────────────────────────────────────────────────────────────────────────

    def _run_encoder(self, tile: np.ndarray) -> np.ndarray:
        """
        tile: uint8 [CANVAS_H, CANVAS_W]
        Returns: float32 [seq_len, 512]
        """
        # Normalise.
        tile_f = (tile.astype(np.float32) / 255.0
                  - self.ENC_MEAN) / self.ENC_STD  # [H, W]

        # Encoder expects NCHW [1, 1, H, W].
        inp = tile_f[None, None, :, :]  # [1, 1, H, W]

        enc_in  = self._enc_interp.get_input_details()
        enc_out = self._enc_interp.get_output_details()

        self._enc_interp.set_tensor(enc_in[0]['index'],
                                    inp.astype(np.float32))
        self._enc_interp.invoke()

        out = self._enc_interp.get_tensor(enc_out[0]['index'])  # [1, seq_len, 512]
        return out[0]  # [seq_len, 512]

    # ─────────────────────────────────────────────────────────────────────────
    #  Stage 6: greedy decoder with KV-cache
    #
    #  Mirrors C++ DecoderRunner::decode() exactly.
    #  At step 0: pass full encoder output as context.
    #  At steps 1+: pass only the first encoder vector (context[:1, :]).
    # ─────────────────────────────────────────────────────────────────────────

    def _run_decoder(self, encoder_out: np.ndarray) -> List[int]:
        """
        encoder_out: float32 [seq_len, 512]
        Returns: list of token IDs (excluding SOS/EOS/PAD).
        """
        interp = self._dec_interp
        result: List[int] = []
        prev_token = self.SOS_ID

        # KV-cache: list of NUM_KV arrays, each grows by 1 along axis 2 per step.
        kv_cache: List[Optional[np.ndarray]] = [None] * self.NUM_KV

        for step in range(self.MAX_SEQ):
            # Choose context shape.
            if step == 0:
                context = encoder_out[None, :, :]        # [1, seq_len, 512]
            else:
                context = encoder_out[None, :1, :]       # [1, 1, 512]

            # ── Resize dynamic-shape inputs before allocating ─────────────────
            # Input 0: token_in [1, 1]
            interp.resize_tensor_input(
                interp.get_input_details()[0]['index'], [1, 1])
            # Input 1: context [1, ?, 512]
            interp.resize_tensor_input(
                interp.get_input_details()[1]['index'],
                [1, context.shape[1], 512])
            # Input 2: cache_len [1]
            interp.resize_tensor_input(
                interp.get_input_details()[2]['index'], [1])
            # Inputs 3..34: kv_in_i [1, NUM_HEADS, step, HEAD_DIM]
            for i in range(self.NUM_KV):
                interp.resize_tensor_input(
                    interp.get_input_details()[3 + i]['index'],
                    [1, self.NUM_HEADS, step, self.HEAD_DIM])

            interp.allocate_tensors()

            # ── Set input values ─────────────────────────────────────────────
            inp_details = interp.get_input_details()

            # token_in
            interp.set_tensor(inp_details[0]['index'],
                               np.array([[prev_token]], dtype=np.int32))
            # context
            interp.set_tensor(inp_details[1]['index'],
                               context.astype(np.float32))
            # cache_len
            interp.set_tensor(inp_details[2]['index'],
                               np.array([step], dtype=np.int32))
            # kv_in_i
            for i in range(self.NUM_KV):
                if kv_cache[i] is not None:
                    interp.set_tensor(inp_details[3 + i]['index'],
                                      kv_cache[i])
                else:
                    # Step 0: empty cache (zero values).
                    interp.set_tensor(
                        inp_details[3 + i]['index'],
                        np.zeros((1, self.NUM_HEADS, 0, self.HEAD_DIM),
                                 dtype=np.float32))

            interp.invoke()

            # ── Read outputs ─────────────────────────────────────────────────
            out_details = interp.get_output_details()

            # logits: [1, 1, vocab_size]
            logits   = interp.get_tensor(out_details[0]['index'])
            next_id  = int(np.argmax(logits[0, 0]))

            # kv_out: [1, NUM_HEADS, step+1, HEAD_DIM]
            for i in range(self.NUM_KV):
                kv_cache[i] = interp.get_tensor(
                    out_details[1 + i]['index']).copy()

            if next_id == self.EOS_ID:
                break
            if next_id == self.PAD_ID:
                continue

            result.append(next_id)
            prev_token = next_id

        return result
