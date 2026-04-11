"""
export_tflite.py  —  Export trained OMR PyTorch models to TFLite INT8.

Conversion pipeline for each model:
    PyTorch .pt  →  ONNX  →  (onnx2tf)  →  TFLite .tflite

Version tagging
───────────────
Every export saves two copies:
    assets/segnet_INT8_v2.tflite   ← versioned (permanent, never overwritten)
    assets/segnet_INT8.tflite      ← latest  (overwritten on each export)

Pass --version to set the tag.  Keep old versioned files as rollback snapshots.

Fine-tuning learning rate note
───────────────────────────────
When you continue training from an existing checkpoint (--resume), the learning
rate should be LOWER than the initial run — typically 3–10× smaller.

    Initial Phase 2 run : --lr 1e-4
    Fine-tuning run     : --lr 3e-5  (roughly 3× smaller)

Why?  The checkpoint already sits near a loss minimum.  Using the same lr as
the first run risks overshooting that minimum and overwriting representations
that took many epochs to build (this is sometimes called "catastrophic
forgetting of the optimisation trajectory").  A smaller lr lets the model
absorb new data by taking small, careful gradient steps.

Dependencies (install once, separate from training deps):
    pip install onnx onnxsim onnx2tf tensorflow opencv-python

Usage
─────
    python omr/training/export_tflite.py \\
      --segnet    models/segnet_best.pt \\
      --seq2seq   models/seq2seq_best.pt \\
      --data_dir  data/train \\
      --tokenizer data/tokenizer.json \\
      --out_dir   assets/ \\
      --version   v1

Output:
    assets/segnet_INT8_v1.tflite
    assets/encoder_INT8_v1.tflite
    assets/decoder_INT8_v1.tflite   (FP32 — see note below)
    assets/tokenizer_v1.json
    assets/segnet_INT8.tflite       (latest — identical to v1 until next export)
    assets/encoder_INT8.tflite
    assets/decoder_INT8.tflite
    assets/tokenizer.json

Decoder quantization note
──────────────────────────
SegNet and Encoder are quantized to INT8 (no accuracy loss in practice).
The Decoder is exported as FP32 by default because INT8 quantization of
autoregressive Transformer decoders often causes significant token-error
degradation.  Pass --quantize_decoder to enable decoder INT8 as an experiment.
"""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

# Import model definitions from the same directory.
sys.path.insert(0, os.path.dirname(__file__))
from model import (
    SegNet, OmrSeq2Seq,
    CANVAS_H, CANVAS_W, SEQ_LEN, VOCAB_SIZE,
    NUM_CLASSES, SOS_ID,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_state(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt.get('model', ckpt)


def load_segnet(ckpt_path: str, device: torch.device) -> SegNet:
    model = SegNet(num_classes=NUM_CLASSES)
    model.load_state_dict(_load_state(ckpt_path, device))
    return model.eval().to(device)


def load_seq2seq(ckpt_path: str, device: torch.device) -> OmrSeq2Seq:
    model = OmrSeq2Seq()
    model.load_state_dict(_load_state(ckpt_path, device))
    return model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration data builders (for INT8 PTQ)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_pngs(data_dir: str, n: int) -> list:
    paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))[:n]
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {data_dir}")
    return paths


def _build_segnet_calib(paths: list, out_npy: str, n: int = 100) -> str:
    """Random 320×320 patches → [N, 1, 320, 320] float32 normalised to [-1, 1]."""
    arrays = []
    for p in paths[:n]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        y = np.random.randint(0, max(1, h - 320))
        x = np.random.randint(0, max(1, w - 320))
        patch = img[y:y + 320, x:x + 320]
        patch = cv2.resize(patch, (320, 320)).astype(np.float32) / 127.5 - 1.0
        arrays.append(patch[np.newaxis, np.newaxis])   # [1,1,320,320]
    data = np.concatenate(arrays, axis=0)
    np.save(out_npy, data)
    print(f"    Calibration data: {out_npy}  shape={data.shape}")
    return out_npy


def _build_encoder_calib(paths: list, out_npy: str, n: int = 50) -> str:
    """Full-canvas crops → [N, 1, 256, 1280] float32 normalised."""
    arrays = []
    for p in paths[:n]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        canvas = cv2.resize(img, (CANVAS_W, CANVAS_H)).astype(np.float32)
        canvas = (canvas / 255.0 - 0.7931) / 0.1738
        arrays.append(canvas[np.newaxis, np.newaxis])  # [1,1,256,1280]
    data = np.concatenate(arrays, axis=0)
    np.save(out_npy, data)
    print(f"    Calibration data: {out_npy}  shape={data.shape}")
    return out_npy


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def _export_onnx_segnet(model: SegNet, out: str):
    dummy = torch.zeros(1, 1, 320, 320)
    torch.onnx.export(
        model, dummy, out,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,
    )
    print(f"    ONNX → {out}")


def _export_onnx_encoder(encoder: nn.Module, out: str):
    dummy = torch.zeros(1, 1, CANVAS_H, CANVAS_W)
    torch.onnx.export(
        encoder, dummy, out,
        input_names=['canvas'],
        output_names=['memory'],
        dynamic_axes={'canvas': {0: 'batch'}, 'memory': {0: 'batch'}},
        opset_version=17,
    )
    print(f"    ONNX → {out}")


class _DecoderStepWrapper(nn.Module):
    """
    Single-step decoder wrapper for export.

    Inputs:
      past_ids : [1, T]      int64   — token ids so far (starts with SOS)
      memory   : [1, S, 512] float32 — encoder output

    Output:
      next_logits : [1, VOCAB_SIZE] float32 — logits for the next token

    The C++ decoder_runner must call this once per decoding step, passing all
    previously generated token ids as past_ids (growing by 1 each step).
    This avoids the complexity of explicit KV-cache tensors at the cost of
    O(T) attention recomputation per step.  For sequences ≤608 tokens and
    on-device inference this overhead is acceptable.
    """
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, past_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(past_ids, memory)   # [1, T, vocab]
        return logits[:, -1, :]                    # [1, vocab]


def _export_onnx_decoder(seq2seq: OmrSeq2Seq, out: str):
    wrapper = _DecoderStepWrapper(seq2seq.decoder).eval()
    dummy_ids = torch.tensor([[SOS_ID]], dtype=torch.long)
    dummy_mem = torch.zeros(1, SEQ_LEN, 512)
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_mem), out,
        input_names=['past_ids', 'memory'],
        output_names=['next_logits'],
        dynamic_axes={
            'past_ids':    {0: 'batch', 1: 'seq_len'},
            'memory':      {0: 'batch', 1: 'enc_seq'},
            'next_logits': {0: 'batch'},
        },
        opset_version=17,
    )
    print(f"    ONNX → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX simplification (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _simplify(onnx_path: str):
    try:
        import onnx
        import onnxsim
        m = onnx.load(onnx_path)
        m_sim, ok = onnxsim.simplify(m)
        if ok:
            onnx.save(m_sim, onnx_path)
            print(f"    ONNX simplified")
        else:
            print(f"    ONNX simplification had no effect")
    except ImportError:
        print("    onnxsim not installed — skipping (pip install onnxsim)")


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX → TFLite via onnx2tf
# ─────────────────────────────────────────────────────────────────────────────

def _convert_tflite(onnx_path: str, tflite_path: str,
                    calib_npy: str | None, quantize: bool) -> bool:
    try:
        import onnx2tf
    except ImportError:
        print("    ERROR: onnx2tf not installed.")
        print("    Run: pip install onnx2tf tensorflow")
        return False

    tmp_tf = tflite_path + "_saved_model"
    os.makedirs(tmp_tf, exist_ok=True)

    kwargs: dict = dict(
        input_onnx_file_path=onnx_path,
        output_folder_path=tmp_tf,
        non_verbose=True,
    )

    if quantize and calib_npy and os.path.exists(calib_npy):
        kwargs['output_integer_quantization'] = True
        kwargs['representative_dataset_file_path'] = calib_npy
        print(f"    Quantizing INT8 with: {calib_npy}")
    else:
        print(f"    Exporting as FP32")

    onnx2tf.convert(**kwargs)

    generated = [f for f in os.listdir(tmp_tf) if f.endswith('.tflite')]
    if not generated:
        print(f"    ERROR: no .tflite found in {tmp_tf}")
        shutil.rmtree(tmp_tf, ignore_errors=True)
        return False

    shutil.copy(os.path.join(tmp_tf, generated[0]), tflite_path)
    shutil.rmtree(tmp_tf, ignore_errors=True)
    print(f"    TFLite → {tflite_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Version management
# ─────────────────────────────────────────────────────────────────────────────

def _save_versioned(src: str, out_dir: str, stem: str, version: str):
    """
    Copy src to:
      out_dir/{stem}_{version}.tflite  — permanent versioned snapshot
      out_dir/{stem}.tflite            — latest (overwritten each run)
    """
    versioned = os.path.join(out_dir, f"{stem}_{version}.tflite")
    latest    = os.path.join(out_dir, f"{stem}.tflite")
    shutil.copy(src, versioned)
    shutil.copy(src, latest)
    size_kb = os.path.getsize(versioned) / 1024
    print(f"    Versioned → {versioned}  ({size_kb:.0f} KB)")
    print(f"    Latest    → {latest}")


def _save_tokenizer(src: str, out_dir: str, version: str):
    versioned = os.path.join(out_dir, f"tokenizer_{version}.json")
    latest    = os.path.join(out_dir, "tokenizer.json")
    shutil.copy(src, versioned)
    shutil.copy(src, latest)
    print(f"    Versioned → {versioned}")
    print(f"    Latest    → {latest}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Export OMR models to TFLite.  "
                    "Saves both a versioned copy and an overwriting 'latest' copy."
    )
    p.add_argument('--segnet',           required=True,
                   help='segnet_best.pt checkpoint')
    p.add_argument('--seq2seq',          required=True,
                   help='seq2seq_best.pt checkpoint (encoder + decoder)')
    p.add_argument('--tokenizer',        default='data/tokenizer.json')
    p.add_argument('--data_dir',         default='data/train',
                   help='Directory with .png images used for INT8 calibration')
    p.add_argument('--out_dir',          default='assets/',
                   help='Destination directory for .tflite files')
    p.add_argument('--version',          default='v1',
                   help='Version tag appended to filenames (e.g. v1, v2)')
    p.add_argument('--calib_n',          type=int, default=100,
                   help='How many images to use for INT8 calibration (default: 100)')
    p.add_argument('--quantize_decoder', action='store_true',
                   help='Also quantize decoder to INT8 (experimental, may hurt accuracy)')
    p.add_argument('--no_quantize',      action='store_true',
                   help='Skip INT8 quantization for all models (FP32 export)')
    p.add_argument('--device',           default='cpu',
                   help='torch device for loading checkpoints')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp = os.path.join(args.out_dir, '_export_tmp')
    os.makedirs(tmp, exist_ok=True)

    device   = torch.device(args.device)
    quantize = not args.no_quantize

    print(f"\n{'='*60}")
    print(f"  OMR TFLite Export  —  version: {args.version}")
    print(f"{'='*60}")
    print(f"  segnet   : {args.segnet}")
    print(f"  seq2seq  : {args.seq2seq}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  INT8     : {quantize}")
    print(f"{'='*60}\n")

    # ── Collect calibration images ────────────────────────
    image_paths: list = []
    if quantize:
        try:
            image_paths = _collect_pngs(args.data_dir, args.calib_n)
            print(f"Calibration images: {len(image_paths)} from {args.data_dir}\n")
        except FileNotFoundError as e:
            print(f"WARNING: {e}\nFalling back to FP32 export.\n")
            quantize = False

    # ── 1. SegNet ─────────────────────────────────────────
    print("─── [1/3] SegNet ─────────────────────────────────────")
    segnet    = load_segnet(args.segnet, device)
    onnx_seg  = os.path.join(tmp, 'segnet.onnx')
    _export_onnx_segnet(segnet, onnx_seg)
    _simplify(onnx_seg)

    calib_seg = None
    if quantize:
        calib_seg = os.path.join(tmp, 'calib_segnet.npy')
        _build_segnet_calib(image_paths, calib_seg, n=args.calib_n)

    tflite_seg = os.path.join(tmp, 'segnet_INT8.tflite')
    if _convert_tflite(onnx_seg, tflite_seg, calib_seg, quantize):
        _save_versioned(tflite_seg, args.out_dir, 'segnet_INT8', args.version)
    print()

    # ── 2. Encoder ────────────────────────────────────────
    print("─── [2/3] Encoder ────────────────────────────────────")
    seq2seq   = load_seq2seq(args.seq2seq, device)
    onnx_enc  = os.path.join(tmp, 'encoder.onnx')
    _export_onnx_encoder(seq2seq.encoder, onnx_enc)
    _simplify(onnx_enc)

    calib_enc = None
    if quantize:
        calib_enc = os.path.join(tmp, 'calib_encoder.npy')
        _build_encoder_calib(image_paths, calib_enc, n=min(50, len(image_paths)))

    tflite_enc = os.path.join(tmp, 'encoder_INT8.tflite')
    if _convert_tflite(onnx_enc, tflite_enc, calib_enc, quantize):
        _save_versioned(tflite_enc, args.out_dir, 'encoder_INT8', args.version)
    print()

    # ── 3. Decoder ────────────────────────────────────────
    print("─── [3/3] Decoder ────────────────────────────────────")
    print("    Interface: (past_ids[1,T], memory[1,S,512]) → next_logits[1,978]")
    print("    T grows by 1 each decoding step (no explicit KV-cache tensors)")
    onnx_dec  = os.path.join(tmp, 'decoder.onnx')
    _export_onnx_decoder(seq2seq, onnx_dec)
    _simplify(onnx_dec)

    dec_quantize = quantize and args.quantize_decoder
    if quantize and not args.quantize_decoder:
        print("    Quantization: FP32 (use --quantize_decoder to enable INT8)")

    tflite_dec = os.path.join(tmp, 'decoder_INT8.tflite')
    if _convert_tflite(onnx_dec, tflite_dec, None, dec_quantize):
        _save_versioned(tflite_dec, args.out_dir, 'decoder_INT8', args.version)
    print()

    # ── 4. Tokenizer ──────────────────────────────────────
    print("─── Tokenizer ────────────────────────────────────────")
    if os.path.exists(args.tokenizer):
        _save_tokenizer(args.tokenizer, args.out_dir, args.version)
    else:
        print(f"    WARNING: tokenizer not found at {args.tokenizer}")
    print()

    # ── Cleanup ───────────────────────────────────────────
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"{'='*60}")
    print(f"  Done.  Output: {os.path.abspath(args.out_dir)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
