#!/usr/bin/env python3
"""
evaluate_round_accuracy.py  --  Per-round accuracy evaluation for OMR models.

Evaluates a trained PyTorch checkpoint (.pt) against test images for a
specific training round, then reports per-token accuracy, overall accuracy,
and a confusion matrix summary for the most common error pairs.

The script loads the OmrSeq2Seq model (encoder + decoder) from a .pt file,
runs greedy inference on each test image, and compares the predicted token
sequence against the ground-truth JSON label.

Metrics:
  Overall accuracy   = max(0, 1 - TER)   where TER = Levenshtein / gt_length
  Note accuracy      = same, restricted to note/chord/rest tokens only
  Per-token accuracy = fraction of token types predicted exactly right

Usage:
    # Evaluate Round-2 checkpoint against Round-2 test images
    python ml/scripts/evaluate_round_accuracy.py \\
        --round 2 \\
        --weights checkpoints/round2.pt \\
        --test-dir ml/data/Round2/test \\
        --tokenizer ml/data/tokenizer.json

    # Also save a CSV report and show confusion matrix
    python ml/scripts/evaluate_round_accuracy.py \\
        --round 1 \\
        --weights checkpoints/round1_best.pt \\
        --test-dir ml/data/Round1/test \\
        --tokenizer ml/data/tokenizer.json \\
        --report ml/reports/round1_eval.csv \\
        --confusion-top 20

    # Use a SegNet checkpoint for full end-to-end evaluation
    python ml/scripts/evaluate_round_accuracy.py \\
        --round 3 \\
        --weights checkpoints/round3.pt \\
        --segnet-weights checkpoints/segnet_round3.pt \\
        --test-dir ml/data/Round3/test \\
        --tokenizer ml/data/tokenizer.json

    # Evaluate against training data to check overfitting
    python ml/scripts/evaluate_round_accuracy.py \\
        --round 2 \\
        --weights checkpoints/round2.pt \\
        --test-dir ml/data/Round2 \\
        --tokenizer ml/data/tokenizer.json \\
        --max-samples 50

Dependencies:
    pip install torch torchvision opencv-python numpy tqdm
"""

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Repo-relative imports: add omr/ to path so model.py / dataset.py are found.
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "omr" / "training"))
sys.path.insert(0, str(_REPO_ROOT / "omr" / "utils"))


# ─────────────────────────────────────────────────────────────────────────────
#  Levenshtein / Token Error Rate
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(a: List, b: List) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            curr[j] = (prev[j - 1] if a[i - 1] == b[j - 1]
                       else 1 + min(prev[j], curr[j - 1], prev[j - 1]))
        prev, curr = curr, prev
    return prev[n]


def token_error_rate(pred: List, gt: List) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def note_only(tokens: List[str]) -> List[str]:
    return [t for t in tokens
            if t.startswith("note-") or t.startswith("chord-") or t.startswith("rest-")]


# ─────────────────────────────────────────────────────────────────────────────
#  Ground-truth loader
# ─────────────────────────────────────────────────────────────────────────────

def load_gt(json_path: str) -> Optional[List[str]]:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        toks = data.get("tokens", [])
        # Strip <SOS> and <EOS> — model output does not include them.
        return [t for t in toks if t not in ("<SOS>", "<EOS>", "<PAD>")]
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Test-pair discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_test_pairs(test_dir: str,
                    max_samples: Optional[int]) -> List[Tuple[str, str, str]]:
    """Return sorted (stem, image_path, gt_path) tuples with both image + JSON."""
    EXTS = {".png", ".jpg", ".jpeg"}
    pairs = []
    for fname in sorted(os.listdir(test_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in EXTS:
            continue
        gt_path  = os.path.join(test_dir, base + ".json")
        img_path = os.path.join(test_dir, fname)
        if os.path.isfile(gt_path):
            pairs.append((base, img_path, gt_path))
    if max_samples and len(pairs) > max_samples:
        pairs = pairs[:max_samples]
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_seq2seq_model(weights_path: str, tokenizer_path: str, device):
    """Load OmrSeq2Seq from a .pt checkpoint."""
    import torch
    from model import OmrSeq2Seq, VOCAB_SIZE
    from dataset import load_tokenizer

    tok2id, id2tok = load_tokenizer(tokenizer_path)
    vocab_size = len(tok2id)

    model = OmrSeq2Seq(vocab_size=vocab_size).to(device)
    ckpt  = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded seq2seq weights : {weights_path}  (vocab={vocab_size})")
    return model, tok2id, id2tok


def load_segnet_model(weights_path: str, device):
    """Load SegNet from a .pt checkpoint (optional)."""
    import torch
    from model import SegNet, NUM_CLASSES

    model = SegNet(num_classes=NUM_CLASSES).to(device)
    ckpt  = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded segnet weights  : {weights_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Image preprocessing (mirrors omr_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

IMG_MEAN = 0.7931
IMG_STD  = 0.1738
CANVAS_H = 256
CANVAS_W = 1280

def preprocess_image(img_path: str) -> Optional[np.ndarray]:
    """Load and preprocess a sheet music image to a (1, 1, H, W) float32 tensor."""
    try:
        import cv2
    except ImportError:
        sys.exit("ERROR: opencv-python is required.  Run: pip install opencv-python")

    bgr = cv2.imread(img_path)
    if bgr is None:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()

    # Resize to 1920 width preserving aspect ratio.
    H, W = gray.shape
    target_w = 1920
    if W != target_w:
        scale  = target_w / W
        new_h  = int(round(H * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray   = cv2.resize(gray, (target_w, new_h), interpolation=interp)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.bilateralFilter(gray, 9, 20.0, 7.0)

    # Fit to canvas (256 x 1280) — simple single-tile crop from top strip.
    H, W = gray.shape
    tile = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    copy_h = min(CANVAS_H, H)
    copy_w = min(CANVAS_W, W)
    tile[:copy_h, :copy_w] = gray[:copy_h, :copy_w]

    # Normalise and expand to (1, 1, H, W).
    tile_f = (tile.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    return tile_f[None, None, :, :]   # [1, 1, 256, 1280]


# ─────────────────────────────────────────────────────────────────────────────
#  Greedy decode (PyTorch, no TFLite)
# ─────────────────────────────────────────────────────────────────────────────

SOS_ID  = 1
EOS_ID  = 2
PAD_ID  = 0
MAX_SEQ = 608

@staticmethod
def _noop_static():
    pass

def greedy_decode_pt(seq2seq, canvas_tensor, device, max_len: int = MAX_SEQ) -> List[int]:
    """Greedy decode a single canvas tensor using the PyTorch seq2seq model."""
    import torch

    seq2seq.eval()
    with torch.no_grad():
        canvas = canvas_tensor.to(device)
        memory = seq2seq.encode(canvas)                 # [1, S, D]
        past   = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
        result = []

        for _ in range(max_len):
            logits = seq2seq.decode_step(None, memory, past)   # [1, vocab]
            nxt    = int(logits.argmax(-1).item())
            if nxt == EOS_ID:
                break
            if nxt != PAD_ID:
                result.append(nxt)
            past = torch.cat(
                [past, torch.tensor([[nxt]], dtype=torch.long, device=device)],
                dim=1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Per-token accuracy analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_token_accuracy(all_preds: List[List[str]],
                                all_gts:   List[List[str]]
                                ) -> Dict[str, Dict]:
    """
    For each token type that appears in GT, compute:
      recall    = (#times correctly predicted at the right position) / (#times in GT)
    Uses position-aligned comparison after LCS alignment (approximate: zip up to min len).
    """
    token_gt_count   = collections.Counter()
    token_hit_count  = collections.Counter()

    for pred_toks, gt_toks in zip(all_preds, all_gts):
        for gt_tok in gt_toks:
            token_gt_count[gt_tok] += 1
        # Rough alignment: compare zipped elements
        for p, g in zip(pred_toks, gt_toks):
            if p == g:
                token_hit_count[g] += 1

    result = {}
    for tok, count in sorted(token_gt_count.items(), key=lambda x: -x[1]):
        hits = token_hit_count.get(tok, 0)
        result[tok] = {
            "gt_count":  count,
            "hit_count": hits,
            "recall":    hits / count if count > 0 else 0.0,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion matrix: most-confused token pairs
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion_pairs(all_preds: List[List[str]],
                             all_gts:   List[List[str]],
                             top_n:     int = 20) -> List[Tuple]:
    """
    Return the top_n (gt_token, pred_token, count) triples where
    gt_token was in GT but pred_token was predicted instead.
    """
    confusion: Dict[Tuple[str, str], int] = collections.Counter()
    for pred_toks, gt_toks in zip(all_preds, all_gts):
        for p, g in zip(pred_toks, gt_toks):
            if p != g:
                confusion[(g, p)] += 1
    return sorted(confusion.items(), key=lambda x: -x[1])[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
#  Report formatters
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list, round_num: int, weights_path: str,
                 test_dir: str, threshold: float,
                 per_token: Dict, confusion_pairs: list,
                 confusion_top: int):

    valid = [r for r in results if r["error"] is None]
    n_total = len(results)
    n_valid = len(valid)

    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  OMR Round-{round_num} Accuracy Report")
    print(SEP)
    print(f"  Weights    : {weights_path}")
    print(f"  Test dir   : {test_dir}")
    print(f"  Samples    : {n_total} found  /  {n_valid} evaluated")
    print(f"  Threshold  : {threshold*100:.1f}%")
    print()

    if n_valid == 0:
        print("  No valid samples evaluated.")
        return False

    mean_acc   = sum(r["accuracy"]  for r in valid) / n_valid
    mean_note  = sum(r["note_acc"]  for r in valid) / n_valid
    n_pass     = sum(1 for r in valid if r["accuracy"] >= threshold)
    passed     = mean_acc >= threshold

    print(f"  RESULT     : {'PASS' if passed else 'FAIL'}")
    print(f"  Overall    : {mean_acc*100:.1f}%  (TER {(1-mean_acc)*100:.1f}%)")
    print(f"  Note-only  : {mean_note*100:.1f}%")
    print(f"  Per-sample : {n_pass}/{n_valid} pass threshold")
    print()

    # Per-sample table.
    print(f"  {'Sample':<22}  {'GT':>4}  {'Pred':>4}  {'TER':>6}  {'NoteAcc':>7}  {'':>4}")
    print(f"  {'-'*22}  {'--':>4}  {'----':>4}  {'------':>6}  {'-------':>7}  {'':>4}")
    for r in results:
        if r["error"]:
            print(f"  {r['stem']:<22}  ERROR: {r['error']}")
            continue
        status = "PASS" if r["accuracy"] >= threshold else "FAIL"
        print(f"  {r['stem']:<22}  {r['gt_len']:>4}  {r['pred_len']:>4}"
              f"  {r['ter']*100:>5.1f}%  {r['note_acc']*100:>6.1f}%  {status}")

    # Per-token accuracy table (top 30 by GT frequency).
    print(f"\n  {'─'*70}")
    print(f"  Per-Token Recall (top 30 by GT frequency)")
    print(f"  {'─'*70}")
    print(f"  {'Token':<32}  {'GT':>6}  {'Hits':>6}  {'Recall':>7}")
    print(f"  {'-'*32}  {'------':>6}  {'------':>6}  {'-------':>7}")
    for tok, stats in list(per_token.items())[:30]:
        recall_bar = "#" * int(stats["recall"] * 10) + "." * (10 - int(stats["recall"] * 10))
        print(f"  {tok:<32}  {stats['gt_count']:>6}  {stats['hit_count']:>6}"
              f"  {stats['recall']*100:>6.1f}%  [{recall_bar}]")

    # Confusion matrix.
    if confusion_pairs:
        print(f"\n  {'─'*70}")
        print(f"  Top-{confusion_top} Confused Token Pairs  (GT → Pred)")
        print(f"  {'─'*70}")
        print(f"  {'GT token':<32}  {'Pred token':<32}  {'Count':>5}")
        print(f"  {'-'*32}  {'-'*32}  {'-----':>5}")
        for (gt_tok, pred_tok), cnt in confusion_pairs[:confusion_top]:
            print(f"  {gt_tok:<32}  {pred_tok:<32}  {cnt:>5}")

    print(f"\n{SEP}\n")
    return passed


def write_csv(results: list, path: str, threshold: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "gt_len", "pred_len",
                    "ter", "accuracy", "note_ter", "note_acc", "pass", "error"])
        for r in results:
            w.writerow([
                r["stem"], r.get("gt_len", 0), r.get("pred_len", 0),
                f"{r.get('ter', 1):.6f}", f"{r.get('accuracy', 0):.6f}",
                f"{r.get('note_ter', 1):.6f}", f"{r.get('note_acc', 0):.6f}",
                1 if (r["error"] is None and r.get("accuracy", 0) >= threshold) else 0,
                r["error"] or "",
            ])
    print(f"  CSV saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-round OMR accuracy evaluation using .pt checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--round",      type=int, required=True, choices=[1, 2, 3, 4],
                   help="Training round being evaluated (1–4)")
    p.add_argument("--weights",    required=True,
                   help="Path to seq2seq .pt checkpoint")
    p.add_argument("--test-dir",   required=True,
                   help="Directory containing image + .json pairs")
    p.add_argument("--tokenizer",  default=str(_REPO_ROOT / "data" / "tokenizer.json"),
                   help="Path to tokenizer.json (default: ml/data/tokenizer.json)")
    p.add_argument("--segnet-weights", default=None,
                   help="Optional SegNet .pt checkpoint (uses simple canvas crop if omitted)")
    p.add_argument("--threshold",  type=float, default=0.90,
                   help="Accuracy threshold for PASS/FAIL (default: 0.90)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap evaluation at this many samples")
    p.add_argument("--report",     default=None,
                   help="Save CSV report to this path")
    p.add_argument("--confusion-top", type=int, default=20,
                   help="Number of top confused token pairs to show (default: 20)")
    p.add_argument("--device",     default="auto",
                   help="Device: auto | cuda | cpu (default: auto)")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-sample progress lines")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Dependency checks ─────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: torch is required.  Run: pip install torch")

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device     : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if not os.path.isfile(args.weights):
        sys.exit(f"ERROR: weights file not found: {args.weights}")
    if not os.path.isfile(args.tokenizer):
        sys.exit(f"ERROR: tokenizer not found: {args.tokenizer}")

    seq2seq, tok2id, id2tok = load_seq2seq_model(args.weights, args.tokenizer, device)

    segnet = None
    if args.segnet_weights:
        if not os.path.isfile(args.segnet_weights):
            print(f"WARNING: segnet weights not found: {args.segnet_weights}  (skipping)")
        else:
            segnet = load_segnet_model(args.segnet_weights, device)

    # ── Test pairs ────────────────────────────────────────────────────────────
    pairs = find_test_pairs(args.test_dir, args.max_samples)
    if not pairs:
        sys.exit(f"ERROR: no image+json pairs found in {args.test_dir}")

    print(f"Test dir   : {args.test_dir}")
    print(f"Pairs      : {len(pairs)}")
    print(f"Round      : {args.round}")
    print(f"Threshold  : {args.threshold*100:.1f}%\n")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_results = []
    all_pred_str: List[List[str]] = []
    all_gt_str:   List[List[str]] = []

    for i, (stem, img_path, gt_path) in enumerate(pairs):
        res = {"stem": stem, "error": None,
               "gt_len": 0, "pred_len": 0,
               "ter": 1.0, "accuracy": 0.0,
               "note_ter": 1.0, "note_acc": 0.0}

        gt_tokens = load_gt(gt_path)
        if gt_tokens is None:
            res["error"] = "could not load ground truth"
            all_results.append(res)
            continue

        res["gt_len"] = len(gt_tokens)

        if not args.quiet:
            print(f"  [{i+1:3d}/{len(pairs)}] {stem} ...", end=" ", flush=True)

        canvas_np = preprocess_image(img_path)
        if canvas_np is None:
            res["error"] = "image load failed"
            all_results.append(res)
            if not args.quiet:
                print("IMAGE LOAD FAILED")
            continue

        import torch
        canvas_t = torch.from_numpy(canvas_np)

        try:
            pred_ids = greedy_decode_pt(seq2seq, canvas_t, device)
        except Exception as exc:
            res["error"] = f"inference error: {exc}"
            all_results.append(res)
            if not args.quiet:
                print(f"INFERENCE FAILED: {exc}")
            continue

        pred_tokens = [id2tok.get(idx, "<UNK>") for idx in pred_ids]

        res["pred_len"] = len(pred_tokens)
        res["ter"]      = token_error_rate(pred_tokens, gt_tokens)
        res["accuracy"] = max(0.0, 1.0 - res["ter"])
        res["note_ter"] = token_error_rate(note_only(pred_tokens), note_only(gt_tokens))
        res["note_acc"] = max(0.0, 1.0 - res["note_ter"])

        if not args.quiet:
            status = "PASS" if res["accuracy"] >= args.threshold else "FAIL"
            print(f"TER={res['ter']*100:5.1f}%  NoteAcc={res['note_acc']*100:5.1f}%  {status}")

        all_results.append(res)
        all_pred_str.append(pred_tokens)
        all_gt_str.append(gt_tokens)

    # ── Analysis ──────────────────────────────────────────────────────────────
    per_token       = compute_per_token_accuracy(all_pred_str, all_gt_str)
    confusion_pairs = compute_confusion_pairs(all_pred_str, all_gt_str,
                                               top_n=args.confusion_top)

    # ── Report ────────────────────────────────────────────────────────────────
    passed = print_report(
        results        = all_results,
        round_num      = args.round,
        weights_path   = args.weights,
        test_dir       = args.test_dir,
        threshold      = args.threshold,
        per_token      = per_token,
        confusion_pairs= confusion_pairs,
        confusion_top  = args.confusion_top,
    )

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        write_csv(all_results, args.report, args.threshold)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
