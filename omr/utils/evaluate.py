"""
evaluate.py  –  OMR recognition-rate evaluation tool.

Runs the OMR pipeline on every image in <test_dir> that has a matching
ground-truth .json label, then reports Token Error Rate (TER) and
note-only accuracy.  Exits with code 0 if mean accuracy >= threshold,
code 1 otherwise (suitable for use in CI / training callbacks).

Two execution modes:
  --engine PATH   Use the pre-built C++ omr_eval binary (fastest, uses
                  the identical engine as production).  Recommended once
                  the C++ binary has been built.
  --tflite        Use the Python TFLite inference pipeline from
                  omr_inference.py (no C++ build required, slower).

Metrics:
  TER      = Levenshtein(pred, gt) / len(gt)         (lower is better)
  Accuracy = max(0, 1 - TER)                          (higher is better)
  Note TER = same, computed only over note/chord/rest tokens

Usage examples:
  # Python TFLite mode (during training)
  python omr/utils/evaluate.py \\
      --test_dir  data/test \\
      --segnet    assets/models/segnet.tflite \\
      --encoder   assets/models/encoder.tflite \\
      --decoder   assets/models/decoder.tflite \\
      --tokenizer data/tokenizer.json \\
      --threshold 0.90

  # C++ engine mode (after building)
  python omr/utils/evaluate.py \\
      --test_dir  data/test \\
      --engine    omr/engine/build/omr_eval \\
      --segnet    assets/models/segnet.tflite \\
      --encoder   assets/models/encoder.tflite \\
      --decoder   assets/models/decoder.tflite \\
      --tokenizer data/tokenizer.json \\
      --threshold 0.90 \\
      --report    eval_report.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Token Error Rate (Levenshtein / gt_length)
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(a: List[str], b: List[str]) -> int:
    """
    Standard two-row dynamic programming edit distance.
    Ref: Levenshtein V.I. (1966) Soviet Physics Doklady 10(8):707-710.
    """
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def token_error_rate(pred: List[str], gt: List[str]) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def note_only(tokens: List[str]) -> List[str]:
    return [t for t in tokens
            if t.startswith('note-') or t.startswith('chord-') or t.startswith('rest-')]


# ─────────────────────────────────────────────────────────────────────────────
#  Ground-truth JSON loader
# ─────────────────────────────────────────────────────────────────────────────

def load_gt(json_path: str) -> Optional[List[str]]:
    """Load ground-truth tokens from a .json label file."""
    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('tokens', []) or None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Image file discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_test_pairs(test_dir: str) -> List[Tuple[str, str, str]]:
    """
    Return (stem, image_path, gt_path) tuples for samples that have
    both an image and a .json ground-truth label in test_dir.
    """
    EXTS = {'.png', '.jpg', '.jpeg'}
    pairs = []
    for fname in sorted(os.listdir(test_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in EXTS:
            continue
        gt_path  = os.path.join(test_dir, base + '.json')
        img_path = os.path.join(test_dir, fname)
        if os.path.isfile(gt_path):
            pairs.append((base, img_path, gt_path))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Inference backends
# ─────────────────────────────────────────────────────────────────────────────

class CppBackend:
    """Wraps the pre-built omr_eval C++ binary."""

    def __init__(self, engine_path: str, segnet: str, encoder: str,
                 decoder: str, tokenizer: str):
        self.engine    = engine_path
        self.segnet    = segnet
        self.encoder   = encoder
        self.decoder   = decoder
        self.tokenizer = tokenizer

    def predict(self, image_path: str) -> Optional[List[str]]:
        """
        Run the C++ omr_test binary on a single image and parse its output.
        The binary prints lines of the form:  [  N] ID  token_text
        """
        try:
            result = subprocess.run(
                [self.engine, image_path,
                 self.segnet, self.encoder, self.decoder, self.tokenizer],
                capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return None

        tokens = []
        for line in result.stdout.splitlines():
            line = line.strip()
            # Lines look like: [   0]     4  clef-G
            if line.startswith('['):
                parts = line.split()
                if len(parts) >= 3:
                    tokens.append(parts[-1])
        return tokens if tokens else None


class PythonBackend:
    """Python TFLite inference pipeline (omr_inference.OmrInference)."""

    def __init__(self, segnet: str, encoder: str,
                 decoder: str, tokenizer: str, num_threads: int = 4):
        # Import here so non-Python-mode runs do not require TFLite installed.
        sys.path.insert(0, os.path.dirname(__file__))
        from omr_inference import OmrInference
        self._engine = OmrInference(segnet, encoder, decoder, tokenizer,
                                     num_threads=num_threads)

    def predict(self, image_path: str) -> Optional[List[str]]:
        try:
            return self._engine.process(image_path)
        except Exception as e:
            print(f"    [inference error] {e}", file=sys.stderr)
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  Per-sample result
# ─────────────────────────────────────────────────────────────────────────────

class SampleResult:
    def __init__(self, stem: str):
        self.stem       = stem
        self.gt_len     = 0
        self.pred_len   = 0
        self.ter        = 1.0   # full-sequence Token Error Rate
        self.note_ter   = 1.0   # note-only TER
        self.accuracy   = 0.0   # max(0, 1-ter)
        self.note_acc   = 0.0   # max(0, 1-note_ter)
        self.error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(pairs:      List[Tuple[str, str, str]],
                   backend,
                   threshold:  float,
                   verbose:    bool = True) -> List[SampleResult]:

    results: List[SampleResult] = []

    for i, (stem, img_path, gt_path) in enumerate(pairs):
        res = SampleResult(stem)

        gt = load_gt(gt_path)
        if gt is None:
            res.error = 'could not load ground truth'
            results.append(res)
            continue

        res.gt_len = len(gt)

        if verbose:
            print(f"  [{i+1:3d}/{len(pairs)}] {stem} ...", end=' ', flush=True)

        pred = backend.predict(img_path)

        if pred is None:
            res.error = 'inference failed'
            results.append(res)
            if verbose:
                print('INFERENCE FAILED')
            continue

        res.pred_len = len(pred)
        res.ter      = token_error_rate(pred, gt)
        res.accuracy = max(0.0, 1.0 - res.ter)
        res.note_ter = token_error_rate(note_only(pred), note_only(gt))
        res.note_acc = max(0.0, 1.0 - res.note_ter)

        if verbose:
            status = 'PASS' if res.accuracy >= threshold else 'FAIL'
            print(f'TER={res.ter*100:5.1f}%  NoteAcc={res.note_acc*100:5.1f}%  {status}')

        results.append(res)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: List[SampleResult], threshold: float,
                 test_dir: str, mode: str) -> bool:
    """Print the summary report. Returns True if global PASS."""

    valid = [r for r in results if r.error is None]
    n_total  = len(results)
    n_valid  = len(valid)

    if n_valid == 0:
        print('\nNo valid samples evaluated.')
        return False

    mean_acc  = sum(r.accuracy  for r in valid) / n_valid
    mean_note = sum(r.note_acc  for r in valid) / n_valid
    n_pass    = sum(1 for r in valid if r.accuracy >= threshold)
    passed    = mean_acc >= threshold

    SEP = '=' * 65
    print(f'\n{SEP}')
    print(f'  OMR Evaluation Report')
    print(SEP)
    print(f'  Directory  : {test_dir}')
    print(f'  Mode       : {mode}')
    print(f'  Samples    : {n_total} total  /  {n_valid} evaluated')
    print(f'  Threshold  : {threshold*100:.1f}% accuracy  (TER <= {(1-threshold)*100:.1f}%)')
    print()
    print(f'  RESULT     : {"PASS" if passed else "FAIL"}')
    print(f'  Mean Acc   : {mean_acc*100:.1f}%  (TER: {(1-mean_acc)*100:.1f}%)')
    print(f'  Note Acc   : {mean_note*100:.1f}%')
    print(f'  Per-sample : {n_pass} / {n_valid} pass threshold')
    print()
    print(f'  {"Sample":<22}  {"GT":>4}  {"Pred":>4}  {"TER":>6}  {"NoteAcc":>7}  {"":>4}')
    print(f'  {"-"*22}  {"--":>4}  {"----":>4}  {"---":>6}  {"-------":>7}  {"":>4}')

    for r in results:
        if r.error:
            print(f'  {r.stem:<22}  ERROR: {r.error}')
            continue
        status = 'PASS' if r.accuracy >= threshold else 'FAIL'
        print(f'  {r.stem:<22}  {r.gt_len:>4}  {r.pred_len:>4}'
              f'  {r.ter*100:>5.1f}%  {r.note_acc*100:>6.1f}%  {status}')

    print(SEP)
    return passed


def write_csv(results: List[SampleResult], path: str, threshold: float) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'gt_len', 'pred_len', 'ter', 'accuracy',
                    'note_ter', 'note_acc', 'pass', 'error'])
        for r in results:
            w.writerow([
                r.stem,
                r.gt_len,
                r.pred_len,
                f'{r.ter:.6f}',
                f'{r.accuracy:.6f}',
                f'{r.note_ter:.6f}',
                f'{r.note_acc:.6f}',
                1 if (r.error is None and r.accuracy >= threshold) else 0,
                r.error or '',
            ])
    print(f'\n  CSV report saved to: {path}')


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='OMR recognition-rate evaluator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument('--test_dir',   required=True,
                   help='Directory containing image + .json pairs')
    p.add_argument('--segnet',     required=True,
                   help='Path to segnet.tflite')
    p.add_argument('--encoder',    required=True,
                   help='Path to encoder.tflite')
    p.add_argument('--decoder',    required=True,
                   help='Path to decoder.tflite')
    p.add_argument('--tokenizer',  required=True,
                   help='Path to tokenizer.json')

    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--engine',  default=None,
                      help='Path to omr_eval C++ binary (fast mode)')
    mode.add_argument('--tflite',  action='store_true',
                      help='Use Python TFLite inference (no C++ build needed)')

    p.add_argument('--threshold',  type=float, default=0.90,
                   help='Accuracy threshold for PASS/FAIL (default: 0.90)')
    p.add_argument('--report',     default=None,
                   help='Save detailed CSV report to this path')
    p.add_argument('--threads',    type=int, default=4,
                   help='TFLite interpreter thread count (Python mode only)')
    p.add_argument('--quiet',      action='store_true',
                   help='Suppress per-sample progress output')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Determine execution mode ──────────────────────────────────────────────
    if args.engine:
        if not os.path.isfile(args.engine):
            sys.exit(f'ERROR: C++ binary not found: {args.engine}')
        backend = CppBackend(args.engine, args.segnet, args.encoder,
                              args.decoder, args.tokenizer)
        mode_label = f'C++ engine ({os.path.basename(args.engine)})'
    else:
        # Default to Python TFLite if --engine not supplied.
        for path, name in [(args.segnet,    'segnet'),
                           (args.encoder,   'encoder'),
                           (args.decoder,   'decoder'),
                           (args.tokenizer, 'tokenizer')]:
            if not os.path.isfile(path):
                sys.exit(f'ERROR: {name} file not found: {path}')
        backend = PythonBackend(args.segnet, args.encoder, args.decoder,
                                args.tokenizer, num_threads=args.threads)
        mode_label = 'Python TFLite'

    # ── Discover test pairs ───────────────────────────────────────────────────
    pairs = find_test_pairs(args.test_dir)
    if not pairs:
        sys.exit(f'ERROR: no image+json pairs found in {args.test_dir}')

    print(f'Found {len(pairs)} test pairs in {args.test_dir}')
    print(f'Mode      : {mode_label}')
    print(f'Threshold : {args.threshold*100:.1f}%\n')

    # ── Run ───────────────────────────────────────────────────────────────────
    results = run_evaluation(pairs, backend, args.threshold,
                              verbose=not args.quiet)

    # ── Report ────────────────────────────────────────────────────────────────
    passed = print_report(results, args.threshold, args.test_dir, mode_label)

    if args.report:
        write_csv(results, args.report, args.threshold)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
