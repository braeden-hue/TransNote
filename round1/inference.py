"""
round1train/inference.py  –  Round 1 PyTorch 추론 (단일 오선).

사용법:
    python round1train/inference.py \\
        --seq2seq round1train/models_r1/seq2seq_best.pt \\
        image.png

    # 일괄 평가
    python round1train/inference.py \\
        --seq2seq round1train/models_r1/seq2seq_best.pt \\
        --eval_dir round1train/Round1 \\
        --n_eval 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model import OmrSeq2Seq, SOS_ID, EOS_ID, PAD_ID, MAX_SEQ
from dataset import (preprocess, detect_staffs, extract_staff_canvas,
                     IMG_MEAN, IMG_STD, load_tokenizer)


@torch.no_grad()
def greedy_decode(seq2seq: OmrSeq2Seq, canvas: np.ndarray,
                  device: torch.device, max_len: int = MAX_SEQ) -> List[int]:
    tile_f = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    inp    = torch.from_numpy(tile_f).unsqueeze(0).unsqueeze(0).to(device)
    seq2seq.eval()
    memory = seq2seq.encode(inp)
    past   = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    result = []
    for _ in range(max_len):
        logits = seq2seq.decode_step(None, memory, past)
        nxt    = int(logits.argmax(-1).item())
        if nxt == EOS_ID: break
        if nxt != PAD_ID: result.append(nxt)
        past = torch.cat([past, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return result


def run_image(image_path: str, seq2seq: OmrSeq2Seq,
              tok2id: Dict[str, int], id2tok: Dict[int, str],
              device: torch.device) -> List[str]:
    SKIP = {'<PAD>', '<SOS>', '<EOS>'}
    bgr  = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    gray   = preprocess(bgr)
    staffs = detect_staffs(gray)
    if not staffs:
        return []

    all_tokens: List[str] = []
    for staff in staffs:
        canvas = extract_staff_canvas(gray, staff)
        ids    = greedy_decode(seq2seq, canvas, device)
        ids    = fix_span_tokens(fix_chord_tokens(ids, id2tok), id2tok)
        all_tokens.extend(id2tok.get(i, '<UNK>') for i in ids
                          if id2tok.get(i, '') not in SKIP)
    return all_tokens


def levenshtein(a: List[int], b: List[int]) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            curr[j] = (prev[j-1] if a[i-1] == b[j-1]
                       else 1 + min(prev[j], curr[j-1], prev[j-1]))
        prev, curr = curr, prev
    return prev[n]


# ─────────────────────────────────────────────────────────────────────────────
#  포스트프로세싱 + 마디 단위 TER (OMR-NED 방식, 수정 1~4)
# ─────────────────────────────────────────────────────────────────────────────

_BARLINE_TOKEN_STRS = frozenset({
    'barline', 'barline-final', 'barline-start-repeat', 'barline-end-repeat',
})

_SPAN_PAIRS = {
    'slur-start':           'slur-end',
    'hairpin-cresc-start':  'hairpin-cresc-end',
    'hairpin-dim-start':    'hairpin-dim-end',
    'ottava-8va-start':     'ottava-8va-end',
    'ottava-8vb-start':     'ottava-8vb-end',
    'tuplet-3-start':       'tuplet-3-end',
}
_SPAN_ENDS = frozenset(_SPAN_PAIRS.values())


def fix_chord_tokens(token_ids: List[int], id2tok: dict) -> List[int]:
    """고아 chord- 토큰 제거: note- 또는 chord- 바로 뒤에만 허용."""
    result = []
    for tid in token_ids:
        tok = id2tok.get(tid, '')
        if tok.startswith('chord-'):
            prev = id2tok.get(result[-1], '') if result else ''
            if prev.startswith('note-') or prev.startswith('chord-'):
                result.append(tid)
        else:
            result.append(tid)
    return result


def fix_span_tokens(token_ids: List[int], id2tok: dict) -> List[int]:
    """짝 없는 span start/end 토큰 제거 (stack 기반)."""
    remove = set()
    stacks: dict = {s: [] for s in _SPAN_PAIRS}
    for idx, tid in enumerate(token_ids):
        tok = id2tok.get(tid, '')
        if tok in _SPAN_PAIRS:
            stacks[tok].append(idx)
        elif tok in _SPAN_ENDS:
            start = next(s for s, e in _SPAN_PAIRS.items() if e == tok)
            if stacks[start]:
                stacks[start].pop()
            else:
                remove.add(idx)
    for indices in stacks.values():
        remove.update(indices)
    return [tid for i, tid in enumerate(token_ids) if i not in remove]


def _split_measures(token_ids: List[int], barline_ids: set) -> List[List[int]]:
    """barline ID로 마디 분리 (barline 포함)."""
    measures, cur = [], []
    for tid in token_ids:
        cur.append(tid)
        if tid in barline_ids:
            measures.append(cur); cur = []
    if cur:
        measures.append(cur)
    return measures


def measure_segmented_ter(pred: List[int], gt: List[int],
                           barline_ids: set) -> float:
    """마디 단위 TER (OMR-NED 방식). 마디 간 오류 전파 차단."""
    if not gt:
        return 0.0 if not pred else 1.0
    pred_m = _split_measures(pred, barline_ids)
    gt_m   = _split_measures(gt,   barline_ids)
    total_err = total_len = 0
    for i in range(max(len(pred_m), len(gt_m))):
        p = pred_m[i] if i < len(pred_m) else []
        g = gt_m[i]   if i < len(gt_m)   else []
        total_err += levenshtein(p, g)
        total_len += len(g)
    return total_err / max(total_len, 1)


def eval_dir(eval_dir_path: str, seq2seq: OmrSeq2Seq,
             tok2id: Dict[str, int], id2tok: Dict[int, str],
             device: torch.device, n_eval: int = 200):
    pairs = []
    for fname in sorted(os.listdir(eval_dir_path)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        stem  = os.path.splitext(fname)[0]
        gt_p  = os.path.join(eval_dir_path, stem + '.json')
        img_p = os.path.join(eval_dir_path, fname)
        if os.path.isfile(gt_p):
            pairs.append((img_p, gt_p))
    pairs = pairs[:n_eval]
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    print(f"평가 샘플: {len(pairs)}개")

    ter_sum = note_ter_sum = n_pass = 0
    t0 = time.time()

    for i, (img_p, gt_p) in enumerate(pairs):
        with open(gt_p, encoding='utf-8') as f:
            gt_toks = [t for t in json.load(f).get('tokens', [])
                       if t not in ('<SOS>', '<EOS>', '<PAD>')]
        try:
            pred_toks = run_image(img_p, seq2seq, tok2id, id2tok, device)
        except Exception as e:
            print(f"  [ERROR] {img_p}: {e}")
            continue

        gt_ids   = [tok2id.get(t, 3) for t in gt_toks]
        pred_ids = [tok2id.get(t, 3) for t in pred_toks]
        ter      = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        ter_sum += ter

        gt_notes   = [t for t in gt_toks   if t.startswith('note-')]
        pred_notes = [t for t in pred_toks if t.startswith('note-')]
        note_ter_sum += levenshtein(
            [tok2id.get(t, 3) for t in pred_notes],
            [tok2id.get(t, 3) for t in gt_notes]
        ) / max(len(gt_notes), 1)

        if ter == 0.0:
            n_pass += 1

        if (i + 1) % 50 == 0:
            avg = ter_sum / (i + 1)
            print(f"  [{i+1}/{len(pairs)}]  avg TER={avg*100:.1f}%  acc={((1-avg)*100):.1f}%")

    n        = len(pairs)
    avg_ter  = ter_sum / max(n, 1)
    avg_note = note_ter_sum / max(n, 1)
    elapsed  = time.time() - t0

    print(f"\n{'─'*50}")
    print(f"샘플       : {n}")
    print(f"TER        : {avg_ter*100:.1f}%")
    print(f"Acc (1-TER): {(1-avg_ter)*100:.1f}%")
    print(f"Note Acc   : {(1-avg_note)*100:.1f}%")
    print(f"Pass (TER=0): {n_pass/max(n,1)*100:.1f}%  ({n_pass}/{n})")
    print(f"소요 시간  : {elapsed:.1f}s")


def main():
    p = argparse.ArgumentParser(description='Round 1 추론/평가')
    p.add_argument('image',      nargs='?', help='단일 이미지 경로')
    p.add_argument('--seq2seq',  required=True, help='seq2seq_best.pt 경로')
    p.add_argument('--tokenizer',
                   default=str(_HERE / 'tokenizer.json'))
    p.add_argument('--eval_dir', default=None, help='일괄 평가 디렉토리')
    p.add_argument('--n_eval',   type=int, default=200)
    p.add_argument('--device',   default='auto')
    args = p.parse_args()

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f"Device: {device}")

    tok2id, id2tok = load_tokenizer(args.tokenizer)
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt    = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}")

    if args.eval_dir:
        eval_dir(args.eval_dir, seq2seq, tok2id, id2tok, device, args.n_eval)
    elif args.image:
        tokens = run_image(args.image, seq2seq, tok2id, id2tok, device)
        print('추론 결과:')
        print(' '.join(tokens))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
