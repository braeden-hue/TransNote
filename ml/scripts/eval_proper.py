"""
eval_proper.py  --  수정된 전처리로 라운드별 모델 정확도 평가

기존 evaluate_round_accuracy.py의 문제점(단순 좌상단 crop) 수정:
  - dataset.py의 detect_staffs + extract_canvas_tiles 사용
  - R3는 첫 번째 staff(treble) + 두 번째 staff(bass) 타일을 함께 처리

Usage:
    python ml/scripts/eval_proper.py \
        --round 1 \
        --weights ml/models/round1/seq2seq_best.pt \
        --test-dir ml/data/test_eval/round1 \
        --tokenizer ml/data/tokenizer.json

    # 전체 라운드 한 번에:
    python ml/scripts/eval_proper.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "omr" / "training"))

from dataset import preprocess, detect_staffs, extract_canvas_tiles, load_tokenizer, IMG_MEAN, IMG_STD, CANVAS_H, CANVAS_W

SOS_ID = 1
EOS_ID = 2
PAD_ID = 0
MAX_SEQ = 608


# ─────────────────────────────────────────────────────────────────────────────
#  전처리: dataset.py 방식과 동일 (핵심 수정)
# ─────────────────────────────────────────────────────────────────────────────

def image_to_canvas(img_path: str, staff_idx: int = 0) -> Optional[np.ndarray]:
    """
    이미지 → detect_staffs → extract_canvas_tiles → [1,1,H,W] float32 tensor.
    staff_idx: 0=treble(또는 단일), 1=bass
    staff가 검출 안 되면 fallback으로 단순 crop 사용.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None

    gray = preprocess(bgr)
    staffs = detect_staffs(gray)

    if staffs and staff_idx < len(staffs):
        tiles = extract_canvas_tiles(gray, staffs[staff_idx])
        tile = tiles[0]
    else:
        # fallback: 단순 resize to CANVAS_H × CANVAS_W
        tile = cv2.resize(gray, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

    canvas = (tile.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    return canvas[None, None, :, :]   # [1, 1, H, W]


# ─────────────────────────────────────────────────────────────────────────────
#  Levenshtein / TER
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(a: List, b: List) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j], curr[j-1], prev[j-1])
        prev = curr
    return prev[n]


def ter(pred: List, gt: List) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


# ─────────────────────────────────────────────────────────────────────────────
#  Greedy decode
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(model, canvas_np: np.ndarray, device) -> List[int]:
    canvas_t = torch.from_numpy(canvas_np).to(device)
    memory = model.encode(canvas_t)
    past = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    result = []
    for _ in range(MAX_SEQ):
        logits = model.decode_step(None, memory, past)
        nxt = int(logits.argmax(-1).item())
        if nxt == EOS_ID:
            break
        if nxt != PAD_ID:
            result.append(nxt)
        past = torch.cat([past, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  단일 라운드 평가
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_round(round_num: int, weights_path: str, test_dir: str,
                   tokenizer_path: str, device) -> dict:
    from model import OmrSeq2Seq

    tok2id, id2tok = load_tokenizer(tokenizer_path)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # checkpoint vocab이 tokenizer vocab보다 클 수 있으므로 checkpoint 기준으로 모델 생성
    ckpt_vocab = ckpt["model"]["decoder.token_emb.weight"].shape[0]
    actual_vocab = max(len(tok2id), ckpt_vocab)
    model = OmrSeq2Seq(vocab_size=actual_vocab).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model"}
    stored_acc = (1 - meta.get("val_ter", 1.0)) * 100 if "val_ter" in meta else None

    pairs = []
    for fname in sorted(os.listdir(test_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        gt_path = os.path.join(test_dir, base + ".json")
        if os.path.isfile(gt_path):
            pairs.append((base, os.path.join(test_dir, fname), gt_path))

    if not pairs:
        return {"round": round_num, "error": "no pairs found"}

    ters = []
    details = []

    for stem, img_path, gt_path in pairs:
        with open(gt_path, encoding="utf-8") as f:
            gt_tokens = [t for t in json.load(f).get("tokens", [])
                         if t not in ("<SOS>", "<EOS>", "<PAD>")]

        # R3: grand staff → treble staff만 추론 (staff_idx=0)
        # 정확도를 보수적으로 측정
        canvas = image_to_canvas(img_path, staff_idx=0)
        if canvas is None:
            details.append({"stem": stem, "error": "load failed"})
            continue

        try:
            pred_ids = greedy_decode(model, canvas, device)
        except Exception as e:
            details.append({"stem": stem, "error": str(e)})
            continue

        pred_tokens = [id2tok.get(i, "<UNK>") for i in pred_ids]

        # R3: staff-bass 이후 토큰 포함한 전체 시퀀스와 비교
        sample_ter = ter(pred_tokens, gt_tokens)
        ters.append(sample_ter)

        details.append({
            "stem": stem,
            "gt_len": len(gt_tokens),
            "pred_len": len(pred_tokens),
            "ter": round(sample_ter, 4),
            "acc": round((1 - sample_ter) * 100, 1),
        })

        print(f"  R{round_num} {stem:<12}  GT={len(gt_tokens):3d}  Pred={len(pred_tokens):3d}"
              f"  TER={sample_ter*100:5.1f}%  Acc={(1-sample_ter)*100:5.1f}%")

    mean_ter = sum(ters) / len(ters) if ters else 1.0
    mean_acc = (1 - mean_ter) * 100

    return {
        "round": round_num,
        "n_samples": len(pairs),
        "n_evaluated": len(ters),
        "stored_acc": stored_acc,
        "new_acc": round(mean_acc, 1),
        "new_ter": round(mean_ter * 100, 1),
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

ROUND_DEFAULTS = {
    1: {"weights": str(_REPO / "models/round1/seq2seq_best.pt"),
        "test_dir": str(_REPO / "data/test_eval/round1")},
    2: {"weights": str(_REPO / "models/round2/seq2seq_best.pt"),
        "test_dir": str(_REPO / "data/test_eval/round2")},
    3: {"weights": str(_REPO / "models/round3/seq2seq_best.pt"),
        "test_dir": str(_REPO / "data/test_eval/round3")},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round",      type=int, choices=[1, 2, 3])
    p.add_argument("--all",        action="store_true", help="전체 라운드 평가")
    p.add_argument("--weights",    default=None)
    p.add_argument("--test-dir",   default=None)
    p.add_argument("--tokenizer",  default=str(_REPO / "data/tokenizer.json"))
    p.add_argument("--device",     default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}\n")

    rounds = [1, 2, 3] if args.all else [args.round]
    all_results = []

    for r in rounds:
        d = ROUND_DEFAULTS[r]
        weights  = args.weights  or d["weights"]
        test_dir = args.test_dir or d["test_dir"]

        if not os.path.isfile(weights):
            print(f"[R{r}] weights not found: {weights}")
            continue
        if not os.path.isdir(test_dir):
            print(f"[R{r}] test_dir not found: {test_dir}")
            continue

        print(f"{'='*60}")
        print(f"  Round {r}  —  {os.path.basename(weights)}")
        print(f"  Test : {test_dir}")
        print(f"{'='*60}")

        res = evaluate_round(r, weights, test_dir, args.tokenizer, device)
        all_results.append(res)

    # 최종 요약
    print(f"\n{'='*60}")
    print(f"  최종 요약")
    print(f"{'='*60}")
    print(f"  {'라운드':<8}  {'저장 Acc':>10}  {'새 Acc':>10}  {'변화':>8}  판단")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  ----")

    for res in all_results:
        if "error" in res:
            print(f"  R{res['round']}        ERROR: {res['error']}")
            continue
        stored = f"{res['stored_acc']:.1f}%" if res['stored_acc'] is not None else "  n/a"
        new_a  = res['new_acc']
        delta  = new_a - res['stored_acc'] if res['stored_acc'] is not None else 0
        sign   = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

        if new_a >= 80:
            judgment = "재학습 불필요"
        elif new_a >= 65:
            judgment = "하이퍼파라미터 수정 후 재학습"
        else:
            judgment = "재학습 필요"

        print(f"  R{res['round']}        {stored:>10}  {new_a:>9.1f}%  {sign:>8}  {judgment}")

    print()


if __name__ == "__main__":
    main()
