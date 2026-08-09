"""89곡 exactPicture 실측에서 dur- 토큰(음표 길이) 정확도만 따로 집계.
계이름(register_accuracy_r89.py)과 별개로, "길이 종류를 얼마나 잘 맞추는지"를 본다.
"""
import argparse
import difflib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from inference import run_image
from model import OmrSeq2Seq, infer_arch_from_state_dict
from dataset import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--beam_width', type=int, default=1)
    ap.add_argument('--exclude', default='')
    args = ap.parse_args()
    exclude = set(s.strip() for s in args.exclude.split(',') if s.strip())

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}  device={device}")

    fnames = sorted(f for f in os.listdir(args.data_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if exclude:
        fnames = [f for f in fnames if os.path.splitext(f)[0] not in exclude]
    print(f"전체 {len(fnames)}곡")

    total = correct = 0
    dur_confusion = Counter()

    for fname in fnames:
        stem = os.path.splitext(fname)[0]
        img_p = os.path.join(args.data_dir, fname)
        gt_p = os.path.join(args.data_dir, stem + '.json')
        if not os.path.isfile(gt_p):
            continue
        with open(gt_p, encoding='utf-8') as f:
            data = json.load(f)
        gt_toks = [t for t in data.get('tokens', []) if t not in ('<SOS>', '<EOS>', '<PAD>')]
        try:
            pred_toks = run_image(img_p, seq2seq, tok2id, id2tok, device, beam_width=args.beam_width)
        except Exception as e:
            print(f"  [{stem}] 추론 실패: {e}")
            continue

        def bucket(gi, correct_flag, ptok=None):
            nonlocal total, correct
            tok = gt_toks[gi]
            if not tok.startswith('dur-'):
                return
            total += 1
            if correct_flag:
                correct += 1
            else:
                dur_confusion[f"{tok} -> {ptok if ptok else '(누락)'}"] += 1

        if gt_toks == pred_toks:
            for gi in range(len(gt_toks)):
                bucket(gi, True)
            continue

        sm = difflib.SequenceMatcher(a=gt_toks, b=pred_toks, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    bucket(i1 + k, True)
            elif tag == 'replace':
                glen, plen = i2 - i1, j2 - j1
                m = min(glen, plen)
                for k in range(m):
                    gi = i1 + k
                    gtok, ptok = gt_toks[gi], pred_toks[j1 + k]
                    bucket(gi, gtok == ptok, ptok)
                for k in range(m, glen):
                    bucket(i1 + k, False)
            elif tag == 'delete':
                for k in range(i1, i2):
                    bucket(k, False)

    print(f"\n=== dur- 토큰(음표 길이) 정확도: {correct}/{total} "
          f"({correct/total*100 if total else 0:.1f}%) ===")
    print("\n=== 상위 오인식 (최대 20) ===")
    for k, v in dur_confusion.most_common(20):
        print(f"  {k:30s} {v:5d}")


if __name__ == '__main__':
    main()
