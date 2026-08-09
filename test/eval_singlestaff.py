"""단일 오선 인식률 확인용 -- 지정한 체크포인트로 단일 오선 테스트셋(properly-scoped:
artic/ornament/slur/ottava/hairpin=0, tuplet 포함)에 대해 exact-match와 토큰 Acc를 잰다."""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', default='tokenizer258.json')
    ap.add_argument('--test_dir', required=True)
    args = ap.parse_args()

    device = torch.device('cpu')
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()

    pngs = sorted(glob.glob(os.path.join(args.test_dir, '*.png')))
    print(f"{len(pngs)} images, checkpoint={os.path.basename(args.seq2seq)}\n")

    n_exact, total_ter, n = 0, 0.0, 0
    for p in pngs:
        jf = p[:-4] + '.json'
        if not os.path.exists(jf):
            continue
        with open(jf, encoding='utf-8') as f:
            gt_tokens = json.load(f)['tokens']
        gt_clean = [t for t in gt_tokens if t not in ('<SOS>', '<EOS>')]
        gt_ids = [tok2id[t] for t in gt_clean if t in tok2id]

        pred_tokens = run_image(p, seq2seq, tok2id, id2tok, device, beam_width=1)
        pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]
        pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)

        exact = (pred_tokens == gt_clean)
        ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        acc = max(0.0, 1 - ter) * 100
        total_ter += ter
        n += 1
        if exact:
            n_exact += 1
        stem = os.path.splitext(os.path.basename(p))[0]
        print(f"{stem}: exact={exact} Acc={acc:.1f}%")

    print(f"\n=== exact_match: {n_exact}/{n} ({n_exact/n*100:.1f}%) ===")
    print(f"=== 평균 token Acc: {max(0.0, 1 - total_ter/n)*100:.1f}% ===")


if __name__ == '__main__':
    main()
