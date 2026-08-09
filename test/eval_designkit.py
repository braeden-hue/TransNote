"""designKit/ 20장(대보표 10 + 단일오선 10, 노이즈 없는 신선한 로컬 렌더)에 실제 학습된
체크포인트를 돌려 GT와 비교. train.py와 같은 토큰-레벨 Acc 지표(measure_segmented_ter)
사용 -- eval_token_acc.py와 동일한 방법론."""
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

DESIGNKIT = os.path.join(os.path.dirname(__file__), '..', 'designKit')
CKPT = os.path.join(os.path.dirname(__file__), '..', 'secrets', 'checkpoints', 'seq2seq_p2s5n5_best.pt')
TOKENIZER = os.path.join(os.path.dirname(__file__), 'tokenizer258.json')


def main():
    device = torch.device('cpu')
    tok2id, id2tok = load_tokenizer(TOKENIZER)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()

    pngs = sorted(glob.glob(os.path.join(DESIGNKIT, 'num9*.png')))
    print(f"{len(pngs)} designKit images, checkpoint={os.path.basename(CKPT)}\n")

    total_ter, n = 0.0, 0
    results = []
    for p in pngs:
        stem = os.path.splitext(os.path.basename(p))[0]
        jf = p[:-4] + '.json'
        if not os.path.exists(jf):
            continue
        with open(jf, encoding='utf-8') as f:
            gt_tokens = json.load(f)['tokens']
        gt_ids = [tok2id[t] for t in gt_tokens if t in tok2id and t not in ('<SOS>', '<EOS>')]

        pred_tokens = run_image(p, seq2seq, tok2id, id2tok, device, beam_width=1)
        pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]

        pred_ids = fix_chord_tokens(pred_ids, id2tok)
        pred_ids = fix_span_tokens(pred_ids, id2tok)

        ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        acc = max(0.0, 1 - ter) * 100
        total_ter += ter
        n += 1
        is_grand = 'staff-bass' in gt_tokens
        results.append((stem, is_grand, acc))
        print(f"{stem} ({'grand' if is_grand else 'single'}): Acc={acc:.1f}%")

    print(f"\n=== 평균 Acc: {max(0.0, 1 - total_ter/n)*100:.1f}% (n={n}) ===")
    grand = [a for _, g, a in results if g]
    single = [a for _, g, a in results if not g]
    if grand:
        print(f"대보표만: {sum(grand)/len(grand):.1f}% (n={len(grand)})")
    if single:
        print(f"단일오선만: {sum(single)/len(single):.1f}% (n={len(single)})")


if __name__ == '__main__':
    main()
