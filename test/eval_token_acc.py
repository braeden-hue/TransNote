"""
round3train/eval_token_acc.py -- 학습 로그의 val_acc(=1-measure_segmented_ter)와 동일한 방식으로
디렉토리 전체에 대해 토큰 단위 정확도를 계산한다. error_breakdown.py는 완전일치/오류종류만
보여주고 학습 때 쓰던 Acc와 직접 비교 가능한 숫자를 안 주기 때문에 별도로 작성.

사용법:
    python3 eval_token_acc.py --seq2seq <ckpt.pt> --tokenizer <tok.json> --data_dir <dir>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import run_image
from model import OmrSeq2Seq
from dataset import load_tokenizer
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}

    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}")

    fnames = sorted(f for f in os.listdir(args.data_dir) if f.lower().endswith('.png'))
    ter_sum = 0.0
    n = 0
    for fname in fnames:
        stem = os.path.splitext(fname)[0]
        gt_p = os.path.join(args.data_dir, stem + '.json')
        if not os.path.isfile(gt_p):
            continue
        with open(gt_p, encoding='utf-8') as f:
            data = json.load(f)
        gt_toks = [t for t in data.get('tokens', []) if t not in ('<SOS>', '<EOS>', '<PAD>')]
        gt_ids  = [tok2id[t] for t in gt_toks if t in tok2id]

        pred_toks = run_image(os.path.join(args.data_dir, fname), seq2seq, tok2id, id2tok, device)
        pred_ids  = [tok2id[t] for t in pred_toks if t in tok2id]
        pred_ids  = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)

        ter_sum += measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        n += 1

    val_ter = ter_sum / max(n, 1)
    val_acc = max(0.0, 1.0 - val_ter) * 100.0
    print(f"n={n}  TER={val_ter*100:.1f}%  Acc={val_acc:.1f}%  (학습 로그 val_acc와 동일 방식)")


if __name__ == '__main__':
    main()
