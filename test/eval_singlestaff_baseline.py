"""단일 오선 인식 능력이 현재 체크포인트에서 얼마나 남아있는지(파인튜닝 전 baseline)
빠르게 확인. generate_scores.py --single-staff로 만든 num*.json/png를 그대로 비교."""
import glob
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS


def main():
    ckpt_path = sys.argv[1]
    test_dir = Path(sys.argv[2])
    tokenizer_path = sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).resolve().parent / 'tokenizer258.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok2id, id2tok = load_tokenizer(tokenizer_path)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {ckpt_path}  device={device}")

    results = []
    for json_path in sorted(test_dir.glob('num*.json')):
        name = json_path.stem
        png_path = test_dir / f"{name}.png"
        if not png_path.exists():
            continue
        with open(json_path, encoding='utf-8') as f:
            gt_tokens = json.load(f)['tokens']
        gt_tokens = [t for t in gt_tokens if t not in ('<SOS>', '<EOS>')]
        gt_ids = [tok2id[t] for t in gt_tokens if t in tok2id]

        pred_tokens = run_image(str(png_path), seq2seq, tok2id, id2tok, device, beam_width=1)
        pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]
        pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
        ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        acc = max(0.0, 1 - ter) * 100
        results.append(acc)
        print(f"[{name}] Acc={acc:.1f}%")

    if results:
        print(f"=== 단일 오선 평균 Acc: {sum(results)/len(results):.1f}% (n={len(results)}장) ===")


if __name__ == '__main__':
    main()
