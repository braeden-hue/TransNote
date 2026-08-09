"""generate_scores.py로 만든 "쇼팽 스타일" 합성 샘플(designKit/chopin_style_test*/num*.png)을
현재 체크포인트로 인식시켜 정확도 확인. num*.json이 이미 현재 토크나이저 포맷(note-{pitch}+
dur-{dur} 분리)이라 relabel/trim 없이 바로 비교 가능."""
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

CKPT = Path(__file__).resolve().parent.parent / 'secrets' / 'checkpoints' / 'seq2seq_p2s5n5_best.pt'
TOKENIZER = Path(__file__).resolve().parent / 'tokenizer258.json'


def main(test_dir):
    test_dir = Path(test_dir)
    device = torch.device('cpu')
    tok2id, id2tok = load_tokenizer(str(TOKENIZER))
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(str(CKPT), map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {CKPT.name}\n")

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
        print(f"  GT  : {' '.join(gt_tokens)}")
        print(f"  PRED: {' '.join(pred_tokens)}\n")

    if results:
        print(f"=== 전체 평균 Acc: {sum(results)/len(results):.1f}% (n={len(results)}장) ===")


if __name__ == '__main__':
    test_dir = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / 'designKit' / 'chopin_style_test3')
    main(test_dir)
