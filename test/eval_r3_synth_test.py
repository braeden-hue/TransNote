"""라운드3 학습 규칙(BASE_ARGS/GRAND_ONLY_ARGS)으로 새로 생성한 held-out 합성
테스트셋(학습에 쓰인 seed/start-idx 범위와 겹치지 않음)에 대해 라운드3 체크포인트의
자기회귀 정확도를 측정. exactPicture(실사 촬영) 검증과 달리 순수 합성 이미지라
촬영 노이즈 영향 없이 "모델이 라운드3 규칙 자체를 얼마나 배웠는지"만 분리해서 본다.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq, infer_arch_from_state_dict
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS

DATA_DIR = Path(__file__).resolve().parent / 'data' / 'local_pools' / 'r3_synth_test50'
CKPT = Path(__file__).resolve().parent.parent / 'secrets' / 'checkpoints' / 'seq2seq_r3_density_register_clef_best.pt'
TOKENIZER = Path(__file__).resolve().parent / 'tokenizer258.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(DATA_DIR))
    ap.add_argument('--ckpt', default=str(CKPT))
    ap.add_argument('--tokenizer', default=str(TOKENIZER))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--beam_width', type=int, default=1,
                     help='1(기본)=greedy, 2 이상이면 beam search(inference.beam_decode)')
    ap.add_argument('--exclude', default='',
                     help='제외할 곡 stem 콤마구분 목록')
    args = ap.parse_args()
    exclude = set(s.strip() for s in args.exclude.split(',') if s.strip())

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.ckpt}\n")

    json_paths = sorted(p for p in glob.glob(str(data_dir / '*.json')) if not p.endswith('_staffs.json'))
    if exclude:
        json_paths = [p for p in json_paths if Path(p).stem not in exclude]
    results = []
    for json_path in json_paths:
        stem = Path(json_path).stem
        png_path = data_dir / f"{stem}.png"
        if not png_path.exists():
            continue
        with open(json_path, encoding='utf-8') as f:
            gt_tokens = json.load(f)['tokens']
        gt_clean = [t for t in gt_tokens if t not in ('<SOS>', '<EOS>')]
        gt_ids = [tok2id[t] for t in gt_clean if t in tok2id]

        pred_tokens = run_image(str(png_path), seq2seq, tok2id, id2tok, device, beam_width=args.beam_width)
        pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]
        pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
        ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        acc = max(0.0, 1 - ter) * 100
        results.append(acc)
        print(f"[{stem}] Acc={acc:.1f}%")

    if results:
        print(f"\n=== 전체 평균 Acc: {sum(results)/len(results):.1f}% (n={len(results)}장) ===")


if __name__ == '__main__':
    main()
