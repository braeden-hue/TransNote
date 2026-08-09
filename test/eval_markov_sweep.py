"""
round3train/eval_markov_sweep.py -- 디코딩 단계 마르코프 융합(inference.MarkovDecodeConfig)의
weight 값을 바꿔가며, 같은 체크포인트/같은 테스트 이미지 세트로 정답과 얼마나 비슷해지는지
비교한다. weight=0이 마르코프 없음(기존 동작) 기준선.

주의: data_dir로 6reg1 같은 풀을 쓰면, 그 체크포인트가 아직 모르는 콘텐츠(교차음역 등)라
절대 정확도 자체는 낮게 나올 수 있다(별도로 진단된 문제, eval_page_noise.py 참고) -- 이
실험의 목적은 절대값이 아니라 "같은 조건에서 weight만 바꿨을 때 상대적으로 얼마나
달라지는지"다.

사용법:
    python3 eval_markov_sweep.py --seq2seq <ckpt.pt> --tokenizer <tok.json> \
        --data_dir data/local_pools/6reg1 --markov_table markov_transitions.json \
        --n 50 --weights 0,0.3,0.5,0.7,1.0,1.5,2.0
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OmrSeq2Seq
from dataset import load_tokenizer
from inference import run_image, MarkovDecodeConfig, load_markov_table
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--markov_table', required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--beam_width', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--weights', type=str, default='0,0.3,0.5,0.7,1.0,1.5,2.0')
    args = ap.parse_args()

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}

    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}")

    table, max_interval = load_markov_table(args.markov_table)
    print(f"마르코프 테이블 로드: {args.markov_table} (max_interval={max_interval})")

    fnames = sorted(f for f in os.listdir(args.data_dir) if f.lower().endswith('.png'))
    random.Random(args.seed).shuffle(fnames)
    fnames = fnames[:args.n]

    samples = []
    for fname in fnames:
        stem = os.path.splitext(fname)[0]
        gt_p = os.path.join(args.data_dir, stem + '.json')
        if not os.path.isfile(gt_p):
            continue
        with open(gt_p, encoding='utf-8') as f:
            data = json.load(f)
        gt_toks = [t for t in data.get('tokens', []) if t not in ('<SOS>', '<EOS>', '<PAD>')]
        gt_ids = [tok2id[t] for t in gt_toks if t in tok2id]
        samples.append((os.path.join(args.data_dir, fname), gt_ids))

    print(f"샘플 {len(samples)}장, weight 스윕: {args.weights}")

    weights = [float(x) for x in args.weights.split(',')]
    results = []
    for w in weights:
        markov = MarkovDecodeConfig(id2tok, tok2id, table, max_interval, w) if w > 0 else None
        ter_sum = 0.0
        for img_path, gt_ids in samples:
            pred_toks = run_image(img_path, seq2seq, tok2id, id2tok, device,
                                   beam_width=args.beam_width, markov=markov)
            pred_ids = [tok2id[t] for t in pred_toks if t in tok2id]
            pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
            ter_sum += measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        val_ter = ter_sum / max(len(samples), 1)
        val_acc = max(0.0, 1.0 - val_ter) * 100.0
        results.append((w, val_ter * 100, val_acc))
        print(f"  weight={w:<5} TER={val_ter*100:5.1f}%  Acc={val_acc:5.1f}%", flush=True)

    print()
    print("=== 요약 (n=%d) ===" % len(samples))
    print(f"{'weight':>8} {'TER%':>8} {'Acc%':>8}")
    for w, ter, acc in results:
        print(f"{w:8.2f} {ter:8.1f} {acc:8.1f}")
    best = max(results, key=lambda r: r[2])
    print(f"\n최고: weight={best[0]} (Acc={best[2]:.1f}%)")


if __name__ == '__main__':
    main()
