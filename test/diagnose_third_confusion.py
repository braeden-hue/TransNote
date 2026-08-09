"""D4<->F4, C4<->A3, D6<->B5 등 단/장3도류 오독이 정확히 어떤 문맥에서 나는지 진단.
GT-PRED를 정렬해서, 음이름 대체 오류 중 반음 간격이 3~4(단/장3도)인 경우만 골라
그 앞뒤 문맥 토큰(코드/클렙전환/임시표/붙임줄 근접 여부)을 함께 출력한다.
"""
import argparse
import difflib
import json
import os
import sys
from pathlib import Path

import torch
from music21.pitch import Pitch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from inference import run_image
from model import OmrSeq2Seq, infer_arch_from_state_dict
from dataset import load_tokenizer


def midi_of(pitch_str):
    try:
        return Pitch(pitch_str.replace('b', '-')).midi
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--context', type=int, default=6)
    args = ap.parse_args()

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}")

    fnames = sorted(f for f in os.listdir(args.data_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    found = []

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
            pred_toks = run_image(img_p, seq2seq, tok2id, id2tok, device, beam_width=1)
        except Exception:
            continue
        if gt_toks == pred_toks:
            continue

        sm = difflib.SequenceMatcher(a=gt_toks, b=pred_toks, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != 'replace':
                continue
            glen, plen = i2 - i1, j2 - j1
            m = min(glen, plen)
            for k in range(m):
                gi, ji = i1 + k, j1 + k
                gtok, ptok = gt_toks[gi], pred_toks[ji]
                if not (gtok.startswith('note-') and ptok.startswith('note-')):
                    continue
                gp, pp = gtok.split('-', 1)[1], ptok.split('-', 1)[1]
                gm, pmidi = midi_of(gp), midi_of(pp)
                if gm is None or pmidi is None:
                    continue
                diff = abs(gm - pmidi)
                if diff in (3, 4):
                    ctx_lo, ctx_hi = max(0, gi - args.context), min(len(gt_toks), gi + args.context + 1)
                    ctx = gt_toks[ctx_lo:gi] + [f'[[{gtok}->{ptok}]]'] + gt_toks[gi+1:ctx_hi]
                    found.append((stem, gi, ' '.join(ctx)))

    print(f"\n총 {len(found)}건 단/장3도 오독 발견\n")
    for stem, gi, ctx in found[:40]:
        print(f"-- {stem} (idx={gi}) --")
        print(f"   {ctx}\n")


if __name__ == '__main__':
    main()
