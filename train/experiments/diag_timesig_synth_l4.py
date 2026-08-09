"""박자표(time-*) 기호 가시성 실험(2026-08-04). generate_scores.py --hide-timesig-prob로
같은 시드의 두 배치(박자표 보임 vs 숨김, 음악 내용은 완전 동일)를 만들어 L4 노이즈 하에서
비교 -- "박자표 기호가 안 보이면 모델이 time- 토큰을 못 맞춘다"는 가설을 직접 검증한다.
diag_chord_synth_l4.py와 동일 인프라(OMRDataset page_level_noise 강제 L4 + 로컬 CPU decode).
"""
import argparse
import time
from collections import Counter

import torch

from dataset import load_tokenizer, OMRDataset
from model import OmrSeq2Seq, infer_arch_from_state_dict
from train import fix_chord_tokens, fix_span_tokens
from diag_chord_synth_l4 import edit_distance_opcodes, greedy_decode_from_tensor


def run_one(seq2seq, id2tok, ds, device, tag):
    n = len(ds)
    acc_list = []
    time_correct = 0
    time_total = 0
    t0 = time.time()
    for idx in range(n):
        canvas_t, tgt_in, tgt_out = ds[idx]
        gt_ids = tgt_out[:-1].tolist()
        inp = canvas_t.unsqueeze(0).to(device)
        pred_ids = greedy_decode_from_tensor(seq2seq, inp, device)
        pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)

        gt_toks = [id2tok.get(i, '') for i in gt_ids]
        pred_toks = [id2tok.get(i, '') for i in pred_ids]

        if gt_toks == pred_toks:
            acc = 100.0
        else:
            ops = edit_distance_opcodes(gt_toks, pred_toks)
            n_err = sum(max(i2 - i1, j2 - j1) for tg, i1, i2, j1, j2 in ops if tg != 'equal')
            acc = max(0.0, 1 - n_err / max(1, len(gt_toks))) * 100
        acc_list.append(acc)

        if len(gt_toks) > 2 and gt_toks[2].startswith('time-'):
            time_total += 1
            if len(pred_toks) > 2 and pred_toks[2] == gt_toks[2]:
                time_correct += 1

    elapsed = time.time() - t0
    print(f"\n=== [{tag}] 표본={n}  경과={elapsed:.1f}s ===")
    print(f"전체 정확도: 평균 {sum(acc_list)/len(acc_list):.1f}%  중앙값 {sorted(acc_list)[len(acc_list)//2]:.1f}%")
    print(f"time- 토큰(헤더 3번째) 정확도: {time_correct}/{time_total} ({100*time_correct/max(1,time_total):.1f}%)")
    return acc_list, time_correct, time_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--ctrl_dir', required=True, help='박자표 보임(대조군)')
    ap.add_argument('--hide_dir', required=True, help='박자표 숨김(실험군)')
    ap.add_argument('--tokenizer', default='tokenizer258.json')
    args = ap.parse_args()

    device = torch.device('cpu')
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.seq2seq, map_location=device, weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    in_ch = arch.get('in_ch', 1)
    print(f"모델 로드: {args.seq2seq}  in_ch={in_ch}")

    ds_ctrl = OMRDataset(args.ctrl_dir, tok2id, augment=True,
                         noise_level=4, noise_level_max=None, page_level_noise=True, in_ch=in_ch)
    ds_hide = OMRDataset(args.hide_dir, tok2id, augment=True,
                         noise_level=4, noise_level_max=None, page_level_noise=True, in_ch=in_ch)

    run_one(seq2seq, id2tok, ds_ctrl, device, '대조군(박자표 보임)')
    run_one(seq2seq, id2tok, ds_hide, device, '실험군(박자표 숨김)')


if __name__ == '__main__':
    main()
