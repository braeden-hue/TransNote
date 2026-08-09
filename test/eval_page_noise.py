"""
round3train/eval_page_noise.py -- 5n5 실측 때 썼던 "노이즈 낀 사진 -> correct_perspective ->
오선 재검출 -> 디코딩" 실제 추론 경로 그대로, 깨끗한 렌더 이미지 풀 위에서 즉석으로
페이지 레벨 노이즈(dataset.page_noise_and_redetect)를 입혀 L1~L4 정확도를 잰다.

원래 test100_5n5_L2/L3_tokenacc.log를 만들 때 쓴 test100_local 풀은 로컬에서 정리돼
남아있지 않아서, 대신 이미 존재하는 클린 렌더 풀(예: data/local_pools/6reg1)에서 N장을
무작위로 뽑아 그 자리에서 노이즈를 입힌다 -- eval_token_acc.py(정적 파일 기반)와 달리
파일을 새로 저장하지 않고 메모리에서 바로 디코딩까지 수행.

재검출 실패(page_noise_and_redetect가 None 반환)는 실제 운영에서도 완전 실패로 이어지므로
TER=1.0(해당 샘플 완전 불일치)으로 집계한다 -- 성공률만 별도로도 같이 보고.

사용법:
    python3 eval_page_noise.py --seq2seq <ckpt.pt> --tokenizer <tok.json> \
        --data_dir data/local_pools/6reg1 --level 2 --n 100 --seed 42
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
from dataset import (load_tokenizer, load_preprocessed,
                      best_effort_staff_detection, page_noise_and_redetect)
from inference import beam_decode, correct_time_signature, _strip_leading_header
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS

SKIP_TOKS = {'<PAD>', '<SOS>', '<EOS>'}


def _decode_system(seq2seq, canvas, device, id2tok, tok2id, beam_width, is_grand, strip_header):
    ids = beam_decode(seq2seq, canvas, device, beam_width=beam_width)
    ids = fix_span_tokens(fix_chord_tokens(ids, id2tok), id2tok)
    ids = correct_time_signature(ids, id2tok, tok2id, is_grand=is_grand)
    if strip_header:
        ids = _strip_leading_header(ids, id2tok)
    return [id2tok.get(t, '<UNK>') for t in ids if id2tok.get(t, '<UNK>') not in SKIP_TOKS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--level', type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--beam_width', type=int, default=1)
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
    random.Random(args.seed).shuffle(fnames)
    fnames = fnames[:args.n]

    ter_sum = 0.0
    n_scored = 0
    n_redetect_fail = 0
    n_no_staff = 0
    for fname in fnames:
        stem = os.path.splitext(fname)[0]
        gt_p = os.path.join(args.data_dir, stem + '.json')
        if not os.path.isfile(gt_p):
            continue
        with open(gt_p, encoding='utf-8') as f:
            data = json.load(f)
        gt_toks = [t for t in data.get('tokens', []) if t not in ('<SOS>', '<EOS>', '<PAD>')]
        gt_ids = [tok2id[t] for t in gt_toks if t in tok2id]

        img_path = os.path.join(args.data_dir, fname)
        gray0 = load_preprocessed(img_path)
        staffs, _ = best_effort_staff_detection(gray0)
        if not staffs:
            n_no_staff += 1
            ter_sum += 1.0
            n_scored += 1
            continue

        n_staff = len(staffs)
        all_tokens = []
        redetect_failed = False
        if n_staff >= 2 and n_staff % 2 == 0:
            n_systems = n_staff // 2
            for sys_i in range(n_systems):
                pair = [staffs[sys_i * 2], staffs[sys_i * 2 + 1]]
                canvas = page_noise_and_redetect(gray0, pair, args.level)
                if canvas is None:
                    redetect_failed = True
                    break
                all_tokens.extend(_decode_system(seq2seq, canvas, device, id2tok, tok2id,
                                                  args.beam_width, True, sys_i > 0))
        else:
            for staff in staffs:
                canvas = page_noise_and_redetect(gray0, staff, args.level)
                if canvas is None:
                    redetect_failed = True
                    break
                all_tokens.extend(_decode_system(seq2seq, canvas, device, id2tok, tok2id,
                                                  args.beam_width, False, False))

        if redetect_failed:
            n_redetect_fail += 1
            ter_sum += 1.0
            n_scored += 1
            continue

        pred_ids = [tok2id[t] for t in all_tokens if t in tok2id]
        ter_sum += measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        n_scored += 1

    val_ter = ter_sum / max(n_scored, 1)
    val_acc = max(0.0, 1.0 - val_ter) * 100.0
    print(f"n={n_scored}  level={args.level}  TER={val_ter*100:.1f}%  Acc={val_acc:.1f}%  "
          f"(재검출실패={n_redetect_fail}, 오선미검출={n_no_staff})")


if __name__ == '__main__':
    main()
