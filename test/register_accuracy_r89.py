"""89곡 exactPicture 실측에서, GT 음의 옥타브가 2~5(정상 노출 구간)인지 극단(0/1/6+,
Step1 데이터에 사실상 없던 구간)인지로 나눠 계이름 정확도를 따로 집계.
2026-07-31 발견: Step1 학습 데이터의 옥타브 분포가 2~5에 집중(21.7~27.5%씩)돼 있고
옥타브1/6은 사실상 0건 -- 그 노출 격차가 실측 오독(D6->B5, C6->A5, Bb1->Db2 등)의
원인인지 확인하기 위한 베이스라인(Round3 체크포인트) 측정.
"""
import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from inference import run_image
from model import OmrSeq2Seq, infer_arch_from_state_dict
from dataset import load_tokenizer

_NOTE_RE = re.compile(r'^([A-G])(b|#)?(-?\d+)$')


def octave_of(pitch_str: str):
    m = _NOTE_RE.match(pitch_str)
    if not m:
        return None
    return int(m.group(3))


def is_extreme(octv: int) -> bool:
    return octv <= 1 or octv >= 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--beam_width', type=int, default=1)
    ap.add_argument('--exclude', default='',
                     help='제외할 곡 stem 콤마구분 목록 (예: 옥타브1 포함 4곡 제외하고 '
                          '85곡만 볼 때)')
    args = ap.parse_args()
    exclude = set(s.strip() for s in args.exclude.split(',') if s.strip())

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}  device={device}")

    fnames = sorted(f for f in os.listdir(args.data_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if exclude:
        before = len(fnames)
        fnames = [f for f in fnames if os.path.splitext(f)[0] not in exclude]
        print(f"제외 {before - len(fnames)}곡: {sorted(exclude)}")
    print(f"전체 {len(fnames)}곡")

    normal_total = normal_correct = 0
    extreme_total = extreme_correct = 0
    octave_total = {}
    octave_correct = {}

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
            pred_toks = run_image(img_p, seq2seq, tok2id, id2tok, device, beam_width=args.beam_width)
        except Exception as e:
            print(f"  [{stem}] 추론 실패: {e}")
            continue

        def bucket(gi, correct):
            tok = gt_toks[gi]
            if not (tok.startswith('note-') or tok.startswith('chord-')):
                return
            octv = octave_of(tok.split('-', 1)[1])
            if octv is None:
                return
            nonlocal normal_total, normal_correct, extreme_total, extreme_correct
            octave_total[octv] = octave_total.get(octv, 0) + 1
            if correct:
                octave_correct[octv] = octave_correct.get(octv, 0) + 1
            if is_extreme(octv):
                extreme_total += 1
                if correct:
                    extreme_correct += 1
            else:
                normal_total += 1
                if correct:
                    normal_correct += 1

        if gt_toks == pred_toks:
            for gi in range(len(gt_toks)):
                bucket(gi, True)
            continue

        sm = difflib.SequenceMatcher(a=gt_toks, b=pred_toks, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    bucket(i1 + k, True)
            elif tag == 'replace':
                glen, plen = i2 - i1, j2 - j1
                m = min(glen, plen)
                for k in range(m):
                    gi = i1 + k
                    gtok, ptok = gt_toks[gi], pred_toks[j1 + k]
                    correct = (gtok == ptok)
                    bucket(gi, correct)
                for k in range(m, glen):
                    bucket(i1 + k, False)
            elif tag == 'delete':
                for k in range(i1, i2):
                    bucket(k, False)
            # insert(과잉)는 GT 옥타브가 없으므로 집계 제외

    print(f"\n=== 옥타브 2~5(정상 노출) 계이름 정확도: {normal_correct}/{normal_total} "
          f"({normal_correct/normal_total*100:.1f}%) ===" if normal_total else "\n정상구간 데이터 없음")
    print(f"=== 옥타브 0~1/6+(극단, Step1 데이터에 거의 없던 구간) 계이름 정확도: "
          f"{extreme_correct}/{extreme_total} "
          f"({extreme_correct/extreme_total*100 if extreme_total else 0:.1f}%) ===")
    print("\n=== 옥타브별 상세 ===")
    for o in sorted(octave_total):
        t = octave_total[o]
        c = octave_correct.get(o, 0)
        print(f"  옥타브 {o}: {c}/{t} ({c/t*100:.1f}%)")


if __name__ == '__main__':
    main()
