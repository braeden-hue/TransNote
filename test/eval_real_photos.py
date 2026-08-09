"""ml/data/test/chop*/의 실사 촬영 사진 + 정답 라벨로 실제 인식률 검증.

라벨은 두 가지를 먼저 맞춰야 함:
  1) 구 토큰 포맷(note-{pitch}-{dur} 결합) -> 현재 포맷(note-{pitch} + dur-{dur} 분리)
     relabel_notes.py의 split_note_token()과 동일한 변환.
  2) 마디 수 -- 라벨은 5~6마디인데 실제 사진엔 한 줄(약 4마디)만 들어있어서, 현재
     스코프(최대 4마디)와 사진에 실제로 보이는 내용에 맞춰 앞 4마디만 잘라 비교.
"""
import argparse
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS

TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'test')
CKPT = os.path.join(os.path.dirname(__file__), '..', 'secrets', 'checkpoints', 'seq2seq_p2s5n5_best.pt')
TOKENIZER = os.path.join(os.path.dirname(__file__), 'tokenizer258.json')
MAX_MEASURES = 4

_NOTE_DUR_RE = re.compile(r"^note-(.+)-(\d+/\d+)$")
_OUT_OF_SCOPE_PREFIXES = ('artic-', 'ornament-', 'slur-')


def strip_out_of_scope(tokens):
    return [t for t in tokens if not t.startswith(_OUT_OF_SCOPE_PREFIXES)]


def split_note_token(tok):
    m = _NOTE_DUR_RE.match(tok)
    if not m:
        return [tok]
    return [f"note-{m.group(1)}", f"dur-{m.group(2)}"]


def relabel(tokens):
    out = []
    for t in tokens:
        out.extend(split_note_token(t))
    return out


def trim_to_measures(tokens, max_measures):
    """헤더(clef/key/time) 뒤로 barline 토큰 max_measures개까지만 남기고 자른다
    (barline-final 등 어떤 barline 변형이든 카운트)."""
    out = []
    n_bar = 0
    for t in tokens:
        out.append(t)
        if t.startswith('barline'):
            n_bar += 1
            if n_bar >= max_measures:
                break
    if not out[-1].startswith('barline'):
        out.append('barline-final')
    elif out[-1] != 'barline-final':
        out[-1] = 'barline-final'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--tokenizer', default=TOKENIZER)
    ap.add_argument('--test_dir', default=TEST_DIR)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.ckpt}\n")

    folders = sorted(d for d in glob.glob(os.path.join(args.test_dir, 'chop*')) if os.path.isdir(d))
    all_acc = []
    for folder in folders:
        name = os.path.basename(folder)
        # mscz_to_label.py로 .mscz에서 뽑은 라벨(*_new.json)을 우선 사용 -- 기존 손으로 쓴
        # *.json은 마디 경계/tie 처리가 부정확했음(2026-07-28 eval_mscz_clean.py와 동일 이유).
        json_paths = glob.glob(os.path.join(folder, '*_new.json'))
        if not json_paths:
            json_paths = glob.glob(os.path.join(folder, '*.json'))
        photo_paths = sorted(glob.glob(os.path.join(folder, 'photo_*.jpg')))
        if not json_paths or not photo_paths:
            print(f"[{name}] json/photo 없음, 건너뜀")
            continue
        with open(json_paths[0], encoding='utf-8') as f:
            raw_tokens = json.load(f)['tokens']
        gt_full = relabel(raw_tokens)
        gt_trimmed = trim_to_measures(gt_full, MAX_MEASURES)
        gt_clean = strip_out_of_scope([t for t in gt_trimmed if t not in ('<SOS>', '<EOS>')])
        gt_ids = [tok2id[t] for t in gt_clean if t in tok2id]
        print(f"[{name}] 원본 {sum(1 for t in gt_full if t.startswith('barline'))}마디 -> "
              f"{MAX_MEASURES}마디로 축소, GT: {' '.join(gt_clean)}")

        for photo in photo_paths:
            pname = os.path.basename(photo)
            try:
                pred_tokens = run_image(photo, seq2seq, tok2id, id2tok, device, beam_width=1)
            except Exception as exc:
                print(f"  {pname}: 추론 실패 ({exc})")
                continue
            pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]
            pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
            ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
            acc = max(0.0, 1 - ter) * 100
            all_acc.append(acc)
            print(f"  {pname}: Acc={acc:.1f}%")
            print(f"    PRED: {' '.join(pred_tokens)}")
        print()

    if all_acc:
        print(f"=== 전체 평균 Acc: {sum(all_acc)/len(all_acc):.1f}% (n={len(all_acc)}장) ===")


if __name__ == '__main__':
    main()
