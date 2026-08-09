"""오류 '위치' 분석: 마디 순번(1~4번째), 성부(치/베이스), 화음·셋잇단음표 포함 여부별로
마디 단위 정확도를 쪼개서 본다. error_breakdown.py는 오류 '종류'(음높이/길이 등) 위주라
공간적 위치는 안 보여줘서 보완용으로 작성.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq
from train import fix_chord_tokens, fix_span_tokens, _BARLINE_TOKEN_STRS

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / 'data' / 'local_pools' / 'r4_synth_test50'
CKPT = _HERE.parent / 'secrets' / 'checkpoints' / 'seq2seq_r3_density_register_clef_best.pt'
TOKENIZER = _HERE / 'tokenizer258.json'


def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = (prev[j-1] if a[i-1] == b[j-1]
                       else 1 + min(prev[j], curr[j-1], prev[j-1]))
        prev = curr
    return prev[n]


def split_measures(toks, barline_strs):
    measures, cur = [], []
    for t in toks:
        cur.append(t)
        if t in barline_strs:
            measures.append(cur); cur = []
    if cur:
        measures.append(cur)
    return measures


def split_treble_bass(measure_toks):
    if 'staff-bass' in measure_toks:
        idx = measure_toks.index('staff-bass')
        return measure_toks[:idx], measure_toks[idx+1:]
    return measure_toks, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(DATA_DIR))
    ap.add_argument('--ckpt', default=str(CKPT))
    ap.add_argument('--tokenizer', default=str(TOKENIZER))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_strs = frozenset({'barline', 'barline-final', 'barline-start-repeat', 'barline-end-repeat'})
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.ckpt}")

    by_measure_pos = defaultdict(lambda: [0, 0])   # pos(1-based) -> [err, len]
    by_staff = defaultdict(lambda: [0, 0])          # 'treble'/'bass' -> [err, len]
    by_chord_ctx = defaultdict(lambda: [0, 0])       # True/False(화음 포함 마디) -> [err, len]
    by_tuplet_ctx = defaultdict(lambda: [0, 0])      # True/False(3연음 포함 마디) -> [err, len]

    json_paths = sorted(p for p in glob.glob(str(data_dir / '*.json')) if not p.endswith('_staffs.json'))
    n_done = 0
    for jp in json_paths:
        stem = Path(jp).stem
        png_p = data_dir / f"{stem}.png"
        if not png_p.exists():
            continue
        with open(jp, encoding='utf-8') as f:
            gt_tokens = json.load(f)['tokens']
        gt_clean = [t for t in gt_tokens if t not in ('<SOS>', '<EOS>')]

        pred_tokens = run_image(str(png_p), seq2seq, tok2id, id2tok, device, beam_width=1)
        pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]
        pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
        pred_clean = [id2tok[i] for i in pred_ids]

        gt_measures = split_measures(gt_clean, barline_strs)
        pred_measures = split_measures(pred_clean, barline_strs)

        for pos in range(max(len(gt_measures), len(pred_measures))):
            g = gt_measures[pos] if pos < len(gt_measures) else []
            p = pred_measures[pos] if pos < len(pred_measures) else []
            err = levenshtein(g, p)
            by_measure_pos[pos + 1][0] += err
            by_measure_pos[pos + 1][1] += len(g)

            has_chord = any(t.startswith('chord-') for t in g)
            by_chord_ctx[has_chord][0] += err
            by_chord_ctx[has_chord][1] += len(g)

            has_tuplet = 'tuplet-3-start' in g
            by_tuplet_ctx[has_tuplet][0] += err
            by_tuplet_ctx[has_tuplet][1] += len(g)

            g_treble, g_bass = split_treble_bass(g)
            p_treble, p_bass = split_treble_bass(p)
            if g_bass or p_bass:  # 대보표 마디
                by_staff['treble'][0] += levenshtein(g_treble, p_treble)
                by_staff['treble'][1] += len(g_treble)
                by_staff['bass'][0] += levenshtein(g_bass, p_bass)
                by_staff['bass'][1] += len(g_bass)
            else:
                by_staff['single-staff'][0] += err
                by_staff['single-staff'][1] += len(g)

        n_done += 1

    print(f"\n분석 완료: {n_done}곡\n")
    print("=== 마디 순번별 정확도 (마디 단위 Levenshtein, 낮을수록 뒤쪽 마디에서 오류 누적/전파 의심) ===")
    for pos in sorted(by_measure_pos):
        err, tot = by_measure_pos[pos]
        acc = max(0.0, 1 - err / max(tot, 1)) * 100
        print(f"  {pos}번째 마디: Acc={acc:5.1f}%  (오류 {err}/{tot} 토큰, 표본 마디 존재)")

    print("\n=== 성부별 정확도 ===")
    for k in ('treble', 'bass', 'single-staff'):
        err, tot = by_staff[k]
        if tot == 0:
            continue
        acc = max(0.0, 1 - err / tot) * 100
        print(f"  {k:12s}: Acc={acc:5.1f}%  (오류 {err}/{tot})")

    print("\n=== 화음 포함 마디 vs 미포함 마디 정확도 ===")
    for k, label in ((True, '화음 포함'), (False, '화음 없음')):
        err, tot = by_chord_ctx[k]
        if tot == 0:
            continue
        acc = max(0.0, 1 - err / tot) * 100
        print(f"  {label}: Acc={acc:5.1f}%  (오류 {err}/{tot})")

    print("\n=== 셋잇단음표 포함 마디 vs 미포함 마디 정확도 ===")
    for k, label in ((True, '3연음 포함'), (False, '3연음 없음')):
        err, tot = by_tuplet_ctx[k]
        if tot == 0:
            continue
        acc = max(0.0, 1 - err / tot) * 100
        print(f"  {label}: Acc={acc:5.1f}%  (오류 {err}/{tot})")


if __name__ == '__main__':
    main()
