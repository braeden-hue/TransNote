"""
4b_1류(단일 duration) 진단용: 검증셋 전체에 대해 GT vs PRED 토큰을 정렬해서
오류를 종류별(음높이/길이/note-rest 혼동/구조 토큰/누락/과잉)로 집계.

사용법:
    python3 error_breakdown.py --seq2seq <ckpt.pt> --tokenizer <tok.json> \
        --data_dir <dir> --val_ratio 0.1 --seed 42
"""
import argparse
import difflib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from inference import run_image
from model import OmrSeq2Seq, infer_arch_from_state_dict
from dataset import load_tokenizer


def token_kind(tok: str) -> str:
    if tok.startswith('note-'):
        return 'note'
    if tok.startswith('dur-'):
        return 'dur'
    if tok.startswith('rest-'):
        return 'rest'
    if tok.startswith('barline'):
        return 'barline'
    if tok in ('clef-G', 'clef-F') or tok.startswith('clef-'):
        return 'clef'
    if tok.startswith('key-'):
        return 'key'
    if tok.startswith('time-'):
        return 'time'
    if tok == 'staff-bass':
        return 'staff-bass'
    return 'other'


def classify_replace(g: str, p: str, counter: Counter, pitch_confusion: Counter = None):
    gk, pk = token_kind(g), token_kind(p)
    if gk == 'note' and pk == 'note':
        gp, gd = g.split('-')[1], None
        pp = p.split('-')[1]
        g_letter, p_letter = gp[0], pp[0]
        if g_letter == p_letter:
            counter['pitch_wrong_same_letter(옥타브/변화표만 틀림)'] += 1
        else:
            counter['pitch_wrong_diff_letter(음이름 자체 틀림)'] += 1
        if pitch_confusion is not None:
            pitch_confusion[f'{gp} -> {pp}'] += 1
    elif gk == 'dur' and pk == 'dur':
        counter['duration_wrong(길이 종류 혼동)'] += 1
    elif gk == 'note' and pk == 'rest':
        counter['note_to_rest(음표를 쉼표로 잘못 인식)'] += 1
    elif gk == 'rest' and pk == 'note':
        counter['rest_to_note(쉼표를 음표로 잘못 인식)'] += 1
    elif gk == pk:
        counter[f'{gk}_wrong(같은 종류끼리 다른 값)'] += 1
    else:
        counter[f'kind_confused({gk}->{pk})'] += 1


def dotted8_context_mask(gt_toks):
    """점8분음표(dur-3/16) 본인 또는 바로 다음 음(통상 그 뒤에 오는 16분음표)에 해당하는
    note- 토큰 인덱스 집합. '점8분+16분' 리듬 셀의 계이름 정확도만 따로 보기 위한 필터."""
    mask = set()
    n = len(gt_toks)
    for i, tok in enumerate(gt_toks):
        if not tok.startswith('note-'):
            continue
        is_dotted8_itself = i + 1 < n and gt_toks[i + 1] == 'dur-3/16'
        follows_dotted8 = i >= 1 and gt_toks[i - 1] == 'dur-3/16'
        if is_dotted8_itself or follows_dotted8:
            mask.add(i)
    return mask


def tie_context_mask(gt_toks):
    """붙임줄(tie) 토큰 자신 + 바로 다음 note- 토큰(연결되는 음표) 인덱스 집합.
    2026-07-30 사용자 보고: 이전 학습에서 tie 미인식이 그 뒤 음표 인식까지 연쇄적으로
    무너뜨렸음 -- 전체 val_acc(토큰 평균)엔 이 구간 하나의 개선 여부가 묻히므로,
    tie 자체와 그 착지점 정확도만 따로 본다(dotted8_context_mask와 동일 패턴)."""
    mask = set()
    n = len(gt_toks)
    for i, tok in enumerate(gt_toks):
        if tok != 'tie':
            continue
        mask.add(i)
        j = i + 1
        while j < n and not gt_toks[j].startswith('note-'):
            j += 1
        if j < n:
            mask.add(j)
    return mask


def classify_indel(tag: str, tok: str, counter: Counter, pitch_confusion: Counter = None):
    k = token_kind(tok)
    label = '누락(missing)' if tag == 'delete' else '과잉(extra)'
    counter[f'{label}_{k}'] += 1
    if k == 'note' and pitch_confusion is not None:
        p = tok.split('-')[1]
        pitch_confusion[f'{p} -> ({label})'] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--beam_width', type=int, default=1,
                     help='1(기본)=greedy, 2 이상이면 beam search')
    args = ap.parse_args()

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    tok2id, id2tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    # CoordConv(in_ch=2)나 높이 부분 보존(pool_h>1) 체크포인트도 호출부 수정 없이
    # 그대로 불러오도록, 저장된 가중치에서 아키텍처를 그대로 역산한다(2026-07-31/08-01).
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}  device={device}")

    fnames = sorted(f for f in os.listdir(args.data_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    n = len(fnames)
    n_val = max(1, int(n * args.val_ratio))
    perm = np.random.default_rng(args.seed).permutation(n).tolist()
    val_idx = perm[:n_val]
    val_fnames = [fnames[i] for i in val_idx]
    print(f"검증 샘플 수: {len(val_fnames)} / 전체 {n}")

    counter = Counter()
    pitch_confusion = Counter()
    exact_match = 0
    total = 0
    sample_mistakes = []
    dotted8_total = 0
    dotted8_pitch_correct = 0
    tie_total = 0
    tie_correct = 0

    for fname in val_fnames:
        stem = os.path.splitext(fname)[0]
        img_p = os.path.join(args.data_dir, fname)
        gt_p = os.path.join(args.data_dir, stem + '.json')
        if not os.path.isfile(gt_p):
            continue
        with open(gt_p, encoding='utf-8') as f:
            data = json.load(f)
        gt_toks = [t for t in data.get('tokens', []) if t not in ('<SOS>', '<EOS>', '<PAD>')]
        pred_toks = run_image(img_p, seq2seq, tok2id, id2tok, device, beam_width=args.beam_width)

        total += 1
        mask = dotted8_context_mask(gt_toks)
        tmask = tie_context_mask(gt_toks)
        if gt_toks == pred_toks:
            exact_match += 1
            dotted8_total += len(mask)
            dotted8_pitch_correct += len(mask)
            tie_total += len(tmask)
            tie_correct += len(tmask)
            continue

        sm = difflib.SequenceMatcher(a=gt_toks, b=pred_toks, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    if (i1 + k) in mask:
                        dotted8_total += 1
                        dotted8_pitch_correct += 1
                    if (i1 + k) in tmask:
                        tie_total += 1
                        tie_correct += 1
                continue
            elif tag == 'replace':
                # 위치 맞춰서 짝지어 비교 (길이 다르면 남는 쪽은 삽입/삭제 취급)
                glen, plen = i2 - i1, j2 - j1
                m = min(glen, plen)
                for k in range(m):
                    gi = i1 + k
                    classify_replace(gt_toks[gi], pred_toks[j1 + k], counter, pitch_confusion)
                    if gi in mask:
                        dotted8_total += 1
                        if gt_toks[gi] == pred_toks[j1 + k]:
                            dotted8_pitch_correct += 1
                    if gi in tmask:
                        tie_total += 1
                        if gt_toks[gi] == pred_toks[j1 + k]:
                            tie_correct += 1
                for k in range(m, glen):
                    gi = i1 + k
                    classify_indel('delete', gt_toks[gi], counter, pitch_confusion)
                    if gi in mask:
                        dotted8_total += 1
                    if gi in tmask:
                        tie_total += 1
                for k in range(m, plen):
                    classify_indel('insert', pred_toks[j1 + k], counter, pitch_confusion)
            elif tag == 'delete':
                for k in range(i1, i2):
                    classify_indel('delete', gt_toks[k], counter, pitch_confusion)
                    if k in mask:
                        dotted8_total += 1
                    if k in tmask:
                        tie_total += 1
            elif tag == 'insert':
                for k in range(j1, j2):
                    classify_indel('insert', pred_toks[k], counter, pitch_confusion)

        if len(sample_mistakes) < 8:
            sample_mistakes.append((fname, gt_toks, pred_toks))

    print(f"\n완전 일치: {exact_match}/{total} ({exact_match/total*100:.1f}%)")
    if dotted8_total > 0:
        pct = dotted8_pitch_correct / dotted8_total * 100
        print(f"\n=== 점8분+16분음표 리듬 셀 계이름 정확도: "
              f"{dotted8_pitch_correct}/{dotted8_total} ({pct:.1f}%) ===")
    else:
        print("\n=== 점8분+16분음표 리듬 셀: 검증셋에 등장 없음(0건) ===")
    if tie_total > 0:
        pct = tie_correct / tie_total * 100
        print(f"\n=== 붙임줄(tie) 및 착지점 정확도: {tie_correct}/{tie_total} ({pct:.1f}%) "
              f"-- 이전 학습에서 이 지점 미인식이 그 뒤 음표 인식까지 무너뜨린 것으로 보고됨 ===")
    else:
        print("\n=== 붙임줄(tie): 검증셋에 등장 없음(0건) ===")
    print(f"\n=== 오류 종류별 집계 (전체 {sum(counter.values())}건) ===")
    for k, v in counter.most_common():
        print(f"  {k:55s} {v:5d}")

    print(f"\n=== 음표별 오인식 결과표 (GT 음이름 -> PRED/처리, 빈도순, 상위 30) ===")
    print(f"  {'GT -> PRED':30s} {'건수':>5s}")
    for k, v in pitch_confusion.most_common(30):
        print(f"  {k:30s} {v:5d}")

    print(f"\n=== 오류 샘플 예시 (최대 8개) ===")
    for fname, gt, pred in sample_mistakes:
        print(f"\n-- {fname} --")
        print(f"  GT  : {gt}")
        print(f"  PRED: {pred}")


if __name__ == '__main__':
    main()
