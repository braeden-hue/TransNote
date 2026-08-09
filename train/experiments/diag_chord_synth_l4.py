"""화음(chord-*) 표본 부족 문제 해결용: 실사 대신 화음 비중을 높인 합성 데이터를
생성(generate_scores.py)한 뒤, OMRDataset의 page_level_noise 경로로 L4 노이즈를 강제
적용해 실제 촬영 조건을 흉내내고, 로컬 CPU에서 추론해 화음 피치 정확도를 진단한다
(2026-08-04, pod 중단 중이라 실사 신규 촬영 대신 합성으로 표본을 늘림).

Levenshtein 기반 정렬(difflib 아님 -- 큰 블록 오정렬 방지, eval_newage_realphotos_errors.py의
edit_distance_opcodes와 동일 목적을 로컬에 자체 구현)로 GT/PRED를 맞춰 chord- 토큰만 추려
matched/wrong_pitch/missing/extra로 분류한다.
"""
import argparse
import glob
import json
import os
from collections import Counter

import torch

from dataset import load_tokenizer, OMRDataset
from model import OmrSeq2Seq, SOS_ID, EOS_ID, PAD_ID, infer_arch_from_state_dict
from train import fix_chord_tokens, fix_span_tokens
from inference import EOS_BOOST, LONG_DECODE_THRESHOLD, LONG_DECODE_RAMP, INFER_MAX_LEN


def edit_distance_opcodes(a, b):
    """difflib.SequenceMatcher.get_opcodes()와 동일한 형식의 (tag, i1, i2, j1, j2) 리스트를
    표준 레벤슈타인 DP로 계산(치환 비용 1). 토큰 시퀀스가 짧은(수백 개) OMR 용도라 O(n*m)로 충분."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append(('equal', i - 1, i, j - 1, j)); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('replace', i - 1, i, j - 1, j)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('delete', i - 1, i, j, j)); i -= 1
        else:
            ops.append(('insert', i, i, j - 1, j)); j -= 1
    ops.reverse()
    # 인접한 동일 tag 병합(단일 토큰 단위로 나온 걸 합쳐 개수만 줄임, 의미상 동일)
    merged = []
    for tag, i1, i2, j1, j2 in ops:
        if merged and merged[-1][0] == tag and merged[-1][2] == i1 and merged[-1][4] == j1:
            merged[-1] = (tag, merged[-1][1], i2, merged[-1][3], j2)
        else:
            merged.append((tag, i1, i2, j1, j2))
    return merged


@torch.no_grad()
def greedy_decode_from_tensor(seq2seq, inp, device, max_len=INFER_MAX_LEN):
    """inference.greedy_decode와 동일 로직이지만, 이미 정규화+채널구성된 canvas 텐서
    [1,C,H,W]를 직접 받는다(OMRDataset이 만든 텐서를 그대로 재사용하기 위함)."""
    seq2seq.eval()
    memory = seq2seq.encode(inp)
    kv_cache = seq2seq.precompute_memory_kv(memory)
    past = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    result = []
    for step in range(max_len):
        logits = seq2seq.decode_step_cached(kv_cache, past)
        logits[0, EOS_ID] *= EOS_BOOST
        if step > LONG_DECODE_THRESHOLD:
            logits[0, EOS_ID] *= 1.0 + (step - LONG_DECODE_THRESHOLD) * LONG_DECODE_RAMP
        nxt = int(logits.argmax(-1).item())
        if nxt == EOS_ID:
            break
        if nxt != PAD_ID:
            result.append(nxt)
        past = torch.cat([past, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--tokenizer', default='tokenizer258.json')
    ap.add_argument('--n', type=int, default=0, help='0이면 전체')
    args = ap.parse_args()

    device = torch.device('cpu')
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.seq2seq, map_location=device, weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    in_ch = arch.get('in_ch', 1)
    print(f"모델 로드: {args.seq2seq}  in_ch={in_ch}  device={device}")

    ds = OMRDataset(args.data_dir, tok2id, augment=True,
                     noise_level=4, noise_level_max=None, page_level_noise=True,
                     in_ch=in_ch)
    n = len(ds) if args.n <= 0 else min(args.n, len(ds))
    print(f"표본: {n}/{len(ds)}장(시스템/행 단위)")

    acc_list = []
    chord_counter = Counter()
    gt_chord_total = 0
    import time
    t0 = time.time()
    for idx in range(n):
        canvas_t, tgt_in, tgt_out = ds[idx]
        gt_ids = tgt_out[:-1].tolist()  # EOS 제외
        inp = canvas_t.unsqueeze(0).to(device)
        pred_ids = greedy_decode_from_tensor(seq2seq, inp, device)
        pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)

        gt_toks = [id2tok.get(i, '') for i in gt_ids]
        pred_toks = [id2tok.get(i, '') for i in pred_ids]

        if gt_toks == pred_toks:
            acc = 100.0
        else:
            ops = edit_distance_opcodes(gt_toks, pred_toks)
            n_err = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != 'equal')
            acc = max(0.0, 1 - n_err / max(1, len(gt_toks))) * 100
        acc_list.append(acc)

        ops = edit_distance_opcodes(gt_toks, pred_toks)
        for tag, i1, i2, j1, j2 in ops:
            gt_seg = gt_toks[i1:i2]
            pred_seg = pred_toks[j1:j2]
            if tag == 'equal':
                gt_chord_total += sum(1 for t in gt_seg if t.startswith('chord-'))
                chord_counter['matched'] += sum(1 for t in gt_seg if t.startswith('chord-'))
            elif tag == 'delete':
                gt_chord_total += sum(1 for t in gt_seg if t.startswith('chord-'))
                chord_counter['missing'] += sum(1 for t in gt_seg if t.startswith('chord-'))
            elif tag == 'insert':
                chord_counter['extra'] += sum(1 for t in pred_seg if t.startswith('chord-'))
            elif tag == 'replace':
                gt_chords = sum(1 for t in gt_seg if t.startswith('chord-'))
                pred_chords = sum(1 for t in pred_seg if t.startswith('chord-'))
                gt_chord_total += gt_chords
                matched_pitch = 0
                for t in gt_seg:
                    if t.startswith('chord-') and t in pred_seg:
                        matched_pitch += 1
                chord_counter['wrong_pitch'] += max(0, min(gt_chords, pred_chords) - matched_pitch)
                chord_counter['matched'] += matched_pitch
                chord_counter['missing'] += max(0, gt_chords - pred_chords - (gt_chords - matched_pitch - max(0, min(gt_chords, pred_chords) - matched_pitch)))
                chord_counter['extra'] += max(0, pred_chords - gt_chords)

        if (idx + 1) % 5 == 0 or idx == n - 1:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{n}] 누적 평균 Acc={sum(acc_list)/len(acc_list):.1f}%  "
                  f"경과={elapsed:.1f}s  (장당 {elapsed/(idx+1):.1f}s)")

    print(f"\n=== 합성 화음진단(L4 noise) 표본={n} ===")
    print(f"전체 정확도: 평균 {sum(acc_list)/len(acc_list):.1f}%  중앙값 {sorted(acc_list)[len(acc_list)//2]:.1f}%")
    print(f"\nGT chord- 토큰 총수: {gt_chord_total}")
    for k in ('matched', 'wrong_pitch', 'missing', 'extra'):
        print(f"  {k:12s} {chord_counter[k]}")
    matched_or_wrong = chord_counter['matched'] + chord_counter['wrong_pitch']
    if matched_or_wrong:
        print(f"화음 위치 recall(맞았든 틀렸든 화음으로는 인식): {matched_or_wrong}/{gt_chord_total} "
              f"({100*matched_or_wrong/max(1,gt_chord_total):.1f}%)")
        print(f"화음으로 인식된 것 중 피치까지 정확: {chord_counter['matched']}/{matched_or_wrong} "
              f"({100*chord_counter['matched']/max(1,matched_or_wrong):.1f}%)")


if __name__ == '__main__':
    main()
