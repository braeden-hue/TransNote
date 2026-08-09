"""r15_cropfix_coordconv 기준, 신규 검증 곡 오류 원인 분석. eval_newage_realphotos_errors.py의
분류 인프라(error_breakdown.py 재사용)를 그대로 쓰되 대상 곡을 학습에 쓰이지 않은 곡만으로
고정하고, PLAN_r16_hide_timesig.md의 "첫 이탈 토큰 종류" 분석(diag_new6_analysis.py, pod
전용이라 로컬엔 없음)을 추가해서 time- 외에 다른 원인이 있는지 확인한다(2026-08-04).

2026-08-05: sonatine 6곡(2026-08-03 촬영) + newage21~26(2026-08-04 촬영) 총 12곡으로 확장
-- 둘 다 r12_all120_realphotos(메인 학습 데이터)에 없음을 확인함(파일명 stem 대조).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq, infer_arch_from_state_dict
from error_breakdown import classify_replace, classify_indel, token_kind
from train import fix_chord_tokens, fix_span_tokens
from eval_newage_realphotos_errors import (
    edit_distance_opcodes, classify_replace_with_dur, classify_indel_with_dur,
)

HERE = os.path.dirname(__file__)
GT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
PHOTO_ROOT = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')
SONGS = ['sonatine_22_30', 'sonatine_23_38', 'sonatine_23_42',
         'sonatine_32_38', 'sonatine_36_60', 'sonatine_81_92',
         'newage21', 'newage22', 'newage23', 'newage24', 'newage25', 'newage26']
CKPT = os.path.join(HERE, 'checkpoints', 'r15_cropfix_coordconv', 'seq2seq_best.pt')
TOKENIZER = os.path.join(HERE, 'tokenizer258.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--songs', default=None,
                     help='쉼표로 구분한 곡 부분집합(기본 None=SONGS 전체)')
    args = ap.parse_args()
    songs = args.songs.split(',') if args.songs else SONGS

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(TOKENIZER)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.ckpt}  device={device}")

    counter = Counter()
    pitch_confusion = Counter()
    dur_confusion = Counter()
    per_song_pitch_confusion = defaultdict(Counter)
    first_divergence_kind = Counter()  # PLAN_r16류 분석: 첫 어긋난 토큰의 종류
    all_acc = []
    per_song_acc = {}

    for song in songs:
        gt_path = os.path.join(GT_DIR, song + '.json')
        photo_dir = os.path.join(PHOTO_ROOT, song)
        with open(gt_path, encoding='utf-8') as f:
            gt_toks = [t for t in json.load(f)['tokens'] if t not in ('<SOS>', '<EOS>', '<PAD>')]
        photos = sorted(glob.glob(os.path.join(photo_dir, '*.jpg')) +
                         glob.glob(os.path.join(photo_dir, '*.jpeg')))
        song_accs = []
        for photo in photos:
            pname = os.path.basename(photo)
            try:
                pred_toks = run_image(photo, seq2seq, tok2id, id2tok, device, beam_width=1)
            except Exception as exc:
                print(f"  [{song}/{pname}] 추론 실패: {exc}")
                continue
            pred_ids = [tok2id[t] for t in pred_toks if t in tok2id]
            pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
            pred_toks = [id2tok[i] for i in pred_ids]

            # 첫 이탈 지점: gt/pred를 앞에서부터 순서대로 비교해 처음 달라지는 gt 토큰의 종류
            first_i = None
            for i, (g, p) in enumerate(zip(gt_toks, pred_toks)):
                if g != p:
                    first_i = i
                    break
            else:
                if len(gt_toks) != len(pred_toks):
                    first_i = min(len(gt_toks), len(pred_toks))
            if first_i is not None and first_i < len(gt_toks):
                first_divergence_kind[token_kind(gt_toks[first_i])] += 1
            elif first_i is None:
                first_divergence_kind['(완전일치)'] += 1

            if gt_toks == pred_toks:
                acc = 100.0
            else:
                n_err = 0
                for tag, i1, i2, j1, j2 in edit_distance_opcodes(gt_toks, pred_toks):
                    if tag == 'equal':
                        continue
                    if tag == 'replace':
                        m = min(i2 - i1, j2 - j1)
                        for k in range(m):
                            classify_replace_with_dur(gt_toks[i1 + k], pred_toks[j1 + k],
                                                       counter, pitch_confusion, dur_confusion,
                                                       per_song_pitch_confusion[song])
                            n_err += 1
                        for k in range(m, i2 - i1):
                            classify_indel_with_dur('delete', gt_toks[i1 + k], counter,
                                                     pitch_confusion, dur_confusion,
                                                     per_song_pitch_confusion[song])
                            n_err += 1
                        for k in range(m, j2 - j1):
                            classify_indel_with_dur('insert', pred_toks[j1 + k], counter,
                                                     pitch_confusion, dur_confusion,
                                                     per_song_pitch_confusion[song])
                            n_err += 1
                    elif tag == 'delete':
                        for k in range(i1, i2):
                            classify_indel_with_dur('delete', gt_toks[k], counter,
                                                     pitch_confusion, dur_confusion,
                                                     per_song_pitch_confusion[song])
                            n_err += 1
                    elif tag == 'insert':
                        for k in range(j1, j2):
                            classify_indel_with_dur('insert', pred_toks[k], counter,
                                                     pitch_confusion, dur_confusion,
                                                     per_song_pitch_confusion[song])
                            n_err += 1
                acc = max(0.0, 1 - n_err / max(1, len(gt_toks))) * 100
            all_acc.append(acc)
            song_accs.append(acc)
            print(f"  [{song}/{pname}] Acc={acc:.1f}%")
        if song_accs:
            per_song_acc[song] = sum(song_accs) / len(song_accs)
            print(f"[{song}] {len(song_accs)}장 평균 Acc={per_song_acc[song]:.1f}%")

    if all_acc:
        print(f"\n=== 신규 6곡 전체 평균 Acc: {sum(all_acc)/len(all_acc):.1f}% (n={len(all_acc)}장) ===")

    print("\n=== 첫 이탈 토큰 종류 (극초반 이탈 캐스케이드 원인) ===")
    total_div = sum(v for k, v in first_divergence_kind.items() if k != '(완전일치)')
    for k, v in first_divergence_kind.most_common():
        pct = f"{100*v/total_div:.1f}%" if k != '(완전일치)' and total_div else ''
        print(f"  {k:15s} {v:5d}  {pct}")

    print("\n=== 오류 종류별 집계 ===")
    for k, v in counter.most_common(30):
        print(f"  {k:45s} {v:5d}")

    print("\n=== 음높이 혼동 상위 30 ===")
    for k, v in pitch_confusion.most_common(30):
        print(f"  {k:20s} {v:5d}")

    print("\n=== 길이(duration) 혼동 상위 30 ===")
    for k, v in dur_confusion.most_common(30):
        print(f"  {k:30s} {v:5d}")

    print("\n=== 곡별 음높이 혼동 상위 8 ===")
    for song in songs:
        if song not in per_song_pitch_confusion or not per_song_pitch_confusion[song]:
            continue
        top = per_song_pitch_confusion[song].most_common(8)
        items = ', '.join(f'{k}({v})' for k, v in top)
        print(f"  [{song}] {items}")


if __name__ == '__main__':
    main()
