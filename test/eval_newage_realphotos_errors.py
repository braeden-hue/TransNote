"""Round7 최종 검증 전용: newage 14곡(학습에 절대 포함된 적 없는 검증 전용 곡)의 실사
촬영 사진으로 정확도 + 오류 종류별 분석(음높이/길이 등)을 pod에서 직접 실행.

error_breakdown.py의 분류 로직(classify_replace/classify_indel)을 재사용하고, 여기에
duration confusion(어떤 길이가 어떤 길이로 잘못 인식되는지) 추적을 추가했다.
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

HERE = os.path.dirname(__file__)
GT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
PHOTO_ROOT = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')

# eval_heldout60.py와 동일한 held-out 목록(학습에 쓰이지 않은 곡). 순환 임포트를 피하기
# 위해 상수를 복사해서 둔다 -- eval_heldout60.py가 이 파일의 edit_distance_opcodes를 가져다 쓴다.
NEWAGE_HELDOUT = ['newage03', 'newage04', 'newage05', 'newage06', 'newage07',
                  'newage09', 'newage11', 'newage14', 'newage19', 'newage20']
CLASSICAL_HELDOUT = """chern1_57_9 chop18_183 chop18_189 chop18_29 chop18_71 chop34_1_250
chop64_2_33 chop64_3_16 chop64_3_21 chop_etude_25_9_1 fall_23_13 fall_23_5 fall_24_22
fall_24_27 sonata_103_192 sonata_150_104 sonata_27_68 sonata_30_44 sonata_33_18
sonata_34_32 sonata_35_67 sonata_37_114 sonata_47_67 sonata_88_104 sonata_98_57
sonata_98_79 sonata_98_84 sonatineHa_11_57 sonatineHa_12_21 sonatineHa_27_28
sonatineHa_29_78 sonatineHa_30_92 sonatineHa_33_32 sonatineHa_33_34 sonatineHa_34_48
sonatineHa_38_21 sonatineHa_59_35 sonatineHa_68_19 sonatineHa_80_24 sonatineHa_81_44
sonatineHa_94_82 sonatineHa_9_19 sonatineHa_9_23 sonatine_23_38 sonatine_38p_112
summer1_18p_9_2 summer_14p_5 summer_15p_13 summer_16p_33 summer_18p_13
winter_34_13""".split()


def edit_distance_opcodes(a, b):
    """difflib.SequenceMatcher.get_opcodes()와 같은 형식((tag, i1, i2, j1, j2))을
    반환하지만, 최소 편집거리(Levenshtein) 동적계획법 백트레이스로 정렬한다.

    2026-08-03: newage20 실측에서 difflib이 반복되는 유사 프레이즈(A-B-A' 구조)를 보고
    엉뚱한 위치끼리 매칭해 GT 58토큰을 통째로 "완전 누락" 처리하는 오류를 확인 --
    실제로는 accidental(#,b) 몇 개만 빼고 거의 전부 일치(Levenshtein 기준 68.2%)했는데
    difflib 방식으로는 0.0%로 나왔음. difflib의 longest-matching-block 휴리스틱은
    반복 구조가 있으면 전역 최적 정렬을 보장하지 않는 반면, 이 DP는 항상 최소 편집
    횟수를 보장하므로 이런 스퓨리어스한 큰 블록 오정렬이 나오지 않는다."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ai = a[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                row[j] = prev_row[j - 1]
            else:
                row[j] = 1 + min(prev_row[j - 1], prev_row[j], row[j - 1])

    steps = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            steps.append('equal'); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            steps.append('replace'); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            steps.append('delete'); i -= 1
        else:
            steps.append('insert'); j -= 1
    steps.reverse()

    opcodes = []
    ci = cj = 0
    idx = 0
    while idx < len(steps):
        tag = steps[idx]
        i1, j1 = ci, cj
        k = idx
        while k < len(steps) and steps[k] == tag:
            if tag in ('equal', 'replace'):
                ci += 1; cj += 1
            elif tag == 'delete':
                ci += 1
            else:
                cj += 1
            k += 1
        opcodes.append((tag, i1, ci, j1, cj))
        idx = k
    return opcodes


def classify_replace_with_dur(g, p, counter, pitch_confusion, dur_confusion, song_pitch_confusion=None):
    classify_replace(g, p, counter, pitch_confusion)
    if song_pitch_confusion is not None and token_kind(g) == 'note' and token_kind(p) in ('note', 'other'):
        classify_replace(g, p, Counter(), song_pitch_confusion)
    if token_kind(g) == 'dur' and token_kind(p) == 'dur':
        dur_confusion[f'{g} -> {p}'] += 1


def classify_indel_with_dur(tag, tok, counter, pitch_confusion, dur_confusion, song_pitch_confusion=None):
    classify_indel(tag, tok, counter, pitch_confusion)
    if song_pitch_confusion is not None and token_kind(tok) == 'note':
        classify_indel(tag, tok, Counter(), song_pitch_confusion)
    if token_kind(tok) == 'dur':
        label = '누락(missing)' if tag == 'delete' else '과잉(extra)'
        dur_confusion[f'{tok} -> ({label})'] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', default=os.path.join(HERE, 'tokenizer258.json'))
    ap.add_argument('--gt_dir', default=GT_DIR)
    ap.add_argument('--photo_root', default=PHOTO_ROOT)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max_photos_per_song', type=int, default=0, help='0=전부')
    ap.add_argument('--domain', choices=['newage', 'classical', 'all'], default='newage',
                     help='newage=newage*만(기존 기본값), classical=newage 제외 전부, all=전체')
    ap.add_argument('--heldout_only', action='store_true',
                     help='학습(r8_2_diversity 등)에 쓰이지 않은 held-out 곡만 대상으로 함'
                          '(eval_heldout60.py와 동일한 목록)')
    args = ap.parse_args()

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.seq2seq, map_location=device, weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}  device={device}")

    if args.domain == 'newage':
        songs = sorted(s[:-5] for s in os.listdir(args.gt_dir)
                        if s.endswith('.json') and s.startswith('newage'))
    elif args.domain == 'classical':
        songs = sorted(s[:-5] for s in os.listdir(args.gt_dir)
                        if s.endswith('.json') and not s.startswith('newage'))
    else:
        songs = sorted(s[:-5] for s in os.listdir(args.gt_dir) if s.endswith('.json'))
    if args.heldout_only:
        heldout = set(NEWAGE_HELDOUT) | set(CLASSICAL_HELDOUT)
        songs = [s for s in songs if s in heldout]
    print(f"검증 대상({args.domain}{'/heldout' if args.heldout_only else ''}): {len(songs)}곡 -> {songs}")

    counter = Counter()
    pitch_confusion = Counter()
    dur_confusion = Counter()
    per_song_pitch_confusion = defaultdict(Counter)  # 곡별로 어떤 음을 못 읽는지 추적(2026-08-02)
    all_acc = []
    per_song_acc = {}
    per_photo = []  # (song, pname, w, h, aspect, acc) -- 실사 이미지 크기/종횡비가
                     # 제각각이라(스마트폰 촬영 시 화면에 담기는 오선 수가 매번 달라짐),
                     # 크기/비율과 정확도의 상관을 눈으로 확인하기 위한 기록(2026-08-02).

    for song in songs:
        gt_path = os.path.join(args.gt_dir, song + '.json')
        photo_dir = os.path.join(args.photo_root, song)
        if not os.path.isdir(photo_dir):
            print(f"[{song}] 사진 폴더 없음, 건너뜀")
            continue
        with open(gt_path, encoding='utf-8') as f:
            gt_toks = [t for t in json.load(f)['tokens'] if t not in ('<SOS>', '<EOS>', '<PAD>')]

        photos = sorted(glob.glob(os.path.join(photo_dir, '*.jpg')) +
                         glob.glob(os.path.join(photo_dir, '*.jpeg')))
        if args.max_photos_per_song > 0:
            photos = photos[:args.max_photos_per_song]
        if not photos:
            print(f"[{song}] 사진 없음, 건너뜀")
            continue

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
            img = cv2.imread(photo)
            if img is not None:
                h, w = img.shape[:2]
                per_photo.append((song, pname, w, h, w / h, acc))
        if song_accs:
            per_song_acc[song] = sum(song_accs) / len(song_accs)
            print(f"[{song}] {len(song_accs)}장 평균 Acc={per_song_acc[song]:.1f}%")

    if all_acc:
        print(f"\n=== newage 전체 평균 Acc: {sum(all_acc)/len(all_acc):.1f}% (n={len(all_acc)}장, {len(per_song_acc)}곡) ===")

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

    print("\n=== 사진별 크기/종횡비 vs 정확도 (정확도 낮은 순) ===")
    print(f"  {'song/photo':45s} {'W':>5s} {'H':>5s} {'W/H':>6s} {'Acc':>7s}")
    for song, pname, w, h, aspect, acc in sorted(per_photo, key=lambda r: r[5]):
        print(f"  {song + '/' + pname:45s} {w:5d} {h:5d} {aspect:6.2f} {acc:6.1f}%")


if __name__ == '__main__':
    main()
