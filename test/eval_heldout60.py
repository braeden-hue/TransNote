"""newage held-out 10곡 + 클래식 held-out 50곡(r8_2_diversity 학습에 쓰인 50곡 제외) =
총 60곡 실사 검증(2026-08-03). 레지스터 편향 교정 학습이 newage 전용 튜닝이 아니라
일반적으로 통하는지 확인하기 위해, 서로 다른 두 도메인(뉴에이지/클래식)을 한 번에 검증.
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq, infer_arch_from_state_dict
from train import fix_chord_tokens, fix_span_tokens
from eval_newage_realphotos_errors import edit_distance_opcodes

HERE = os.path.dirname(__file__)
GT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
PHOTO_ROOT = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', default=os.path.join(HERE, 'tokenizer258.json'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.seq2seq, map_location=device, weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}  device={device}")

    songs = NEWAGE_HELDOUT + CLASSICAL_HELDOUT
    print(f"검증 대상: newage held-out {len(NEWAGE_HELDOUT)}곡 + 클래식 held-out {len(CLASSICAL_HELDOUT)}곡 = {len(songs)}곡")

    newage_accs, classical_accs = [], []
    per_song_acc = {}

    for song in songs:
        gt_path = os.path.join(GT_DIR, song + '.json')
        photo_dir = os.path.join(PHOTO_ROOT, song)
        if not os.path.isfile(gt_path) or not os.path.isdir(photo_dir):
            print(f"[{song}] GT/사진 없음, 건너뜀")
            continue
        with open(gt_path, encoding='utf-8') as f:
            gt_toks = [t for t in json.load(f)['tokens'] if t not in ('<SOS>', '<EOS>', '<PAD>')]
        photos = sorted(glob.glob(os.path.join(photo_dir, '*.jpg')) +
                         glob.glob(os.path.join(photo_dir, '*.jpeg')))
        if not photos:
            print(f"[{song}] 사진 없음, 건너뜀")
            continue

        song_accs = []
        for photo in photos:
            try:
                pred_toks = run_image(photo, seq2seq, tok2id, id2tok, device, beam_width=1)
            except Exception as exc:
                print(f"  [{song}/{os.path.basename(photo)}] 추론 실패: {exc}")
                continue
            pred_ids = [tok2id[t] for t in pred_toks if t in tok2id]
            pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
            pred_toks = [id2tok[i] for i in pred_ids]

            if gt_toks == pred_toks:
                acc = 100.0
            else:
                n_err = sum(max(i2 - i1, j2 - j1)
                            for tag, i1, i2, j1, j2 in edit_distance_opcodes(gt_toks, pred_toks)
                            if tag != 'equal')
                acc = max(0.0, 1 - n_err / max(1, len(gt_toks))) * 100
            song_accs.append(acc)

        if song_accs:
            avg = sum(song_accs) / len(song_accs)
            per_song_acc[song] = avg
            (newage_accs if song.startswith('newage') else classical_accs).append(avg)
            print(f"[{song}] {len(song_accs)}장 평균 Acc={avg:.1f}%")

    if newage_accs:
        print(f"\n=== newage held-out {len(newage_accs)}곡 평균: {sum(newage_accs)/len(newage_accs):.1f}% ===")
    if classical_accs:
        print(f"=== 클래식 held-out {len(classical_accs)}곡 평균: {sum(classical_accs)/len(classical_accs):.1f}% ===")
    all_accs = newage_accs + classical_accs
    if all_accs:
        print(f"=== 전체 {len(all_accs)}곡 평균: {sum(all_accs)/len(all_accs):.1f}% ===")


if __name__ == '__main__':
    main()
