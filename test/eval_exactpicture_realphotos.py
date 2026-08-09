"""exactPicture 89곡의 "실제 폰카메라 사진"으로 실측 (깨끗한 렌더링이 아니라 진짜 촬영본).

정답은 exactpicture_test_full/<song>.json(이미 4마디로 트리밍된 현재 스코프 포맷)을 그대로
쓰고, 사진은 data/local_pools/exactPicture/<song>/*.jpg (다수의 실제 촬영 파일)을 전부 돈다.
Round4(노이즈 증강) 이전/이후 체크포인트를 이걸로 비교해야 진짜 노이즈 강건성 개선을 알 수 있다
-- exactpicture_test_full(깨끗한 렌더링) 기준 정확도는 노이즈 학습 효과를 볼 수 없음.
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
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS

HERE = os.path.dirname(__file__)
GT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
PHOTO_ROOT = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq2seq', required=True)
    ap.add_argument('--tokenizer', default=os.path.join(HERE, 'tokenizer258.json'))
    ap.add_argument('--gt_dir', default=GT_DIR)
    ap.add_argument('--photo_root', default=PHOTO_ROOT)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--max_photos_per_song', type=int, default=0,
                     help='0=전부. 곡당 촬영본이 많아 전체 실행이 오래 걸리면 제한(예: 3)')
    args = ap.parse_args()
    exclude = set(s.strip() for s in args.exclude.split(',') if s.strip())

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    ckpt = torch.load(args.seq2seq, map_location=device, weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}  device={device}")

    songs = sorted(f[:-5] for f in os.listdir(args.gt_dir) if f.endswith('.json'))
    songs = [s for s in songs if s not in exclude]
    print(f"대상 {len(songs)}곡")

    all_acc = []
    per_song_acc = {}
    for song in songs:
        gt_path = os.path.join(args.gt_dir, song + '.json')
        photo_dir = os.path.join(args.photo_root, song)
        if not os.path.isdir(photo_dir):
            continue
        with open(gt_path, encoding='utf-8') as f:
            gt_tokens = [t for t in json.load(f)['tokens'] if t not in ('<SOS>', '<EOS>', '<PAD>')]
        gt_ids = [tok2id[t] for t in gt_tokens if t in tok2id]

        photos = sorted(glob.glob(os.path.join(photo_dir, '*.jpg')) +
                         glob.glob(os.path.join(photo_dir, '*.jpeg')))
        if args.max_photos_per_song > 0:
            photos = photos[:args.max_photos_per_song]
        if not photos:
            continue

        song_accs = []
        for photo in photos:
            pname = os.path.basename(photo)
            try:
                pred_tokens = run_image(photo, seq2seq, tok2id, id2tok, device, beam_width=1)
            except Exception as exc:
                print(f"  [{song}/{pname}] 추론 실패: {exc}")
                continue
            pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]
            pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
            ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
            acc = max(0.0, 1 - ter) * 100
            all_acc.append(acc)
            song_accs.append(acc)
        if song_accs:
            per_song_acc[song] = sum(song_accs) / len(song_accs)
            print(f"[{song}] {len(song_accs)}장 평균 Acc={per_song_acc[song]:.1f}%")

    if all_acc:
        print(f"\n=== 전체 실사 촬영 사진 평균 Acc: {sum(all_acc)/len(all_acc):.1f}% (n={len(all_acc)}장, {len(per_song_acc)}곡) ===")
        worst = sorted(per_song_acc.items(), key=lambda x: x[1])[:10]
        print("\n=== 최저 성적 10곡 ===")
        for song, acc in worst:
            print(f"  {song}: {acc:.1f}%")


if __name__ == '__main__':
    main()
