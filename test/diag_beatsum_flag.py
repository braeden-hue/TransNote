"""1단계(진단 전용): 대보표 예측 결과에서 "치(treble) 마디 박자 합"이 이미 검증된 박자표의
기대 총합과 안 맞는 마디를 플래그만 하고, 그 마디가 실제로 GT와 다른지(missing/extra
note·dur 오류가 있는지) 대조해서 catch-rate(recall)/정밀도(precision)를 측정한다.

원리: InlineTimeCorrector가 이미 베이스 마디 박자합 다수결로 박자표를 검증하므로, 그
박자표의 "기대 총 박자"를 알고 있다. 치 마디도 같은 박자표를 따라야 하므로, 치 마디 합이
기대치와 다르면 그 마디에서 음표 과잉/누락이 있었다는 신호가 된다. 아직 교정은 안 하고
"이 방법으로 실제 오류의 몇 %를 잡아낼 수 있는지"만 먼저 측정(2026-08-05).

모델/디코딩은 전혀 안 건드림 -- run_image()의 최종 예측 토큰을 사후에 마디 단위로
쪼개서 분석만 한다.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_tokenizer
from inference import (
    run_image, _extract_treble_bass, _measure_beat_sum, _TIME_SIG_BEATS,
    _BARLINE_TOKEN_STRS, _HEADER_PREFIXES,
)
from model import OmrSeq2Seq, infer_arch_from_state_dict
from train import fix_chord_tokens, fix_span_tokens

HERE = os.path.dirname(__file__)
GT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
PHOTO_ROOT = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')
SONGS = ['sonatine_22_30', 'sonatine_23_38', 'sonatine_23_42',
         'sonatine_32_38', 'sonatine_36_60', 'sonatine_81_92',
         'newage21', 'newage22', 'newage23', 'newage24', 'newage25', 'newage26']
CKPT = os.path.join(HERE, 'checkpoints', 'r15_cropfix_coordconv', 'seq2seq_best.pt')
TOKENIZER = os.path.join(HERE, 'tokenizer258.json')


def split_paired_measures(toks):
    """인터리빙 시퀀스를 (치 마디, 베이스 마디) 쌍의 리스트로 쪼갠다(헤더 토큰 제외).

    barline은 인터리빙 포맷상 항상 베이스 세그먼트 처리 중에만 등장한다(치 쪽에는
    barline이 전혀 안 붙음) -- 이미 분리된 치 전용 리스트만으로 barline을 찾으려 하면
    항상 0마디가 나오는 버그(InlineTimeCorrector/correct_time_signature가 이미 한 번
    겪었던 것과 동일 클래스)를 피하려고, 원본 인터리빙 시퀀스를 직접 순회한다."""
    pairs = []
    cur_treble, cur_bass = [], []
    in_bass = False
    for t in toks:
        if t == 'staff-bass':
            in_bass = True
            continue
        if any(t.startswith(p) for p in _HEADER_PREFIXES):
            continue
        if in_bass:
            cur_bass.append(t)
        else:
            cur_treble.append(t)
        if t in _BARLINE_TOKEN_STRS:
            pairs.append((cur_treble, cur_bass))
            cur_treble, cur_bass = [], []
            in_bass = False
    return pairs


def best_time_sig(paired_measures):
    """베이스 마디 박자합 다수결로 박자표 추정(correct_time_signature와 동일 로직)."""
    sums = [_measure_beat_sum(bass_m) for _treble_m, bass_m in paired_measures]
    if not sums:
        return None
    majority_sum, _ = Counter(sums).most_common(1)[0]
    for sig, beats in _TIME_SIG_BEATS.items():
        if beats == majority_sum:
            return sig, beats
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(TOKENIZER)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id), **arch).to(device)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.ckpt}  device={device}")

    total_measures = 0
    flagged = 0
    flagged_and_wrong = 0   # 플래그 됐고 실제로도 GT와 다름 (True Positive)
    flagged_but_right = 0   # 플래그 됐지만 실제로는 GT와 같음 (False Positive)
    not_flagged_but_wrong = 0  # 안 플래그 됐지만 실제로는 GT와 다름 (False Negative, 놓침)
    not_flagged_and_right = 0  # 안 플래그, 실제로도 맞음 (True Negative)
    skipped_no_timesig = 0
    skipped_count_mismatch = 0

    for song in SONGS:
        gt_path = os.path.join(GT_DIR, song + '.json')
        photo_dir = os.path.join(PHOTO_ROOT, song)
        with open(gt_path, encoding='utf-8') as f:
            gt_toks = [t for t in json.load(f)['tokens'] if t not in ('<SOS>', '<EOS>', '<PAD>')]
        gt_pairs = split_paired_measures(gt_toks)

        photos = sorted(glob.glob(os.path.join(photo_dir, '*.jpg')) +
                         glob.glob(os.path.join(photo_dir, '*.jpeg')))
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

            pred_pairs = split_paired_measures(pred_toks)

            sig_info = best_time_sig(pred_pairs)
            if sig_info is None:
                skipped_no_timesig += len(pred_pairs)
                continue
            _sig, expected_beats = sig_info

            if len(pred_pairs) != len(gt_pairs):
                # 마디 수 자체가 다름(음표가 마디 경계를 넘나들며 붕괴) -- 이 사진은
                # 마디별 1:1 대조가 불가능하므로 건너뜀(대신 그 자체가 이미 강한 오류 신호).
                skipped_count_mismatch += len(pred_pairs)
                continue

            for i, (pt, _pb) in enumerate(pred_pairs):
                total_measures += 1
                pred_sum = _measure_beat_sum(pt)
                is_flagged = (pred_sum != expected_beats)
                gt_treble_i = gt_pairs[i][0]
                is_actually_wrong = (pt != gt_treble_i)
                if is_flagged:
                    flagged += 1
                    if is_actually_wrong:
                        flagged_and_wrong += 1
                    else:
                        flagged_but_right += 1
                else:
                    if is_actually_wrong:
                        not_flagged_but_wrong += 1
                    else:
                        not_flagged_and_right += 1

    print(f"\n=== 마디 단위 결과(총 {total_measures}마디, "
          f"박자표 미확정 {skipped_no_timesig}마디 + 마디수 불일치 {skipped_count_mismatch}마디는 제외) ===")
    print(f"플래그됨(불일치 감지): {flagged}마디")
    print(f"  - 그 중 실제로도 틀림(True Positive):  {flagged_and_wrong}")
    print(f"  - 그 중 실제로는 맞음(False Positive):  {flagged_but_right}")
    print(f"플래그 안 됨: {total_measures - flagged}마디")
    print(f"  - 그 중 실제로는 틀림(False Negative, 놓침): {not_flagged_but_wrong}")
    print(f"  - 그 중 실제로도 맞음(True Negative):        {not_flagged_and_right}")

    actual_wrong = flagged_and_wrong + not_flagged_but_wrong
    if actual_wrong:
        recall = flagged_and_wrong / actual_wrong
        print(f"\nRecall(실제 오류 마디 중 이 방법으로 잡아낸 비율): {recall*100:.1f}%")
    if flagged:
        precision = flagged_and_wrong / flagged
        print(f"Precision(플래그된 마디 중 실제로 오류였던 비율): {precision*100:.1f}%")


if __name__ == '__main__':
    main()
