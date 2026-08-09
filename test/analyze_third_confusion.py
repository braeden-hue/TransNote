"""GT vs PRED에서 3도(단3도/장3도, 줄-줄 또는 칸-칸) 음이름 오인식이 발생했을 때,
혼동된 두 음(정답/오답)이 서로 "근처"(같은 마디, 앞뒤 몇 토큰 안)에 배치돼 있었는지
분석. 사용자 질문: "두 음이 연속배치일 때 오류난 건지" 확인용.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path
import difflib

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq
from train import fix_chord_tokens, fix_span_tokens

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / 'data' / 'local_pools' / 'exactpicture_test50'
CKPT = _HERE.parent / 'secrets' / 'checkpoints' / 'seq2seq_r3_density_register_clef_best.pt'
TOKENIZER = _HERE / 'tokenizer258.json'

_NOTE_RE = re.compile(r'^note-([A-G])(#{1,2}|b{1,2})?(\d)$')
_STEP_ORDER = 'CDEFGAB'


def _step_octave(tok):
    m = _NOTE_RE.match(tok)
    if not m:
        return None
    return m.group(1), int(m.group(3))


def _degree(gt_tok, pred_tok):
    """두 note- 토큰 사이 음정 도수(옥타브 무시, 2도=2,3도=3,...)."""
    g = _step_octave(gt_tok)
    p = _step_octave(pred_tok)
    if g is None or p is None:
        return None
    diff = (_STEP_ORDER.index(p[0]) - _STEP_ORDER.index(g[0])) % 7
    return diff + 1


def _semitone_dist(gt_tok, pred_tok):
    m = re.match(r'^note-(.+)$', gt_tok)
    n = re.match(r'^note-(.+)$', pred_tok)
    if not m or not n:
        return None
    from music21.pitch import Pitch
    try:
        return abs(Pitch(m.group(1)).midi - Pitch(n.group(1)).midi)
    except Exception:
        return None


def _same_pitch_class(a_tok, b_tok):
    a = _step_octave(a_tok)
    b = _step_octave(b_tok)
    return a is not None and b is not None and a[0] == b[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(DATA_DIR))
    ap.add_argument('--ckpt', default=str(CKPT))
    ap.add_argument('--tokenizer', default=str(TOKENIZER))
    ap.add_argument('--context_window', type=int, default=12,
                     help='앞뒤 몇 토큰 안을 "근처"로 볼지(기본 12 -- 대략 반 마디~한 마디)')
    ap.add_argument('--local_max_semitones', type=int, default=5,
                     help='이 반음수 이내여야 "국지적(보표 인접 위치) 3도 오독"으로 분류. '
                          '이보다 멀면 옥타브가 다른 레지스터 붕괴가 우연히 음이름만 3도 '
                          '차이로 겹친 경우라 별도 집계(2026-07-31 발견된 방법론 오류 수정)')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.ckpt}\n")

    n_third_conf = 0
    n_local = 0               # 진짜 국지적 3도(인접 옥타브, <=local_max_semitones)
    n_register_collapse = 0   # 음이름만 3도, 실제론 옥타브 다른 레지스터 붕괴
    n_pred_pitch_nearby = 0   # (국지적 케이스만) PRED(오답)의 pitch class가 GT 근처에 이미 등장했는지
    n_gt_pitch_nearby = 0     # (국지적 케이스만) GT(정답)의 pitch class가 근처에도 등장하는지(대조군)
    n_line_line = 0
    n_space_space = 0
    examples = []
    collapse_examples = []

    json_paths = sorted(glob.glob(str(data_dir / '*.json')))
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

        sm = difflib.SequenceMatcher(a=gt_clean, b=pred_clean, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != 'replace':
                continue
            glen, plen = i2 - i1, j2 - j1
            m = min(glen, plen)
            for k in range(m):
                gi, pj = i1 + k, j1 + k
                g_tok, p_tok = gt_clean[gi], pred_clean[pj]
                if not (g_tok.startswith('note-') and p_tok.startswith('note-')):
                    continue
                deg = _degree(g_tok, p_tok)
                if deg != 3:
                    continue
                n_third_conf += 1

                sd = _semitone_dist(g_tok, p_tok)
                is_local = sd is not None and sd <= args.local_max_semitones
                if not is_local:
                    n_register_collapse += 1
                    if len(collapse_examples) < 8:
                        collapse_examples.append((stem, g_tok, p_tok, sd))
                    continue
                n_local += 1

                g_step = _step_octave(g_tok)
                # 줄-줄/칸-칸 판별: 음이름 알파벳 순서 기준 절대 위치(옥타브*7+스텝 인덱스)의
                # 홀짝으로 근사(실제 보표는 클렙에 따라 줄/칸 배정이 다르지만, "3도는 항상
                # 같은 타입끼리"라는 성질 자체는 클렙 무관이므로 홀짝 분리만으로 두 그룹을
                # 나누는 데는 문제 없음).
                abs_idx = g_step[1] * 7 + _STEP_ORDER.index(g_step[0])
                if abs_idx % 2 == 0:
                    n_line_line += 1
                else:
                    n_space_space += 1

                lo = max(0, gi - args.context_window)
                hi = min(len(gt_clean), gi + args.context_window + 1)
                context = gt_clean[lo:gi] + gt_clean[gi+1:hi]
                pred_pitch_nearby = any(
                    t.startswith('note-') and _same_pitch_class(t, p_tok) for t in context)
                gt_pitch_nearby = any(
                    t.startswith('note-') and _same_pitch_class(t, g_tok) for t in context)
                if pred_pitch_nearby:
                    n_pred_pitch_nearby += 1
                if gt_pitch_nearby:
                    n_gt_pitch_nearby += 1

                if len(examples) < 15:
                    examples.append((stem, g_tok, p_tok, pred_pitch_nearby, gt_pitch_nearby,
                                      gt_clean[lo:hi]))

    print(f"=== 음이름만 3도(mod 7) 차이인 오인식 총 {n_third_conf}건 ===")
    print(f"  국지적 3도(<= {args.local_max_semitones}반음, 진짜 보표 인접 위치 오독): {n_local}건")
    print(f"  레지스터 붕괴(음이름만 3도, 실제론 옥타브 다름 -- 별개 현상): {n_register_collapse}건")

    print(f"\n--- 국지적 3도 {n_local}건 상세 ---")
    print(f"  줄-줄(같은 옥타브 절대위치 짝수): {n_line_line}건, 칸-칸(홀수): {n_space_space}건")
    if n_local:
        print(f"  오답 음이 GT 근처(±{args.context_window}토큰)에 이미 등장: "
              f"{n_pred_pitch_nearby}/{n_local} ({n_pred_pitch_nearby/n_local*100:.1f}%)")
        print(f"  정답 음이 GT 근처에도 등장(대조군): "
              f"{n_gt_pitch_nearby}/{n_local} ({n_gt_pitch_nearby/n_local*100:.1f}%)")

    print(f"\n=== 국지적 3도 예시(최대 15개) ===")
    for stem, g, p, pred_near, gt_near, ctx in examples:
        print(f"\n-- {stem} -- GT={g} -> PRED={p}  (오답근처:{pred_near}, 정답근처:{gt_near})")
        print(f"   문맥: {' '.join(ctx)}")

    print(f"\n=== 레지스터 붕괴 예시(최대 8개, 참고용) ===")
    for stem, g, p, sd in collapse_examples:
        print(f"  {stem}: GT={g} -> PRED={p} ({sd}반음 차이)")


if __name__ == '__main__':
    main()
