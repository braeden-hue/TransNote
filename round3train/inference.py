"""
round3train/inference.py  –  Round 3 PyTorch 추론 (대보표 지원).

대보표 처리 방식:
  - 짝수 인덱스 오선 (0, 2, ...) = treble
  - 홀수 인덱스 오선 (1, 3, ...) = bass
  - 같은 시스템의 treble/bass 사이에 'staff-bass' 토큰 삽입
  - 시스템 간 구분은 없음 (연속 읽기)

사용법:
    python round3train/inference.py \\
        --seq2seq round3train/models/seq2seq_best.pt \\
        --tokenizer musicscore_flutter/ml/data/tokenizer.json \\
        image.png

    # 일괄 평가
    python round3train/inference.py \\
        --seq2seq round3train/models/seq2seq_best.pt \\
        --tokenizer musicscore_flutter/ml/data/tokenizer.json \\
        --eval_dir "데이터 학습/Round3" \\
        --n_eval 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model import OmrSeq2Seq, SOS_ID, EOS_ID, PAD_ID, MAX_SEQ
from dataset import (preprocess, detect_staffs, extract_staff_canvas,
                     extract_system_canvas,
                     IMG_MEAN, IMG_STD, load_tokenizer)


# ─────────────────────────────────────────────────────────────────────────────
#  Decode constants
# ─────────────────────────────────────────────────────────────────────────────

INFER_MAX_LEN = 300   # 시스템 전체 인터리빙 시퀀스 기준 (treble+staff-bass+bass)
EOS_BOOST     = 1.5   # EOS logit 증폭 배율


@torch.no_grad()
def greedy_decode(seq2seq: OmrSeq2Seq, canvas: np.ndarray,
                  device: torch.device,
                  max_len: int = INFER_MAX_LEN) -> List[int]:
    tile_f = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    inp    = torch.from_numpy(tile_f).unsqueeze(0).unsqueeze(0).to(device)
    seq2seq.eval()
    memory   = seq2seq.encode(inp)
    kv_cache = seq2seq.precompute_memory_kv(memory)  # cross-attention K,V 1회 계산 (Step1 가속)
    past   = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    result = []
    for _ in range(max_len):
        logits = seq2seq.decode_step_cached(kv_cache, past)
        logits[0, EOS_ID] *= EOS_BOOST
        nxt = int(logits.argmax(-1).item())
        if nxt == EOS_ID: break
        if nxt != PAD_ID: result.append(nxt)
        past = torch.cat([past, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return result


class _Beam:
    __slots__ = ('seq', 'tokens', 'score', 'finished')

    def __init__(self, seq, tokens, score, finished):
        self.seq = seq            # [1, t] decoder input so far (incl. SOS)
        self.tokens = tokens       # generated token ids, excl. SOS/EOS/PAD
        self.score = score         # cumulative log-probability
        self.finished = finished   # hit EOS already


@torch.no_grad()
def beam_decode(seq2seq: OmrSeq2Seq, canvas: np.ndarray,
                device: torch.device,
                beam_width: int = 4,
                length_penalty: float = 0.7,
                max_len: int = INFER_MAX_LEN) -> List[int]:
    """
    ml/omr/engine의 DecoderRunner::decode_beam()과 동일한 알고리즘의 파이썬/PyTorch
    버전. C++ 쪽은 TFLite KV-cache라 빔마다 캐시를 복제해야 하지만, 이 PyTorch
    모델은 매 스텝 시퀀스 전체를 다시 돌리는 방식(캐시 없음)이라 빔마다 그냥
    누적된 입력 시퀀스(seq)를 들고 있으면 된다 — 로직은 동일:
      1. 완료되지 않은 빔마다 다음 토큰 분포를 구해 자기 top-k 후보를 만든다.
      2. 이미 끝난(EOS) 빔은 점수 고정한 채 그대로 다음 라운드 후보로 넘긴다.
      3. 전체 후보 중 top beam_width만 남긴다.
      4. 마지막에 길이 정규화 점수로 최선의 빔을 고른다.
    beam_width=1이면 greedy_decode와 동일해야 한다(검증용).
    """
    if beam_width <= 1:
        return greedy_decode(seq2seq, canvas, device, max_len)

    tile_f = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    inp    = torch.from_numpy(tile_f).unsqueeze(0).unsqueeze(0).to(device)
    seq2seq.eval()
    memory   = seq2seq.encode(inp)
    kv_cache = seq2seq.precompute_memory_kv(memory)  # cross-attention K,V 1회 계산 (Step1 가속)

    sos = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    beams = [_Beam(sos, [], 0.0, False)]

    for _ in range(max_len):
        if all(b.finished for b in beams):
            break

        candidates = []  # (score, parent_idx, token_id_or_None, advances)
        for i, b in enumerate(beams):
            if b.finished:
                candidates.append((b.score, i, None, False))
                continue
            logits = seq2seq.decode_step_cached(kv_cache, b.seq)  # [1, vocab]
            logits[0, EOS_ID] *= EOS_BOOST
            logp = torch.log_softmax(logits[0], dim=-1)
            logp[PAD_ID] = float('-inf')
            topk_logp, topk_idx = logp.topk(beam_width)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                candidates.append((b.score + lp, i, idx, True))

        candidates.sort(key=lambda c: c[0], reverse=True)
        candidates = candidates[:beam_width]

        new_beams = []
        for score, parent, tok, advances in candidates:
            parent_beam = beams[parent]
            if not advances:
                new_beams.append(parent_beam)
                continue
            if tok == EOS_ID:
                new_beams.append(_Beam(parent_beam.seq, parent_beam.tokens, score, True))
            else:
                new_seq = torch.cat(
                    [parent_beam.seq, torch.tensor([[tok]], dtype=torch.long, device=device)],
                    dim=1)
                new_tokens = parent_beam.tokens + ([tok] if tok != PAD_ID else [])
                new_beams.append(_Beam(new_seq, new_tokens, score, False))
        beams = new_beams

    best = max(beams, key=lambda b: b.score / max(len(b.tokens), 1) ** length_penalty)
    return best.tokens


# ─────────────────────────────────────────────────────────────────────────────
#  단일 이미지 추론
# ─────────────────────────────────────────────────────────────────────────────

def run_image(image_path: str, seq2seq: OmrSeq2Seq,
              tok2id: Dict[str, int], id2tok: Dict[int, str],
              device: torch.device) -> List[str]:
    """
    이미지 → 토큰 문자열 리스트.

    오선 감지 수 N에 따라:
    - N % 2 == 0 (≥ 2) : 대보표 → 시스템별 system canvas(384×1280, treble+bass 결합)로
                          greedy_decode 1회 → 전체 인터리빙 시퀀스 (staff-bass 자동 포함)
    - 그 외             : 단일 오선 순차 처리
    """
    SKIP_TOKS = {'<PAD>', '<SOS>', '<EOS>'}

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)

    gray   = preprocess(bgr)
    staffs = detect_staffs(gray)
    if not staffs:
        return []

    n = len(staffs)
    all_tokens: List[str] = []

    if n >= 2 and n % 2 == 0:
        # 대보표: 시스템 캔버스(treble+bass 수직 결합 384×1280)로 시퀀스 한 번에 예측.
        # 모델은 이 방식으로 학습됐음 — 개별 오선을 따로 디코딩하지 않는다.
        n_systems = n // 2
        for sys_i in range(n_systems):
            treble_staff  = staffs[sys_i * 2]
            bass_staff    = staffs[sys_i * 2 + 1]
            sys_canvas    = extract_system_canvas(gray, [treble_staff, bass_staff])
            sys_ids       = greedy_decode(seq2seq, sys_canvas, device)
            sys_ids       = fix_span_tokens(fix_chord_tokens(sys_ids, id2tok), id2tok)
            for tok_id in sys_ids:
                tok_str = id2tok.get(tok_id, '<UNK>')
                if tok_str not in SKIP_TOKS:
                    all_tokens.append(tok_str)

    else:
        # 단일 오선 또는 홀수 오선: 순서대로 이어붙임
        for staff in staffs:
            canvas = extract_staff_canvas(gray, staff)
            ids    = greedy_decode(seq2seq, canvas, device)
            ids    = fix_span_tokens(fix_chord_tokens(ids, id2tok), id2tok)
            for tok_id in ids:
                tok_str = id2tok.get(tok_id, '<UNK>')
                if tok_str not in SKIP_TOKS:
                    all_tokens.append(tok_str)

    return all_tokens


# ─────────────────────────────────────────────────────────────────────────────
#  배치 평가
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(a: List[int], b: List[int]) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            curr[j] = (prev[j-1] if a[i-1] == b[j-1]
                       else 1 + min(prev[j], curr[j-1], prev[j-1]))
        prev, curr = curr, prev
    return prev[n]


# ─────────────────────────────────────────────────────────────────────────────
#  포스트프로세싱 + 마디 단위 TER (OMR-NED 방식, 수정 1~4)
# ─────────────────────────────────────────────────────────────────────────────

_BARLINE_TOKEN_STRS = frozenset({
    'barline', 'barline-final', 'barline-start-repeat', 'barline-end-repeat',
})

_SPAN_PAIRS = {
    'slur-start':           'slur-end',
    'hairpin-cresc-start':  'hairpin-cresc-end',
    'hairpin-dim-start':    'hairpin-dim-end',
    'ottava-8va-start':     'ottava-8va-end',
    'ottava-8vb-start':     'ottava-8vb-end',
    'tuplet-3-start':       'tuplet-3-end',
}
_SPAN_ENDS = frozenset(_SPAN_PAIRS.values())

_NOTE_PREFIXES   = ('note-', 'chord-', 'rest-', 'dur-')
_HEADER_PREFIXES = ('clef-', 'key-', 'time-')


# ─────────────────────────────────────────────────────────────────────────────
#  분리 분석 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _extract_treble_bass(toks: List[str]) -> Tuple[List[str], List[str]]:
    """
    인터리빙 시퀀스 → (treble_toks, bass_toks) 분리.

    구조 (GT/PRED 동일 — 모델이 GT와 같은 인터리빙 포맷으로 학습·예측함,
    run_image()가 개별 오선을 따로 디코딩하지 않고 system canvas 하나로
    전체 인터리빙 시퀀스를 한 번에 생성하기 때문):
      [treble_m0] staff-bass [bass_m0] barline [treble_m1] staff-bass ...
      → barline은 bass 쪽에 포함, barline 이후 treble 모드로 복귀.

    staff-bass/barline 토글을 반복 처리하므로 마디 수와 무관하게 동작한다.
    """
    treble: List[str] = []
    bass:   List[str] = []
    in_bass = False
    for t in toks:
        if t == 'staff-bass':
            in_bass = True
            continue
        (bass if in_bass else treble).append(t)
        if t in _BARLINE_TOKEN_STRS:
            in_bass = False   # barline 이후 다음 마디는 treble
    return treble, bass


def _token_category(tok: str) -> str:
    if any(tok.startswith(p) for p in _NOTE_PREFIXES):
        return 'note/rest'
    if tok in _BARLINE_TOKEN_STRS:
        return 'barline'
    if tok == 'staff-bass':
        return 'staff-bass'
    if any(tok.startswith(p) for p in _HEADER_PREFIXES):
        return 'header'
    if tok.startswith('dynamic-') or tok.startswith('hairpin-'):
        return 'dynamic'
    if tok.startswith('artic-') or tok.startswith('ornament-') or tok == 'fermata':
        return 'artic'
    if any(k in tok for k in ('slur', 'tuplet', 'ottava')):
        return 'span'
    return 'other'


def _category_recall(pred_toks: List[str], gt_toks: List[str]) -> Dict[str, Tuple[int, int]]:
    """카테고리별 multiset recall: (맞춘 수, GT 총수)."""
    from collections import Counter
    gt_c   = Counter(gt_toks)
    pred_c = Counter(pred_toks)
    matched_c = gt_c & pred_c   # multiset intersection

    cats: Dict[str, Tuple[int, int]] = {}
    for tok, total in gt_c.items():
        cat = _token_category(tok)
        m, t = cats.get(cat, (0, 0))
        cats[cat] = (m + matched_c.get(tok, 0), t + total)
    return cats


def _note_error_rate(pred_toks: List[str], gt_toks: List[str]) -> float:
    """note/rest/chord 토큰만 추출해 순서 기반 Levenshtein 비율."""
    p = [t for t in pred_toks if any(t.startswith(x) for x in _NOTE_PREFIXES)]
    g = [t for t in gt_toks   if any(t.startswith(x) for x in _NOTE_PREFIXES)]
    if not g:
        return 0.0 if not p else 1.0
    return levenshtein(p, g) / len(g)


def _first_error(pred_toks: List[str], gt_toks: List[str]) -> Optional[Tuple[int, str, str]]:
    """첫 불일치 위치 → (index, gt_tok, pred_tok). 완전 일치 시 None."""
    for i, (g, p) in enumerate(zip(gt_toks, pred_toks)):
        if g != p:
            return (i, g, p)
    if len(pred_toks) < len(gt_toks):
        return (len(pred_toks), gt_toks[len(pred_toks)], '<누락>')
    if len(pred_toks) > len(gt_toks):
        return (len(gt_toks), '<없음>', pred_toks[len(gt_toks)])
    return None


def analyze_sample(img_path: str, gt_json_path: str,
                   seq2seq: OmrSeq2Seq,
                   tok2id: Dict[str, int], id2tok: Dict[int, str],
                   device: torch.device) -> Dict:
    """단일 샘플 상세 분석 딕셔너리 반환."""
    with open(gt_json_path, encoding='utf-8') as f:
        data = json.load(f)
    gt_toks  = [t for t in data.get('tokens', [])
                if t not in ('<SOS>', '<EOS>', '<PAD>')]
    pred_toks = run_image(img_path, seq2seq, tok2id, id2tok, device)

    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    gt_ids   = [tok2id.get(t, 3) for t in gt_toks]
    pred_ids = [tok2id.get(t, 3) for t in pred_toks]
    overall_ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)

    is_grand = 'staff-bass' in gt_toks

    result: Dict = {
        'file':        os.path.basename(img_path),
        'is_grand':    is_grand,
        'gt_len':      len(gt_toks),
        'pred_len':    len(pred_toks),
        'overall_ter': overall_ter,
        'gt_toks':     gt_toks,
        'pred_toks':   pred_toks,
        'overall_cat': _category_recall(pred_toks, gt_toks),
        'first_error': _first_error(pred_toks, gt_toks),
        'note_err':    _note_error_rate(pred_toks, gt_toks),
    }

    if is_grand:
        gt_tr,   gt_ba   = _extract_treble_bass(gt_toks)
        pred_tr, pred_ba = _extract_treble_bass(pred_toks)

        # bass는 양쪽 모두 barline 있음 → measure_segmented_ter 사용
        # treble는 GT에 barline 없음(interleaved 구조) → note 위주 비교
        ba_ter = measure_segmented_ter(
            [tok2id.get(t, 3) for t in pred_ba],
            [tok2id.get(t, 3) for t in gt_ba],
            barline_ids,
        )
        tr_note_err = _note_error_rate(pred_tr, gt_tr)
        ba_note_err = _note_error_rate(pred_ba, gt_ba)

        result.update({
            'gt_treble':      gt_tr,
            'gt_bass':        gt_ba,
            'pred_treble':    pred_tr,
            'pred_bass':      pred_ba,
            'treble_note_err':  tr_note_err,
            'bass_ter':         ba_ter,
            'bass_note_err':    ba_note_err,
            'treble_cat':     _category_recall(pred_tr, gt_tr),
            'bass_cat':       _category_recall(pred_ba, gt_ba),
            'treble_first':   _first_error(pred_tr, gt_tr),
            'bass_first':     _first_error(pred_ba, gt_ba),
        })

    return result


def _fmt_cat(stats: Dict[str, Tuple[int, int]]) -> str:
    parts = []
    order = ['header', 'note/rest', 'barline', 'staff-bass', 'dynamic', 'artic', 'span']
    for cat in order:
        if cat in stats:
            m, t = stats[cat]
            pct = m / t * 100 if t else 0
            parts.append(f"{cat}={m}/{t}({pct:.0f}%)")
    return '  '.join(parts)


def print_sample_analysis(r: Dict) -> None:
    """분석 결과를 사람이 읽기 좋게 출력."""
    SEP = '═' * 60
    print(f"\n{SEP}")
    grand_mark = '[대보표]' if r['is_grand'] else '[단일]'
    print(f"  {r['file']}  {grand_mark}  GT={r['gt_len']}tok  PRED={r['pred_len']}tok")
    print(SEP)

    print(f"  전체 TER : {r['overall_ter']*100:.1f}%   "
          f"음표 오류율 : {r['note_err']*100:.1f}%")
    err = r['first_error']
    if err:
        pos, g, p = err
        print(f"  첫 오류  : pos {pos}  GT={g}  PRED={p}")
    else:
        print("  첫 오류  : 없음 (완전 일치)")
    print(f"  카테고리 : {_fmt_cat(r['overall_cat'])}")

    if r['is_grand']:
        print()
        tr_ne = r['treble_note_err']
        print(f"  [TREBLE]  음표 오류율={tr_ne*100:.1f}%  음표 정확도={((1-tr_ne)*100):.1f}%")
        if r['treble_first']:
            pos, g, p = r['treble_first']
            print(f"            첫 오류 pos {pos}: GT={g}  PRED={p}")
        print(f"            {_fmt_cat(r['treble_cat'])}")

        ba_ne = r['bass_note_err']
        ba_ter = r['bass_ter']
        print(f"  [BASS]    음표 오류율={ba_ne*100:.1f}%  TER={ba_ter*100:.1f}%  "
              f"음표 정확도={((1-ba_ne)*100):.1f}%")
        if r['bass_first']:
            pos, g, p = r['bass_first']
            print(f"            첫 오류 pos {pos}: GT={g}  PRED={p}")
        print(f"            {_fmt_cat(r['bass_cat'])}")

        # 토큰 스트림 미리보기 (앞 12개)
        print()
        N = 12
        gt_tr_str   = ' '.join(r['gt_treble'][:N])   + ('...' if len(r['gt_treble'])   > N else '')
        pred_tr_str = ' '.join(r['pred_treble'][:N])  + ('...' if len(r['pred_treble']) > N else '')
        gt_ba_str   = ' '.join(r['gt_bass'][:N])      + ('...' if len(r['gt_bass'])     > N else '')
        pred_ba_str = ' '.join(r['pred_bass'][:N])    + ('...' if len(r['pred_bass'])   > N else '')
        print(f"  GT   treble: {gt_tr_str}")
        print(f"  PRED treble: {pred_tr_str}")
        print(f"  GT   bass  : {gt_ba_str}")
        print(f"  PRED bass  : {pred_ba_str}")
    else:
        N = 20
        gt_str   = ' '.join(r['gt_toks'][:N])   + ('...' if len(r['gt_toks'])   > N else '')
        pred_str = ' '.join(r['pred_toks'][:N])  + ('...' if len(r['pred_toks']) > N else '')
        print(f"\n  GT  : {gt_str}")
        print(f"  PRED: {pred_str}")


def run_analyze_dir(dir_path: str, seq2seq: OmrSeq2Seq,
                    tok2id: Dict[str, int], id2tok: Dict[int, str],
                    device: torch.device, n: int = 10) -> None:
    """디렉토리에서 최대 n개 샘플 상세 분석 + 집계 요약 출력."""
    pairs = []
    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        stem  = os.path.splitext(fname)[0]
        gt_p  = os.path.join(dir_path, stem + '.json')
        img_p = os.path.join(dir_path, fname)
        if os.path.isfile(gt_p):
            pairs.append((img_p, gt_p))
    pairs = pairs[:n]

    results = []
    for img_p, gt_p in pairs:
        try:
            r = analyze_sample(img_p, gt_p, seq2seq, tok2id, id2tok, device)
            print_sample_analysis(r)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {img_p}: {e}")

    if not results:
        return

    # 집계 요약
    print(f"\n{'═'*60}")
    print(f"  분석 요약  ({len(results)}개 샘플)")
    print('═' * 60)

    avg_overall = sum(r['overall_ter'] for r in results) / len(results)
    avg_note    = sum(r['note_err']    for r in results) / len(results)
    print(f"  전체 TER 평균    : {avg_overall*100:.1f}%  "
          f"(Acc {(1-avg_overall)*100:.1f}%)")
    print(f"  음표 오류율 평균 : {avg_note*100:.1f}%  "
          f"(음표 Acc {(1-avg_note)*100:.1f}%)")

    grand = [r for r in results if r['is_grand']]
    if grand:
        avg_tr = sum(r['treble_note_err'] for r in grand) / len(grand)
        avg_ba = sum(r['bass_note_err']   for r in grand) / len(grand)
        avg_ba_ter = sum(r['bass_ter']    for r in grand) / len(grand)
        print(f"\n  대보표 ({len(grand)}개):")
        print(f"    Treble 음표 오류율 : {avg_tr*100:.1f}%  "
              f"(Acc {(1-avg_tr)*100:.1f}%)")
        print(f"    Bass   음표 오류율 : {avg_ba*100:.1f}%  "
              f"(Acc {(1-avg_ba)*100:.1f}%)")
        print(f"    Bass   TER         : {avg_ba_ter*100:.1f}%")

        # 카테고리별 집계
        from collections import defaultdict
        cat_m: Dict[str, int] = defaultdict(int)
        cat_t: Dict[str, int] = defaultdict(int)
        for r in grand:
            for cat, (m, t) in r['overall_cat'].items():
                cat_m[cat] += m
                cat_t[cat] += t
        print(f"\n  전체 카테고리 recall:")
        for cat in ['header', 'note/rest', 'barline', 'staff-bass', 'dynamic', 'artic']:
            if cat_t[cat]:
                pct = cat_m[cat] / cat_t[cat] * 100
                print(f"    {cat:<12}: {cat_m[cat]}/{cat_t[cat]} ({pct:.1f}%)")


def fix_chord_tokens(token_ids: List[int], id2tok: dict) -> List[int]:
    """고아 chord- 토큰 제거: note-/dur-/chord- 바로 뒤에만 허용.
    (note-{pitch} 뒤 dur-{dur}이 오고 그 다음에 chord-가 이어지는 구조)"""
    result = []
    for tid in token_ids:
        tok = id2tok.get(tid, '')
        if tok.startswith('chord-'):
            prev = id2tok.get(result[-1], '') if result else ''
            if prev.startswith('note-') or prev.startswith('dur-') or prev.startswith('chord-'):
                result.append(tid)
        else:
            result.append(tid)
    return result


def fix_span_tokens(token_ids: List[int], id2tok: dict) -> List[int]:
    """짝 없는 span start/end 토큰 제거 (stack 기반)."""
    remove = set()
    stacks: dict = {s: [] for s in _SPAN_PAIRS}
    for idx, tid in enumerate(token_ids):
        tok = id2tok.get(tid, '')
        if tok in _SPAN_PAIRS:
            stacks[tok].append(idx)
        elif tok in _SPAN_ENDS:
            start = next(s for s, e in _SPAN_PAIRS.items() if e == tok)
            if stacks[start]:
                stacks[start].pop()
            else:
                remove.add(idx)
    for indices in stacks.values():
        remove.update(indices)
    return [tid for i, tid in enumerate(token_ids) if i not in remove]


def _split_measures(token_ids: List[int], barline_ids: set) -> List[List[int]]:
    """barline ID로 마디 분리 (barline 포함)."""
    measures, cur = [], []
    for tid in token_ids:
        cur.append(tid)
        if tid in barline_ids:
            measures.append(cur); cur = []
    if cur:
        measures.append(cur)
    return measures


def measure_segmented_ter(pred: List[int], gt: List[int],
                           barline_ids: set) -> float:
    """마디 단위 TER (OMR-NED 방식). 마디 간 오류 전파 차단."""
    if not gt:
        return 0.0 if not pred else 1.0
    pred_m = _split_measures(pred, barline_ids)
    gt_m   = _split_measures(gt,   barline_ids)
    total_err = total_len = 0
    for i in range(max(len(pred_m), len(gt_m))):
        p = pred_m[i] if i < len(pred_m) else []
        g = gt_m[i]   if i < len(gt_m)   else []
        total_err += levenshtein(p, g)
        total_len += len(g)
    return total_err / max(total_len, 1)


def eval_dir(eval_dir_path: str, seq2seq: OmrSeq2Seq,
             tok2id: Dict[str, int], id2tok: Dict[int, str],
             device: torch.device, n_eval: int = 200):
    pairs = []
    for fname in sorted(os.listdir(eval_dir_path)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        stem  = os.path.splitext(fname)[0]
        gt_p  = os.path.join(eval_dir_path, stem + '.json')
        img_p = os.path.join(eval_dir_path, fname)
        if os.path.isfile(gt_p):
            pairs.append((img_p, gt_p))
    pairs = pairs[:n_eval]
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    print(f"평가 샘플: {len(pairs)}개")

    ter_sum = note_err_sum = n_pass = 0
    grand_total = grand_pass = 0
    tr_note_sum = ba_note_sum = ba_ter_sum = 0
    from collections import defaultdict
    cat_m: Dict[str, int] = defaultdict(int)
    cat_t: Dict[str, int] = defaultdict(int)
    t0 = time.time()

    for i, (img_p, gt_p) in enumerate(pairs):
        with open(gt_p, encoding='utf-8') as f:
            gt_toks = [t for t in json.load(f).get('tokens', [])
                       if t not in ('<SOS>', '<EOS>', '<PAD>')]
        is_grand = 'staff-bass' in gt_toks
        try:
            pred_toks = run_image(img_p, seq2seq, tok2id, id2tok, device)
        except Exception as e:
            print(f"  [ERROR] {img_p}: {e}")
            continue

        gt_ids   = [tok2id.get(t, 3) for t in gt_toks]
        pred_ids = [tok2id.get(t, 3) for t in pred_toks]
        ter      = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        ter_sum += ter

        note_err_sum += _note_error_rate(pred_toks, gt_toks)

        for cat, (m, t) in _category_recall(pred_toks, gt_toks).items():
            cat_m[cat] += m
            cat_t[cat] += t

        if ter == 0.0:
            n_pass += 1
        if is_grand:
            grand_total += 1
            if ter == 0.0:
                grand_pass += 1
            gt_tr, gt_ba     = _extract_treble_bass(gt_toks)
            pred_tr, pred_ba = _extract_treble_bass(pred_toks)
            tr_note_sum += _note_error_rate(pred_tr, gt_tr)
            ba_note_sum += _note_error_rate(pred_ba, gt_ba)
            ba_ter_sum  += measure_segmented_ter(
                [tok2id.get(t, 3) for t in pred_ba],
                [tok2id.get(t, 3) for t in gt_ba],
                barline_ids,
            )

        if (i + 1) % 50 == 0:
            avg_ter = ter_sum / (i + 1)
            print(f"  [{i+1}/{len(pairs)}]  TER={avg_ter*100:.1f}%  "
                  f"Acc={((1-avg_ter)*100):.1f}%")

    n         = len(pairs)
    avg_ter   = ter_sum      / max(n, 1)
    avg_note  = note_err_sum / max(n, 1)
    pass_rate = n_pass / max(n, 1) * 100.0
    elapsed   = time.time() - t0

    print(f"\n{'─'*50}")
    print(f"샘플            : {n}")
    print(f"전체 TER        : {avg_ter*100:.1f}%   Acc={((1-avg_ter)*100):.1f}%")
    print(f"음표 오류율     : {avg_note*100:.1f}%   음표 Acc={((1-avg_note)*100):.1f}%")
    print(f"Pass (TER=0)    : {pass_rate:.1f}%  ({n_pass}/{n})")

    if grand_total:
        avg_tr     = tr_note_sum / grand_total
        avg_ba     = ba_note_sum / grand_total
        avg_ba_ter = ba_ter_sum  / grand_total
        gpass      = grand_pass  / grand_total * 100
        print(f"\n  대보표 ({grand_total}개):")
        print(f"    Treble 음표 오류율 : {avg_tr*100:.1f}%  Acc={((1-avg_tr)*100):.1f}%")
        print(f"    Bass   음표 오류율 : {avg_ba*100:.1f}%  Acc={((1-avg_ba)*100):.1f}%")
        print(f"    Bass   TER         : {avg_ba_ter*100:.1f}%")
        print(f"    Grand Pass (TER=0) : {gpass:.1f}%  ({grand_pass}/{grand_total})")

    print(f"\n  카테고리 recall:")
    for cat in ['header', 'note/rest', 'barline', 'staff-bass', 'dynamic', 'artic']:
        if cat_t[cat]:
            pct = cat_m[cat] / cat_t[cat] * 100
            print(f"    {cat:<12}: {cat_m[cat]}/{cat_t[cat]} ({pct:.1f}%)")

    print(f"\n  소요 시간 : {elapsed:.1f}s ({elapsed/max(n,1):.2f}s/샘플)")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Round 3 추론/평가 (Grand Staff)')
    p.add_argument('image',         nargs='?', help='단일 이미지')
    p.add_argument('--seq2seq',     required=True, help='seq2seq_best.pt')
    p.add_argument('--tokenizer',   default=str(_HERE / 'tokenizer258.json'))
    p.add_argument('--eval_dir',    default=None,  help='배치 평가 디렉토리')
    p.add_argument('--n_eval',      type=int, default=200)
    p.add_argument('--analyze',     default=None,
                   help='상세 분석 디렉토리 (treble/bass 분리 포함)')
    p.add_argument('--n_analyze',   type=int, default=10,
                   help='분석할 샘플 수 (기본 10)')
    p.add_argument('--device',      default='auto')
    args = p.parse_args()

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f"Device: {device}")

    tok2id, id2tok = load_tokenizer(args.tokenizer)
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt    = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}")

    if args.analyze:
        run_analyze_dir(args.analyze, seq2seq, tok2id, id2tok,
                        device, n=args.n_analyze)
    elif args.eval_dir:
        eval_dir(args.eval_dir, seq2seq, tok2id, id2tok, device, args.n_eval)
    elif args.image:
        tokens = run_image(args.image, seq2seq, tok2id, id2tok, device)
        print("추론 결과:")
        print(' '.join(tokens))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
