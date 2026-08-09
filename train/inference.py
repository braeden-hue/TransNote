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
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from music21.pitch import Pitch

from model import OmrSeq2Seq, SOS_ID, EOS_ID, PAD_ID, MAX_SEQ
from dataset import (load_preprocessed, extract_staff_canvas,
                     extract_system_canvas, best_effort_staff_detection,
                     IMG_MEAN, IMG_STD, load_tokenizer, make_model_input)


def _encoder_in_ch(seq2seq: OmrSeq2Seq) -> int:
    """로드된 체크포인트의 실제 첫 conv 입력 채널 수를 가중치 shape에서 그대로 읽는다
    (CoordConv 실험용 in_ch=2 체크포인트와 기존 in_ch=1 체크포인트를 호출부 수정 없이
    자동으로 구분하기 위함, 2026-07-31)."""
    return seq2seq.encoder.backbone[0].block[0].weight.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
#  Decode constants
# ─────────────────────────────────────────────────────────────────────────────

INFER_MAX_LEN = 300   # 시스템 전체 인터리빙 시퀀스 기준 (treble+staff-bass+bass)
EOS_BOOST     = 1.5   # EOS logit 증폭 배율

# 2026-08-03: 실사 held-out 오류 분석에서, 오선 검출은 정상인데(1시스템만 선택) 디코딩이
# 정상 종료 지점(barline-final)을 지나쳐서도 계속 새 내용을 만들어내는 "과잉생성" 사례를
# 확인 (예: GT 77토큰인데 174토큰 생성, 초반은 GT와 거의 일치하다가 정상 종료 지점 근처부터
# 이탈). 이 프로젝트 발췌(4마디) 기준 관찰된 GT 길이는 대략 77~129토큰 — 이를 한참 넘어서도
# (LONG_DECODE_THRESHOLD 이후) EOS/barline-final 쪽 logit을 스텝이 늘수록 점점 더 강하게
# 밀어준다. 정상 길이 디코딩(절대다수)은 threshold 밑이라 전혀 영향받지 않고, 이미 비정상적으로
# 길어진 시퀀스만 더 빨리 종료되게 만드는 방식이라 회귀 위험이 낮다.
LONG_DECODE_THRESHOLD = 150   # 이 스텝을 넘으면 램프 시작
LONG_DECODE_RAMP      = 0.05  # 스텝당 추가 배율 증가폭


# ─────────────────────────────────────────────────────────────────────────────
#  마르코프 체인(PDMX 실제곡 음정 전이 통계) 디코딩 단계 융합 (shallow fusion)
# ─────────────────────────────────────────────────────────────────────────────
# 2026-07-30 사용자 요청: build_markov_transitions.py로 학습 데이터 생성에 쓰던 것과 같은
# 전이확률표를, 이미 학습된 체크포인트의 "디코딩" 단계에도 적용 -- 재학습 없이 음성인식/OCR의
# 언어모델 shallow fusion과 동일한 방식으로, 시각 정보가 애매할 때(노이즈) 모델이 "그럴듯한
# 다음 음" 쪽으로 살짝 기울도록 로짓에 편향을 더한다. note- 토큰에만 적용하고 나머지(dur-,
# rest-, clef- 등)는 그대로 둔다 -- 모델이 지금 음표를 낼 차례인지 아닌지는 그대로 모델
# 판단에 맡기고, "어떤 음표"만 살짝 조정.
#
# 치/베이스는 완전히 분리 추적해야 함 -- generate_scores.py의 prev_pitch도 치/베이스를 따로
# 추적하므로(대보표 인터리빙 시퀀스에서 바로 앞 토큰이 다른 성부일 수 있음, 예: 베이스 첫
# 음을 예측할 때 "바로 앞"은 치의 마지막 음이지 베이스의 이전 음이 아님) 여기서도 동일하게
# staff-bass/barline* 토큰으로 현재 성부를 추적하고, 성부별 마지막 note- 피치만 prev로 쓴다.

_BARLINE_TOKEN_STRS_LOCAL = frozenset({
    'barline', 'barline-final', 'barline-start-repeat', 'barline-end-repeat',
})


def load_markov_table(path: str) -> Tuple[Dict[int, float], int]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    table = {int(k): v for k, v in data['table'].items()}
    return table, data.get('max_interval', 14)


def build_note_token_info(tok2id: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """어휘 안의 note-* 토큰 id와 그 다이어토닉 스텝(Pitch.diatonicNoteNum)을 미리 계산해둔다
    -- 디코딩 스텝마다 새로 계산하지 않도록(속도)."""
    ids, steps = [], []
    for tok, tid in tok2id.items():
        if tok.startswith('note-'):
            ids.append(tid)
            steps.append(Pitch(tok[len('note-'):]).diatonicNoteNum)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(steps, dtype=torch.long)


def markov_log_prob_table(table: Dict[int, float], max_interval: int) -> torch.Tensor:
    """interval(-max_interval..+max_interval) -> log(prob) 1차원 텐서(인덱스 = interval+max_interval).
    테이블에 없는 간격(사실상 없어야 정상)은 테이블 최솟값을 바닥으로 씀."""
    floor = min(table.values())
    size = 2 * max_interval + 1
    out = torch.empty(size, dtype=torch.float32)
    for i in range(size):
        interval = i - max_interval
        out[i] = float(np.log(table.get(interval, floor)))
    return out


def _update_voice_state(voice: str, prev_treble, prev_bass, tok_str: str):
    """토큰 문자열 하나를 반영해 (현재 성부, 치 마지막 피치, 베이스 마지막 피치) 갱신."""
    if tok_str == 'staff-bass':
        voice = 'bass'
    elif tok_str in _BARLINE_TOKEN_STRS_LOCAL:
        voice = 'treble'
    elif tok_str.startswith('note-'):
        pitch_str = tok_str[len('note-'):]
        if voice == 'treble':
            prev_treble = pitch_str
        else:
            prev_bass = pitch_str
    elif tok_str.startswith('rest-'):
        if voice == 'treble':
            prev_treble = None
        else:
            prev_bass = None
    return voice, prev_treble, prev_bass


def _markov_bias(vocab_size: int, prev_pitch: Optional[str],
                  note_ids: torch.Tensor, note_steps: torch.Tensor,
                  log_prob_table: torch.Tensor, max_interval: int,
                  weight: float, device: torch.device) -> Optional[torch.Tensor]:
    """prev_pitch가 있으면 note- 토큰들에 대해서만 weight * log(P_markov(interval))을 더할
    편향 벡터[vocab_size]를 만든다(그 외 위치는 0). prev_pitch가 없으면(마디/성부 시작 등) None."""
    if prev_pitch is None or weight <= 0:
        return None
    prev_step = Pitch(prev_pitch).diatonicNoteNum
    intervals = torch.clamp(note_steps - prev_step, -max_interval, max_interval) + max_interval
    log_probs = log_prob_table[intervals]
    bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    bias[note_ids.to(device)] = (weight * log_probs).to(device)
    return bias


class MarkovDecodeConfig:
    """greedy_decode/beam_decode에 넘기는 마르코프 융합 설정 묶음. id2tok은 새로 생성된
    토큰 id를 문자열로 바꿔 성부(치/베이스)·직전 피치 상태를 갱신하는 데만 쓴다."""
    def __init__(self, id2tok: Dict[int, str], tok2id: Dict[str, int],
                 markov_table: Dict[int, float], max_interval: int, weight: float):
        self.id2tok = id2tok
        self.max_interval = max_interval
        self.weight = weight
        self.note_ids, self.note_steps = build_note_token_info(tok2id)
        self.log_prob_table = markov_log_prob_table(markov_table, max_interval)


class InlineTimeCorrector:
    """디코딩 도중 마디(barline)가 하나 완성될 때마다 그 마디의 음표/쉼표 길이 합으로
    박자표를 즉시 재추정해 time-* 토큰을 그 자리에서 교체한다.

    2026-08-03: correct_time_signature()(사후처리, 디코딩이 전부 끝난 뒤 최종 토큰
    리스트에서 time-*만 바꿔치기)는 held-out 60곡 재측정에서 개별 time-* 정답률은
    올렸지만(모델 자체 오답의 56% 구제) 전체 정확도는 거의 안 움직였다(61.7%->61.8%) --
    이미 (틀린 박자표를 전제로) 생성돼버린 그 뒤의 음표/길이들은 사후 교체로는 전혀
    소급 수정되지 않기 때문. decode_step_cached는 self-attention을 매 스텝 past 전체에
    대해 새로 계산하므로(Decoder.forward_cached 주석 참고), 이미 생성된 past 안의 토큰을
    나중에 고쳐도 바로 다음 스텝부터 그 수정이 그대로 반영된다 -- 그래서 사후처리 대신
    "마디 하나 끝날 때마다 즉시 재추정 + past 패치"로 옮기면, 최소한 두 번째 마디부터는
    올바른(또는 더 정확한) 박자표를 조건으로 생성되게 만들 수 있다.

    대보표(is_grand=True)에서는 barline이 항상 bass 세그먼트 처리 중에 나온다(interleaving
    포맷 특성 -- correct_time_signature의 treble/bass 버그와 동일한 이유로 bass 쪽 버퍼를
    써야 함), 단일 오선(is_grand=False)에서는 그런 구분이 없다.
    """

    def __init__(self, tok2id: Dict[str, int], id2tok: Dict[int, str], is_grand: bool):
        self.tok2id = tok2id
        self.id2tok = id2tok
        self.is_grand = is_grand
        self.time_idx: Optional[int] = None   # result 리스트 안에서 time-* 토큰의 위치
        self.voice = 'treble'
        self.cur_treble: List[str] = []
        self.cur_bass: List[str] = []
        self.completed_sums: List[Fraction] = []

    def observe(self, tok_str: str, result_idx: int) -> Optional[int]:
        """새로 확정된 토큰 하나(문자열, result 안 위치)를 반영. barline으로 마디가
        방금 끝났고 재추정 결과가 현재 time-* 토큰과 다르면 새 토큰 id를 반환한다
        (교체 불필요/불가능하면 None)."""
        if self.time_idx is None and tok_str.startswith('time-'):
            self.time_idx = result_idx
            return None
        if tok_str == 'staff-bass':
            self.voice = 'bass'
            return None
        if tok_str in _BARLINE_TOKEN_STRS:
            from collections import Counter
            measure = self.cur_bass if self.is_grand else self.cur_treble
            self.cur_treble, self.cur_bass, self.voice = [], [], 'treble'
            if self.time_idx is None:
                return None
            self.completed_sums.append(_measure_beat_sum(measure))
            majority_sum, _cnt = Counter(self.completed_sums).most_common(1)[0]
            candidates = _BEATS_TO_SIGS.get(majority_sum)
            if not candidates:
                return None
            if len(candidates) == 1:
                corrected = candidates[0]
            else:
                n_eighth  = sum(1 for t in measure if t == 'dur-1/8')
                n_quarter = sum(1 for t in measure if t == 'dur-1/4')
                corrected = 'time-6/8' if n_eighth > n_quarter else 'time-3/4'
            return self.tok2id.get(corrected)
        (self.cur_bass if self.voice == 'bass' else self.cur_treble).append(tok_str)
        return None


@torch.no_grad()
def greedy_decode(seq2seq: OmrSeq2Seq, canvas: np.ndarray,
                  device: torch.device,
                  max_len: int = INFER_MAX_LEN,
                  markov: Optional[MarkovDecodeConfig] = None,
                  stop_token_id: Optional[int] = None,
                  time_correct: Optional['InlineTimeCorrector'] = None) -> List[int]:
    tile_f = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    inp    = make_model_input(tile_f, _encoder_in_ch(seq2seq)).unsqueeze(0).to(device)
    seq2seq.eval()
    memory   = seq2seq.encode(inp)
    kv_cache = seq2seq.precompute_memory_kv(memory)  # cross-attention K,V 1회 계산 (Step1 가속)
    past   = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    result = []
    voice, prev_treble, prev_bass = 'treble', None, None
    for step in range(max_len):
        logits = seq2seq.decode_step_cached(kv_cache, past)
        logits[0, EOS_ID] *= EOS_BOOST
        if step > LONG_DECODE_THRESHOLD:
            long_boost = 1.0 + (step - LONG_DECODE_THRESHOLD) * LONG_DECODE_RAMP
            logits[0, EOS_ID] *= long_boost
            if stop_token_id is not None:
                logits[0, stop_token_id] *= long_boost
        if markov is not None:
            prev_pitch = prev_bass if voice == 'bass' else prev_treble
            bias = _markov_bias(logits.shape[-1], prev_pitch, markov.note_ids, markov.note_steps,
                                 markov.log_prob_table, markov.max_interval, markov.weight, device)
            if bias is not None:
                logits = logits + bias.unsqueeze(0)
        nxt = int(logits.argmax(-1).item())
        if nxt == EOS_ID: break
        if nxt != PAD_ID:
            result.append(nxt)
            if markov is not None:
                voice, prev_treble, prev_bass = _update_voice_state(
                    voice, prev_treble, prev_bass, markov.id2tok.get(nxt, ''))
            if time_correct is not None:
                tok_str = time_correct.id2tok.get(nxt, '')
                new_time_id = time_correct.observe(tok_str, len(result) - 1)
                if new_time_id is not None and result[time_correct.time_idx] != new_time_id:
                    result[time_correct.time_idx] = new_time_id
                    past[0, time_correct.time_idx + 1] = new_time_id
            if stop_token_id is not None and nxt == stop_token_id:
                # barline-final은 토큰 문법상 항상 시퀀스의 마지막 토큰이어야 함
                # (generate_scores.py/mscz_to_tokens.py 등 전부 이 토큰으로 트리밍/종료).
                # 화음/임시표가 늘면서 EOS 확신이 떨어져 barline-final 이후에도 새 헤더를
                # 반복 생성하는 붕괴가 관찰됨(2026-08-02, Round9 화음 검증) -- 재학습 없이
                # 디코딩 단계에서 강제 종료해 반복 루프를 차단.
                break
        past = torch.cat([past, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return result


class _Beam:
    __slots__ = ('seq', 'tokens', 'score', 'finished', 'voice', 'prev_treble', 'prev_bass')

    def __init__(self, seq, tokens, score, finished,
                 voice='treble', prev_treble=None, prev_bass=None):
        self.seq = seq            # [1, t] decoder input so far (incl. SOS)
        self.tokens = tokens       # generated token ids, excl. SOS/EOS/PAD
        self.score = score         # cumulative log-probability
        self.finished = finished   # hit EOS already
        self.voice = voice              # 마르코프 융합용(markov=None이면 안 씀): 'treble'/'bass'
        self.prev_treble = prev_treble  # 현재 성부까지 치의 마지막 note- 피치 문자열
        self.prev_bass = prev_bass      # 현재 성부까지 베이스의 마지막 note- 피치 문자열


@torch.no_grad()
def beam_decode(seq2seq: OmrSeq2Seq, canvas: np.ndarray,
                device: torch.device,
                beam_width: int = 4,
                length_penalty: float = 0.7,
                max_len: int = INFER_MAX_LEN,
                markov: Optional[MarkovDecodeConfig] = None,
                stop_token_id: Optional[int] = None,
                time_correct: Optional['InlineTimeCorrector'] = None) -> List[int]:
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

    markov가 주어지면 빔마다 자기 성부(치/베이스) 상태를 따로 들고 다니며(_Beam.voice/
    prev_treble/prev_bass) 토큰 분포에 PDMX 음정 전이 편향을 더한다 -- greedy_decode와
    동일한 shallow fusion, 빔 하나하나가 독립된 시퀀스라 상태도 빔마다 독립.
    """
    if beam_width <= 1:
        return greedy_decode(seq2seq, canvas, device, max_len, markov=markov,
                              stop_token_id=stop_token_id, time_correct=time_correct)

    tile_f = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    inp    = make_model_input(tile_f, _encoder_in_ch(seq2seq)).unsqueeze(0).to(device)
    seq2seq.eval()
    memory   = seq2seq.encode(inp)
    kv_cache = seq2seq.precompute_memory_kv(memory)  # cross-attention K,V 1회 계산 (Step1 가속)

    sos = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    beams = [_Beam(sos, [], 0.0, False)]

    for step in range(max_len):
        if all(b.finished for b in beams):
            break

        candidates = []  # (score, parent_idx, token_id_or_None, advances)
        for i, b in enumerate(beams):
            if b.finished:
                candidates.append((b.score, i, None, False))
                continue
            logits = seq2seq.decode_step_cached(kv_cache, b.seq)  # [1, vocab]
            logits[0, EOS_ID] *= EOS_BOOST
            if step > LONG_DECODE_THRESHOLD:
                long_boost = 1.0 + (step - LONG_DECODE_THRESHOLD) * LONG_DECODE_RAMP
                logits[0, EOS_ID] *= long_boost
                if stop_token_id is not None:
                    logits[0, stop_token_id] *= long_boost
            if markov is not None:
                prev_pitch = b.prev_bass if b.voice == 'bass' else b.prev_treble
                bias = _markov_bias(logits.shape[-1], prev_pitch, markov.note_ids, markov.note_steps,
                                     markov.log_prob_table, markov.max_interval, markov.weight, device)
                if bias is not None:
                    logits = logits + bias.unsqueeze(0)
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
                new_beams.append(_Beam(parent_beam.seq, parent_beam.tokens, score, True,
                                        parent_beam.voice, parent_beam.prev_treble, parent_beam.prev_bass))
            else:
                new_seq = torch.cat(
                    [parent_beam.seq, torch.tensor([[tok]], dtype=torch.long, device=device)],
                    dim=1)
                new_tokens = parent_beam.tokens + ([tok] if tok != PAD_ID else [])
                voice, prev_treble, prev_bass = parent_beam.voice, parent_beam.prev_treble, parent_beam.prev_bass
                if markov is not None and tok != PAD_ID:
                    voice, prev_treble, prev_bass = _update_voice_state(
                        voice, prev_treble, prev_bass, markov.id2tok.get(tok, ''))
                # barline-final 이후 반복 생성 차단(greedy_decode와 동일 이유, 2026-08-02).
                is_stop = stop_token_id is not None and tok == stop_token_id
                new_beams.append(_Beam(new_seq, new_tokens, score, is_stop, voice, prev_treble, prev_bass))
        beams = new_beams

    best = max(beams, key=lambda b: b.score / max(len(b.tokens), 1) ** length_penalty)
    return best.tokens


# ─────────────────────────────────────────────────────────────────────────────
#  단일 이미지 추론
# ─────────────────────────────────────────────────────────────────────────────

def run_image(image_path: str, seq2seq: OmrSeq2Seq,
              tok2id: Dict[str, int], id2tok: Dict[int, str],
              device: torch.device, beam_width: int = 1,
              markov: Optional[MarkovDecodeConfig] = None) -> List[str]:
    """
    이미지 → 토큰 문자열 리스트.

    오선 감지 수 N에 따라:
    - N % 2 == 0 (≥ 2) : 대보표 → 시스템별 system canvas(SYSTEM_CANVAS_H×CANVAS_W, treble+bass 결합)로
                          디코딩 1회 → 전체 인터리빙 시퀀스 (staff-bass 자동 포함)
    - 그 외             : 단일 오선 순차 처리

    beam_width=1(기본)이면 greedy와 동일. 1보다 크면 beam_decode 사용 --
    exposure bias로 그리디가 한 스텝 실수하면 이후 전체가 어긋나는 문제를
    완화(2026-07-21 오류 분석에서 확인된 패턴에 대응).
    """
    SKIP_TOKS = {'<PAD>', '<SOS>', '<EOS>'}
    stop_id = tok2id.get('barline-final')

    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    gray0  = load_preprocessed(image_path)
    # 실사 사진(jpg/jpeg)만 correct_perspective(페이지 검출+워프)를 쓰고, 합성 렌더링
    # (png)은 _deskew(회전 보정만)로 제한한다 -- 페이지 경계가 없는 깨끗한 합성 이미지에
    # 전체 워프를 걸면 콘텐츠가 압축/왜곡되어 오선을 과다 검출하는 버그 확인
    # (2026-08-02, Round9 화음 검증). 실사는 반대로 전체 워프가 실측상 더 나음.
    is_real_photo = image_path.lower().endswith(('.jpg', '.jpeg'))
    staffs, gray = best_effort_staff_detection(gray0, use_full_warp=is_real_photo)
    if not staffs:
        return []

    n = len(staffs)
    all_tokens: List[str] = []

    if n >= 2 and n % 2 == 0:
        # 대보표: 시스템 캔버스(treble+bass 수직 결합, SYSTEM_CANVAS_H×CANVAS_W)로 시퀀스 한 번에 예측.
        # 모델은 이 방식으로 학습됐음 — 개별 오선을 따로 디코딩하지 않는다.
        # dataset.py의 _split_grand_staff_interleaved()가 시스템마다 독립 샘플로 쪼개면서
        # 각 시스템 타깃 앞에 헤더(clef-G/key/time)를 그대로 복제해 붙이므로(학습 설계상
        # 의도된 동작), 모델도 매 시스템을 새 시퀀스처럼 헤더부터 예측한다. 여러 시스템을
        # 한 페이지로 이어붙일 때는 첫 시스템의 헤더만 남기고 나머지는 제거해야
        # 원본(단일 연속 시퀀스) 형식과 일치한다.
        n_systems = n // 2
        for sys_i in range(n_systems):
            treble_staff  = staffs[sys_i * 2]
            bass_staff    = staffs[sys_i * 2 + 1]
            sys_canvas    = extract_system_canvas(gray, [treble_staff, bass_staff])
            sys_ids       = beam_decode(seq2seq, sys_canvas, device, beam_width=beam_width, markov=markov,
                                         stop_token_id=stop_id,
                                         time_correct=InlineTimeCorrector(tok2id, id2tok, is_grand=True))
            sys_ids       = fix_span_tokens(fix_chord_tokens(sys_ids, id2tok), id2tok)
            sys_ids       = correct_time_signature(sys_ids, id2tok, tok2id, is_grand=True)
            sys_ids       = correct_accidentals_by_key(sys_ids, id2tok, tok2id)
            if sys_i > 0:
                sys_ids = _strip_leading_header(sys_ids, id2tok)
            for tok_id in sys_ids:
                tok_str = id2tok.get(tok_id, '<UNK>')
                if tok_str not in SKIP_TOKS:
                    all_tokens.append(tok_str)

    else:
        # 단일 오선 또는 홀수 오선: 순서대로 이어붙임
        for staff in staffs:
            canvas = extract_staff_canvas(gray, staff)
            ids    = beam_decode(seq2seq, canvas, device, beam_width=beam_width, markov=markov,
                                  stop_token_id=stop_id,
                                  time_correct=InlineTimeCorrector(tok2id, id2tok, is_grand=False))
            ids    = fix_span_tokens(fix_chord_tokens(ids, id2tok), id2tok)
            ids    = correct_time_signature(ids, id2tok, tok2id, is_grand=False)
            ids    = correct_accidentals_by_key(ids, id2tok, tok2id)
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


# ─────────────────────────────────────────────────────────────────────────────
#  박자표 사후 보정 (time-* 토큰은 음표들보다 먼저 디코딩되므로, 디코딩 도중에는
#  "음표를 본 뒤 박자를 정한다"가 불가능함 — 대신 디코딩이 끝난 뒤 실제 음표 길이
#  합으로 검증/보정한다. 사진에 박자표 글자가 안 보이는 마디(악보 중간 촬영)에서
#  모델이 시각 단서 없이 찍은 time-* 값을 바로잡는 용도.
# ─────────────────────────────────────────────────────────────────────────────

_TUPLET_SCALE = {'tuplet-3-start': Fraction(2, 3)}   # 셋잇단음표 3개 = 정상 2개 분량
_TUPLET_END   = frozenset({'tuplet-3-end'})

# 시간표기 → 마디 전체 길이(온음표 대비 분수). 3/4와 6/8은 값이 같아 길이 합만으로는
# 구분 불가(beam 묶음 정보가 토크나이저에 없음) — 이 경우 모델 예측을 그대로 신뢰한다.
_TIME_SIG_BEATS: Dict[str, Fraction] = {
    'time-4/4': Fraction(1, 1),
    'time-3/4': Fraction(3, 4),
    'time-2/4': Fraction(1, 2),
    'time-6/8': Fraction(3, 4),
}
_BEATS_TO_SIGS: Dict[Fraction, List[str]] = {}
for _sig, _beats in _TIME_SIG_BEATS.items():
    _BEATS_TO_SIGS.setdefault(_beats, []).append(_sig)


def _tok_duration(tok: str) -> Optional[Fraction]:
    """'dur-a/b' 또는 'rest-a/b' → 온음표 대비 길이. chord-*는 자기 길이가 없으므로 None."""
    for prefix in ('dur-', 'rest-'):
        if tok.startswith(prefix):
            try:
                num, den = tok[len(prefix):].split('/')
                return Fraction(int(num), int(den))
            except (ValueError, ZeroDivisionError):
                return None
    return None


def _measure_beat_sum(measure_toks: List[str]) -> Fraction:
    """한 마디 안 음표/쉼표 길이 합 (셋잇단음표 구간은 2/3배로 스케일)."""
    total = Fraction(0)
    scale = Fraction(1)
    for tok in measure_toks:
        if tok in _TUPLET_SCALE:
            scale = _TUPLET_SCALE[tok]
        elif tok in _TUPLET_END:
            scale = Fraction(1)
        else:
            dur = _tok_duration(tok)
            if dur is not None:
                total += dur * scale
    return total


def correct_time_signature(token_ids: List[int], id2tok: dict, tok2id: dict,
                           is_grand: bool) -> List[int]:
    """디코딩된 time-* 토큰을 실제 마디 길이 합으로 검증해 필요 시 교체.

    여러 마디가 있으면 다수결로 판단(디코딩 오류 한 마디에 흔들리지 않도록).
    길이 합이 4/4·2/4처럼 유일하게 정해지는 값이면 그 값으로 교체하고, 3/4·6/8처럼
    길이 합만으로 안 갈리는 경우엔 8분음표 개수가 4분음표 개수보다 많으면 6/8(복합박자
    특유의 잦은 8분 세분), 아니면 3/4로 판단(2026-07-31 사용자 제안 휴리스틱 -- 정밀한
    리듬 지문 분석은 아니고 근사치). 둘 다 아예 알 수 없는 합이면 모델 예측을 그대로 둔다.
    사진에 박자표 글자가 안 보이는 마디(악보 중간 촬영)에서 모델이 시각 단서 없이 찍은
    time-* 값을 바로잡는 용도.
    """
    toks = [id2tok.get(tid, '<UNK>') for tid in token_ids]
    time_idx = next((i for i, t in enumerate(toks) if t.startswith('time-')), None)
    if time_idx is None:
        return token_ids

    # 2026-08-03: is_grand일 때 여기서 index [0](treble)을 쓰면 안 됨 -- _extract_treble_bass의
    # 인터리빙 분리 로직상 barline은 항상 bass 세그먼트 처리 중(in_bass=True)에 등장해서
    # bass 리스트에만 들어가고 treble에는 barline이 전혀 없다. 그 결과 treble 기준으로는
    # "barline으로 끝난 마디"가 하나도 안 잡혀(정확히 held-out 120장 표본에서 93.3%가
    # no_complete_measure) 이 보정 자체가 대보표(피아노 양손, 클래식 레퍼토리 대다수)에서
    # 사실상 항상 무력화돼 있었음 -- bass로 바꾸면 마디 역산 가능률 6.7%->81.6%,
    # 역산 가능 시 GT 일치율 77.6%, 기존 오답의 56%를 재학습 없이 구제(진단 스크립트로 확인).
    staff_toks = _extract_treble_bass(toks)[1] if is_grand else toks

    measures: List[List[str]] = []
    cur: List[str] = []
    for t in staff_toks:
        if any(t.startswith(p) for p in _HEADER_PREFIXES):
            continue
        cur.append(t)
        if t in _BARLINE_TOKEN_STRS:
            measures.append(cur)
            cur = []

    complete_sums = [_measure_beat_sum(m) for m in measures]  # barline으로 끝난 마디만
    if not complete_sums:
        return token_ids

    from collections import Counter
    majority_sum, _count = Counter(complete_sums).most_common(1)[0]

    candidates = _BEATS_TO_SIGS.get(majority_sum)
    if not candidates:
        return token_ids  # 알 수 없는 합 — 디코딩 자체 오류일 수 있어 손대지 않음

    current = toks[time_idx]

    if len(candidates) > 1:
        # 3/4 vs 6/8: 길이 합(둘 다 3/4)만으로는 안 갈림 -- dur 분포로 근사 판단.
        complete_measures = [m for m, s in zip(measures, complete_sums) if s == majority_sum]
        n_eighth  = sum(1 for m in complete_measures for t in m if t == 'dur-1/8')
        n_quarter = sum(1 for m in complete_measures for t in m if t == 'dur-1/4')
        corrected = 'time-6/8' if n_eighth > n_quarter else 'time-3/4'
    elif current in candidates:
        return token_ids  # 이미 맞음
    else:
        corrected = candidates[0]

    if corrected not in tok2id or current == corrected:
        return token_ids

    new_ids = list(token_ids)
    new_ids[time_idx] = tok2id[corrected]
    return new_ids


# ─────────────────────────────────────────────────────────────────────────────
#  치(treble) 마디 박자 합 사후 보정 (2026-08-05)
#
#  correct_time_signature와 같은 원리(베이스 마디 박자 합 다수결로 박자표를 신뢰할 수
#  있게 확정)를 한 단계 더 활용: 확정된 박자표의 기대 박자 총합과 치 마디 박자 합이
#  다르면 그 마디에서 음표 과잉/누락이 있었다는 뜻이다. diag_beatsum_flag.py로 12곡
#  95마디 검증: precision 94.7%(19개 중 18개 실제 오류), recall 37.5%(전체 오류 마디의
#  1/3 -- 길이가 같은 값끼리 대체된 오류는 박자 합이 안 바뀌므로 원리상 못 잡음).
#
#  교정은 그 마디만 박자 합 제약(남은 박자를 넘는 dur-/rest- 토큰 금지, 정확히 다 찼을
#  때만 staff-bass 허용)을 걸고 그리디로 다시 디코딩 -- 모델 가중치는 그대로, 디코딩
#  로직만 바꾸는 것이라 재학습 불필요. 제약 안에서 legal한 다음 토큰이 없거나
#  max_steps_per_measure를 넘기면 그 마디는 안전하게 원본 그대로 둔다(과감한 교정보다
#  놓치는 쪽이 안전 -- precision을 지키기 위함).
#
#  2026-08-05 실측 결과 -- run_image()에 실제로 연결해서 12곡 재검증한 결과 87.2%->85.7%로
#  오히려 하락(extra_note 28->43건 급증). 진단 단계(precision 94.7%)는 "이 마디가
#  의심스럽다"만 확인했을 뿐 "다시 디코딩하면 더 나아진다"는 검증한 적이 없었음 --
#  실제로는 제약 하에서 마스킹된 토큰이 나오면 모델이 학습 때 한 번도 안 겪어본
#  prefix를 갖게 되고(distribution shift), 그 뒤 예측이 원래 그리디보다 더 나빠지는
#  경우가 재현됨. run_image()의 실제 호출은 되돌렸고, 이 함수는 코드만 남겨둠(같은
#  방향으로 다시 시도할 때 참고용 -- diag_beatsum_flag.py의 진단(높은 precision)과
#  실제 교정 품질은 별개 문제라는 게 이번 세션의 핵심 교훈).
# ─────────────────────────────────────────────────────────────────────────────

def correct_treble_note_counts(token_ids: List[int], id2tok: dict, tok2id: dict,
                                seq2seq: OmrSeq2Seq, canvas: np.ndarray, device: torch.device,
                                is_grand: bool, max_steps_per_measure: int = 40) -> List[int]:
    if not is_grand:
        return token_ids

    toks = [id2tok.get(t, '<UNK>') for t in token_ids]
    n = len(toks)

    # 마디 목록: (치 시작idx, 치 끝idx(=staff-bass 위치, exclusive), staff-bass idx, barline idx)
    measures = []
    treble_start = 0
    i = 0
    while i < n:
        if toks[i] == 'staff-bass':
            staff_bass_idx = i
            j = i + 1
            while j < n and toks[j] not in _BARLINE_TOKEN_STRS:
                j += 1
            if j >= n:
                break  # barline 없이 시퀀스가 끝남(미완성 마디) -- 손대지 않음
            measures.append((treble_start, staff_bass_idx, staff_bass_idx, j))
            treble_start = j + 1
            i = j + 1
            continue
        i += 1
    if not measures:
        return token_ids

    def seg_beats(seg_toks):
        return _measure_beat_sum([t for t in seg_toks if not any(t.startswith(p) for p in _HEADER_PREFIXES)])

    bass_sums = [seg_beats(toks[sb + 1:bl]) for (_ts, _te, sb, bl) in measures]
    from collections import Counter
    majority_sum, _cnt = Counter(bass_sums).most_common(1)[0]
    expected = None
    for sig, beats in _TIME_SIG_BEATS.items():
        if beats == majority_sum:
            expected = beats
            break
    if expected is None:
        return token_ids  # 3/4·6/8처럼 합만으론 안 갈리거나 미지원 박자 -- 손대지 않음

    flagged = [(ts, te, sb, bl) for (ts, te, sb, bl) in measures
               if seg_beats(toks[ts:te]) != expected]
    if not flagged:
        return token_ids

    tile_f = (canvas.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
    inp = make_model_input(tile_f, _encoder_in_ch(seq2seq)).unsqueeze(0).to(device)
    seq2seq.eval()
    with torch.no_grad():
        memory = seq2seq.encode(inp)
        kv_cache = seq2seq.precompute_memory_kv(memory)

    staff_bass_id = tok2id.get('staff-bass')
    dur_costs = [(tid, _tok_duration(t)) for t, tid in tok2id.items() if _tok_duration(t) is not None]

    new_token_ids = list(token_ids)
    for (ts, te, sb, _bl) in reversed(flagged):   # 뒤에서부터: 앞쪽 인덱스가 안 흔들림
        prefix_ids = new_token_ids[:ts]
        past = torch.tensor([[SOS_ID] + prefix_ids], dtype=torch.long, device=device)
        generated: List[int] = []
        running = Fraction(0)
        scale = Fraction(1)
        ok = False
        with torch.no_grad():
            for _step in range(max_steps_per_measure):
                logits = seq2seq.decode_step_cached(kv_cache, past)[0]
                remaining = expected - running
                if staff_bass_id is not None and remaining != 0:
                    logits[staff_bass_id] = float('-inf')
                for tid, dur in dur_costs:
                    if dur * scale > remaining:
                        logits[tid] = float('-inf')
                if torch.isinf(logits).all():
                    break   # 제약 안에서 legal한 다음 토큰이 없음 -- 이 마디는 포기
                nxt = int(torch.argmax(logits).item())
                nxt_str = id2tok.get(nxt, '<UNK>')
                if nxt_str == 'staff-bass' and remaining == 0:
                    ok = True
                    break
                if nxt_str in _TUPLET_SCALE:
                    scale = _TUPLET_SCALE[nxt_str]
                elif nxt_str in _TUPLET_END:
                    scale = Fraction(1)
                else:
                    dur = _tok_duration(nxt_str)
                    if dur is not None:
                        running += dur * scale
                generated.append(nxt)
                past = torch.cat([past, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        if ok:
            new_token_ids = new_token_ids[:ts] + generated + new_token_ids[sb:]
        # 실패(ok=False)하면 이 마디는 원본 그대로 둠(precision 유지를 위한 안전한 폴백)

    return new_token_ids


# ─────────────────────────────────────────────────────────────────────────────
#  조표 임시표 사후 보정 (2026-08-03 -- held-out 실사 오류 분석의 최다 오류 패턴 대응)
#
#  이번 세션 내내 확인된 최다 오류 패턴: 조표(key-*)는 대부분 정확히 맞히면서도, 그
#  조표에 포함된 임시표가 실제로는 개별 음표 디코딩에 지속 반영되지 않음(예: key-Eb
#  (Bb/Eb/Ab)인 곡에서 Eb4->E4, Ab3->A3처럼 임시표가 빠진 자연음으로 디코딩). Round9
#  화음/임시표 강화 학습, 샤프닝 전처리, 실사 다양성 확대 등 재학습 기반 시도가 전부
#  이 패턴을 못 고쳤음 -- 재학습 없이, 이미 맞힌 조표 정보를 디코딩 결과의 개별 음표에
#  결정론적으로 재적용하는 후처리로 시도한다. 임시표가 이미 붙어있는 음표(옳든 틀리든
#  모델이 이미 명시적 판단을 내린 것)는 건드리지 않고, 임시표 없이 나온 자연음 철자가
#  조표상 임시표를 가져야 하는 음이름일 때만 교정한다.
# ─────────────────────────────────────────────────────────────────────────────

_KEY_NAME_TO_SHARPS = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6,
                       'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6}
_SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
_FLAT_ORDER  = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
_PITCH_TOK_RE = re.compile(r'^(note|chord)-([A-G])(#|b)?(\d)$')


def _key_accidental_map(key_name: str) -> Dict[str, str]:
    """조표 이름(key-* 토큰의 접미사) -> {음이름: 임시표('#'|'b')} 매핑."""
    n = _KEY_NAME_TO_SHARPS.get(key_name, 0)
    if n > 0:
        return {letter: '#' for letter in _SHARP_ORDER[:n]}
    if n < 0:
        return {letter: 'b' for letter in _FLAT_ORDER[:-n]}
    return {}


def correct_accidentals_by_key(token_ids: List[int], id2tok: dict, tok2id: dict) -> List[int]:
    """디코딩 시퀀스 안의 key-* 토큰을 찾아, 그 조표가 임시표를 요구하는 음이름인데
    임시표 없이(자연음으로) 디코딩된 note-/chord- 토큰만 조표대로 교정한다."""
    toks = [id2tok.get(tid, '') for tid in token_ids]
    key_tok = next((t for t in toks if t.startswith('key-')), None)
    if key_tok is None:
        return token_ids
    acc_map = _key_accidental_map(key_tok[len('key-'):])
    if not acc_map:
        return token_ids

    new_ids = list(token_ids)
    for i, tok in enumerate(toks):
        m = _PITCH_TOK_RE.match(tok)
        if not m or m.group(3):   # 이미 임시표가 붙어있으면 손대지 않음
            continue
        letter = m.group(2)
        if letter not in acc_map:
            continue
        new_tok = f"{m.group(1)}-{letter}{acc_map[letter]}{m.group(4)}"
        if new_tok in tok2id:
            new_ids[i] = tok2id[new_tok]
    return new_ids


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


def _strip_leading_header(token_ids: List[int], id2tok: dict) -> List[int]:
    """시퀀스 맨 앞의 헤더(clef-G, key-*, time-*)를 제거.
    다중 시스템 재조립 시 첫 시스템 이후의 중복 헤더를 떼어낼 때 사용
    (각 시스템은 dataset.py의 _split_grand_staff_interleaved()에 의해 독립 학습
    샘플로 쪼개지면서 자기 헤더를 포함하므로, 이어붙일 때는 첫 시스템만 남겨야 함)."""
    i = 0
    n = len(token_ids)
    if i < n and id2tok.get(token_ids[i], '') in ('clef-G', 'clef-F'):
        i += 1
        if i < n and id2tok.get(token_ids[i], '').startswith('key-'):
            i += 1
            if i < n and id2tok.get(token_ids[i], '').startswith('time-'):
                i += 1
    return token_ids[i:]


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
             device: torch.device, n_eval: int = 200, beam_width: int = 1):
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
            pred_toks = run_image(img_p, seq2seq, tok2id, id2tok, device, beam_width=beam_width)
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
    p.add_argument('--beam_width',  type=int, default=1,
                   help='1(기본)=greedy, 2 이상이면 beam search')
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
        eval_dir(args.eval_dir, seq2seq, tok2id, id2tok, device, args.n_eval,
                 beam_width=args.beam_width)
    elif args.image:
        tokens = run_image(args.image, seq2seq, tok2id, id2tok, device,
                           beam_width=args.beam_width)
        print("추론 결과:")
        print(' '.join(tokens))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
