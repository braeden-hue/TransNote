"""
round3train/compare_recognition.py

최신 인식 가능 범위(4span_w10 체크포인트 학습 스코프)에 맞는 테스트 악보를 N개 생성하고,
모델이 예측한 토큰을 다시 악보(PNG)로 렌더링해서 실제 악보 PNG와 나란히 비교할 수 있게
출력 폴더에 저장한다.

스코프 (curriculum_4t_4sym.sh의 4tup 단계 gen 인자와 동일):
  대보표 1개, 1~4마디(밀도 기반 줄바꿈), 조표 13종, 화음, 다이나믹(pp/p/mp/mf/f/ff),
  헤어핀, 페르마타, 옥타브, 셋잇단음표. artic/ornament/slur/반복기호/tie는 스코프 밖(0).

사용법:
    python round3train/compare_recognition.py --count 10 --output round3train/compare_out \
        --seq2seq ../secrets/checkpoints/seq2seq_p2s4span_w10_best.pt
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from music21 import bar, clef, dynamics, expressions, key, meter, spanner, duration as dur_mod
from music21.note import Note, Rest
from music21.chord import Chord
from music21.stream import Measure, Part, Score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_scores as gs
from model import OmrSeq2Seq
from dataset import load_tokenizer
from inference import run_image


# ─── 최신 스코프로 생성 파라미터 고정 (curriculum_4t_4sym.sh 4tup 단계와 동일) ─────────────

def _apply_current_scope(ottava_prob: float = 0.35, hairpin_prob: float = 0.20):
    gs.MIN_MEASURES     = 1
    gs.MAX_MEASURES     = 4
    gs.DENSITY_BREAK     = True
    gs.CHORD_PROB        = 0.08
    gs.REPEAT_PROB       = 0.0
    gs.ARTIC_PROB        = 0.0
    gs.ORNAMENT_PROB     = 0.0
    gs.SLUR_PROB         = 0.0
    gs.TUPLET_PROB       = 0.35
    gs.OTTAVA_PROB       = ottava_prob
    gs.DIATONIC_BIAS     = 0.75
    gs.DYNAMIC_PROB      = 0.35
    gs.DYNAMICS_LIST     = ['p', 'f', 'pp', 'ff', 'mp', 'mf']
    gs.HAIRPIN_PROB      = hairpin_prob
    gs.FERMATA_PROB      = 0.04


DUR_TOK_TO_QL = {tok: ql for ql, tok, _w in gs.DURATIONS}
KEY_NAME_TO_SHARPS = {name: sharps for sharps, name in gs.KEY_SIGS}


def _restore_pitch(tok: str) -> str:
    # generate_scores._np()가 '-'(flat) -> 'b'로 바꿔서 토큰화했으므로 역변환.
    # 음이름 철자(A-G)는 항상 대문자라 소문자 'b'와 절대 충돌하지 않는다.
    return tok.replace('b', '-')


class _PartBuilder:
    """예측 토큰(measure별로 분리된 리스트)을 music21 Part로 복원."""

    def __init__(self, clef_obj, ks_sharps, ts_num, ts_den):
        self.part = Part()
        self.part.insert(0, clef_obj)
        self.part.insert(0, key.KeySignature(ks_sharps))
        self.part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))
        self.last_element = None
        self.hairpin_state = None   # {'type', 'start', 'awaiting_start'}
        self.ottava_state  = None
        self.hairpin_spans = []
        self.ottava_spans  = []

    def _register(self, el):
        self.last_element = el
        if self.hairpin_state and self.hairpin_state['awaiting_start']:
            self.hairpin_state['start'] = el
            self.hairpin_state['awaiting_start'] = False
        if self.ottava_state and self.ottava_state['awaiting_start']:
            self.ottava_state['start'] = el
            self.ottava_state['awaiting_start'] = False

    def add_measure(self, m_idx: int, m_toks: list):
        m = Measure(number=m_idx + 1)
        i, n = 0, len(m_toks)
        while i < n:
            t = m_toks[i]
            try:
                if t.startswith('dynamic-'):
                    m.insert(0, dynamics.Dynamic(t[len('dynamic-'):]))
                    i += 1; continue
                if t.startswith('hairpin-') and t.endswith('-start'):
                    self.hairpin_state = {'type': t[len('hairpin-'):-len('-start')],
                                          'start': None, 'awaiting_start': True}
                    i += 1; continue
                if t.startswith('hairpin-') and t.endswith('-end'):
                    if self.hairpin_state and self.last_element is not None:
                        self.hairpin_state['end'] = self.last_element
                        if self.hairpin_state['start'] is not None:
                            self.hairpin_spans.append(self.hairpin_state)
                    self.hairpin_state = None
                    i += 1; continue
                if t.startswith('ottava-') and t.endswith('-start'):
                    self.ottava_state = {'type': t[len('ottava-'):-len('-start')],
                                         'start': None, 'awaiting_start': True}
                    i += 1; continue
                if t.startswith('ottava-') and t.endswith('-end'):
                    if self.ottava_state and self.last_element is not None:
                        self.ottava_state['end'] = self.last_element
                        if self.ottava_state['start'] is not None:
                            self.ottava_spans.append(self.ottava_state)
                    self.ottava_state = None
                    i += 1; continue
                if t == 'tuplet-3-start':
                    i += 1
                    while i < n and m_toks[i] != 'tuplet-3-end':
                        if m_toks[i].startswith('note-'):
                            pname = _restore_pitch(m_toks[i][len('note-'):])
                            if i + 1 < n and m_toks[i + 1].startswith('dur-'):
                                i += 1
                            no = Note(pname)
                            no.duration.quarterLength = 1.0 / 3.0
                            no.duration.appendTuplet(dur_mod.Tuplet(3, 2))
                            m.append(no)
                            self._register(no)
                        i += 1
                    if i < n and m_toks[i] == 'tuplet-3-end':
                        i += 1
                    continue
                if t.startswith('note-'):
                    pname_tok = t[len('note-'):]
                    j = i + 1
                    dur_tok = None
                    if j < n and m_toks[j].startswith('dur-'):
                        dur_tok = m_toks[j][len('dur-'):]
                        j += 1
                    chord_pitches = []
                    while j < n and m_toks[j].startswith('chord-'):
                        chord_pitches.append(m_toks[j][len('chord-'):])
                        j += 1
                    ql = DUR_TOK_TO_QL.get(dur_tok, 1.0)
                    root_name = _restore_pitch(pname_tok)
                    if chord_pitches:
                        names = [root_name] + [_restore_pitch(p) for p in chord_pitches]
                        el = Chord(names, quarterLength=ql)
                    else:
                        el = Note(root_name, quarterLength=ql)
                    m.append(el)
                    self._register(el)
                    i = j
                    continue
                if t.startswith('rest-'):
                    ql = DUR_TOK_TO_QL.get(t[len('rest-'):], 1.0)
                    m.append(Rest(quarterLength=ql))
                    i += 1; continue
                if t == 'fermata':
                    if self.last_element is not None:
                        self.last_element.expressions.append(expressions.Fermata())
                    i += 1; continue
            except Exception:
                pass
            # artic-*/ornament-*/slur-*/기타 미지 토큰: 현재 스코프 밖이므로 무시
            i += 1
        self.part.append(m)

    def finalize(self):
        for hp in self.hairpin_spans:
            try:
                obj = dynamics.Crescendo() if hp['type'] == 'cresc' else dynamics.Diminuendo()
                obj.addSpannedElements([hp['start'], hp['end']])
                self.part.insert(0, obj)
            except Exception:
                pass
        for ott in self.ottava_spans:
            try:
                obj = spanner.Ottava(type=ott['type'])
                obj.addSpannedElements([ott['start'], ott['end']])
                self.part.insert(0, obj)
            except Exception:
                pass
        if len(self.part.getElementsByClass(Measure)) > 0:
            self.part.getElementsByClass(Measure)[-1].rightBarline = bar.Barline('final')
        return self.part


def tokens_to_score(tokens: list) -> Score:
    """예측 토큰 시퀀스(<SOS>/<EOS> 포함 가능) -> music21 Score. build_score_r3()의 역변환.
    모델이 틀리게 예측한 토큰(미지/누락)은 최대한 관대하게 건너뛴다."""
    toks = [t for t in tokens if t not in ('<SOS>', '<EOS>', '<PAD>')]
    if not toks:
        raise ValueError("빈 토큰 시퀀스")

    ks_sharps, ts_num, ts_den = 0, 4, 4
    idx = 0
    while idx < len(toks) and (toks[idx].startswith('clef-') or
                                toks[idx].startswith('key-') or
                                toks[idx].startswith('time-')):
        t = toks[idx]
        if t.startswith('key-'):
            ks_sharps = KEY_NAME_TO_SHARPS.get(t[len('key-'):], 0)
        elif t.startswith('time-'):
            try:
                n_s, d_s = t[len('time-'):].split('/')
                ts_num, ts_den = int(n_s), int(d_s)
            except Exception:
                pass
        idx += 1
    body = toks[idx:]

    treble_measures, bass_measures = [], []
    cur_t, cur_b = [], []
    state = 'treble'
    clef_f_seen = False
    has_bass = False
    for t in body:
        if t == 'barline-start-repeat':
            continue
        if state == 'treble':
            if t == 'staff-bass':
                has_bass = True
                treble_measures.append(cur_t); cur_t = []
                state = 'bass'
            else:
                cur_t.append(t)
        else:
            if t == 'clef-F' and not clef_f_seen:
                clef_f_seen = True
            elif t in ('barline', 'barline-final', 'barline-end-repeat'):
                bass_measures.append(cur_b); cur_b = []
                state = 'treble'
            else:
                cur_b.append(t)
    if cur_t:
        treble_measures.append(cur_t)
    if cur_b:
        bass_measures.append(cur_b)

    n_measures = max(len(treble_measures), len(bass_measures), 1)
    while len(treble_measures) < n_measures:
        treble_measures.append([])
    while len(bass_measures) < n_measures:
        bass_measures.append([])

    tb = _PartBuilder(clef.TrebleClef(), ks_sharps, ts_num, ts_den)
    for i, mt in enumerate(treble_measures):
        tb.add_measure(i, mt)

    score = Score()
    score.insert(0, tb.finalize())
    if has_bass:
        # staff-bass 토큰이 실제로 없는 단일 오선 곡에도 항상 빈 베이스 파트를 만들어
        # 삽입하던 버그(2026-07-31 발견) -- MuseScore가 2개 오선 분량 세로 공간을
        # 요구하면서 단일오선용 좁은 페이지 스타일(wide_page.mss, 4in 높이)에 안 들어가
        # 실제 악보가 2페이지로 밀려나고, render_png()는 1페이지만 캡처해 "제목만 있는
        # 빈 페이지"를 저장함(_looks_blank()가 텍스트 잉크 때문에 못 걸러냄) --
        # exactPicture 89곡 중 5곡(전부 단일오선)이 이 버그로 Acc=0.0%였음.
        bb = _PartBuilder(clef.BassClef(), ks_sharps, ts_num, ts_den)
        for i, mt in enumerate(bass_measures):
            bb.add_measure(i, mt)
        score.insert(0, bb.finalize())
    return score


def main():
    ap = argparse.ArgumentParser(description="최신 스코프 테스트 데이터 생성 + 예측 렌더링 비교")
    ap.add_argument('--count', type=int, default=10)
    ap.add_argument('--output', default='round3train/compare_out')
    ap.add_argument('--seed', type=int, default=777001)
    ap.add_argument('--seq2seq', default='../secrets/checkpoints/seq2seq_p2s4span_w10_best.pt')
    ap.add_argument('--tokenizer', default='tokenizer258.json')
    ap.add_argument('--musescore', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ottava-prob', type=float, default=0.35,
                     help='OTTAVA_PROB (기본 4tup 스코프값 0.35). 0으로 주면 옥타브 제외 테스트')
    ap.add_argument('--hairpin-prob', type=float, default=0.20,
                     help='HAIRPIN_PROB (기본 4tup 스코프값 0.20, 크레센도/디크레센도). 0으로 주면 헤어핀 제외 테스트')
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    _apply_current_scope(ottava_prob=args.ottava_prob, hairpin_prob=args.hairpin_prob)

    musescore_path = gs.find_musescore(args.musescore)
    if not musescore_path:
        raise SystemExit("MuseScore 실행 파일을 찾을 수 없음 (--musescore로 경로 지정)")
    print(f"MuseScore: {musescore_path}")

    out_dir = Path(args.output)
    (out_dir / '_work').mkdir(parents=True, exist_ok=True)
    work = out_dir / '_work'

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(args.seq2seq, map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.seq2seq}")

    report_lines = []
    for i in range(1, args.count + 1):
        score, gt_tokens, _breaks = gs.build_score_r3(i, force_c_major=False)
        gt_xml = work / f"gt{i}.musicxml"
        gt_png = work / f"gt{i}.png"
        score.write("musicxml", fp=str(gt_xml))
        if not gs.render_png(musescore_path, gt_xml, gt_png):
            print(f"  [WARN] gt{i} 렌더링 실패, 건너뜀")
            continue

        pred_tokens = run_image(str(gt_png), seq2seq, tok2id, id2tok, device)

        try:
            pred_score = tokens_to_score(pred_tokens)
            pred_xml = work / f"pred{i}.musicxml"
            pred_png = work / f"pred{i}.png"
            pred_score.write("musicxml", fp=str(pred_xml))
            rendered = gs.render_png(musescore_path, pred_xml, pred_png)
        except Exception as exc:
            print(f"  [WARN] {i}번 예측 -> 악보 복원 실패: {exc}")
            rendered = False

        train_name = out_dir / f"train{i}.png"
        shutil.copyfile(gt_png, train_name)
        if rendered:
            test_name = out_dir / f"test{i}.png"
            shutil.copyfile(pred_png, test_name)
            status = "OK"
        else:
            status = "예측 렌더링 실패(train만 존재)"

        gt_clean   = [t for t in gt_tokens if t not in ('<SOS>', '<EOS>')]
        exact = (gt_clean == pred_tokens)
        print(f"[{i}/{args.count}] {status}  exact_match={exact}")
        report_lines.append(f"=== {i} ({status}, exact_match={exact}) ===")
        report_lines.append("GT  : " + ' '.join(gt_clean))
        report_lines.append("PRED: " + ' '.join(pred_tokens))
        report_lines.append("")

    (out_dir / "compare_report.txt").write_text('\n'.join(report_lines), encoding='utf-8')
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n완료: {out_dir.resolve()}")
    print(f"리포트: {(out_dir / 'compare_report.txt').resolve()}")


if __name__ == '__main__':
    main()
