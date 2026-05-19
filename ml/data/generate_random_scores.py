#!/usr/bin/env python3
"""
generate_random_scores.py  --  Round-aware random sheet music generator.

Wraps omr/data_gen/generate_dataset.py and restricts the token vocabulary
to only the symbols allowed in each training round.

Round progression (cumulative):
  Round 1 — basic notes, rests, barlines; no dynamics or articulations
  Round 2 — all Round-1 symbols + dynamics, hairpins, articulations,
             ornaments, fermata, slur, tuplets, ottava, chord tones, repeats
  Round 3 — all Round-2 symbols + grace notes, tremolos, pedal marks,
             navigation marks (coda/segno), breath marks, multi-measure rests
  Round 4 — all Round-3 symbols + complex tuplets, tempo/expression text,
             volta brackets, chord symbols, arpeggios, extended techniques

Usage:
    # Generate 100 Round-1 scores into data/Round1/
    python data/generate_random_scores.py --round 1 --count 100 --output data/Round1

    # Generate 200 Round-2 scores with a fixed seed
    python data/generate_random_scores.py --round 2 --count 200 --output data/Round2 --seed 42

    # Skip PNG rendering (faster, XML + JSON only)
    python data/generate_random_scores.py --round 1 --count 50 --output data/Round1 --no-png

    # Specify MuseScore path explicitly
    python data/generate_random_scores.py --round 2 --count 100 --output data/Round2 \\
        --musescore "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe"

Dependencies:
    pip install music21 tqdm
    MuseScore 3 or 4 must be installed for PNG rendering.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Locate generate_dataset.py relative to this file's repo position.
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_GEN_MODULE = _REPO_ROOT / "omr" / "data_gen" / "generate_dataset.py"


def _load_gen_module():
    """Dynamically load omr/data_gen/generate_dataset.py as a module."""
    spec = importlib.util.spec_from_file_location("generate_dataset", str(_GEN_MODULE))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
#  Per-round probability overrides
# ─────────────────────────────────────────────────────────────────────────────

def _apply_round_config(gen_mod, round_num: int):
    """
    Monkey-patch the probability constants in generate_dataset so only
    notation symbols appropriate for the given round are generated.
    """
    if round_num == 1:
        # Round 1: plain notes/rests only.
        gen_mod.REST_PROB     = 0.15
        gen_mod.CHORD_PROB    = 0.0
        gen_mod.BASS_CLEF_PROB = 0.20
        gen_mod.DYNAMIC_PROB  = 0.0
        gen_mod.HAIRPIN_PROB  = 0.0
        gen_mod.ARTIC_PROB    = 0.0
        gen_mod.ORNAMENT_PROB = 0.0
        gen_mod.FERMATA_PROB  = 0.0
        gen_mod.SLUR_PROB     = 0.0
        gen_mod.REPEAT_PROB   = 0.0
        gen_mod.TUPLET_PROB              = 0.0
        gen_mod.TRILL_EXT_PROB           = 0.0
        gen_mod.OTTAVA_PROB              = 0.0
        gen_mod.DOUBLE_ACCIDENTAL_PROB   = 0.0
        gen_mod.NATURAL_IN_MEASURE_PROB  = 0.0
        # Restrict key signatures to the 5 most common (C, G, D, F, Bb).
        gen_mod.KEY_SIGS    = [(0,"C"),(1,"G"),(2,"D"),(-1,"F"),(-2,"Bb")]
        gen_mod._KS_WEIGHTS = [0.30, 0.20, 0.15, 0.20, 0.15]
        # Restrict time signatures to simple meters.
        gen_mod.TIME_SIGS    = [(4,4,0.50),(3,4,0.30),(2,4,0.20)]
        gen_mod._TS_WEIGHTS  = [0.50, 0.30, 0.20]
        # Restrict durations (no 32nd notes, no dotted values beyond 3/4).
        gen_mod.DURATIONS = [
            (4.000, "1/1",  0.05),
            (3.000, "3/4",  0.05),
            (2.000, "1/2",  0.15),
            (1.000, "1/4",  0.40),
            (0.500, "1/8",  0.25),
            (0.250, "1/16", 0.10),
        ]
        gen_mod._DUR_WEIGHTS = [w for *_, w in gen_mod.DURATIONS]

    elif round_num == 2:
        # Round 2: restore full defaults from generate_dataset.py.
        # (The module was just loaded, so constants are already at default.)
        pass

    elif round_num == 3:
        # Round 3: same as Round 2 for base generation; grace notes, pedal
        # marks, and navigation marks are added in the caller via post-processing.
        # The base generator does not yet natively support these, so we keep
        # Round-2 probabilities and rely on extended tooling.
        pass

    elif round_num == 4:
        # Round 4: same base as Round 2/3 but allow all time signatures and
        # a wider measure range.
        gen_mod.MEASURES_MIN = 8
        gen_mod.MEASURES_MAX = 20
        gen_mod.TIME_SIGS = [
            (4, 4, 0.40), (3, 4, 0.20), (2, 4, 0.10),
            (6, 8, 0.10), (3, 8, 0.05), (2, 2, 0.10), (5, 4, 0.05),
        ]
        gen_mod._TS_WEIGHTS = [w for *_, w in gen_mod.TIME_SIGS]


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer: build only the subset valid for the given round
# ─────────────────────────────────────────────────────────────────────────────

def _load_round_token_set(round_num: int) -> set:
    """
    Load the round_tokens/round{N}_tokens.json and return the set of all
    allowed token strings (new_tokens list from that file and all prior rounds).
    """
    token_dir = Path(__file__).parent / "round_tokens"
    allowed   = set()
    for n in range(1, round_num + 1):
        path = token_dir / f"round{n}_tokens.json"
        if not path.exists():
            print(f"WARNING: token config not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        new_toks = data.get("new_tokens", [])
        if isinstance(new_toks, list):
            allowed.update(new_toks)
    return allowed


# ─────────────────────────────────────────────────────────────────────────────
#  Token filter: strip tokens not in the allowed set from a label sequence
# ─────────────────────────────────────────────────────────────────────────────

def _filter_tokens(tokens: list, allowed: set) -> list:
    """
    Remove any token not in the allowed vocabulary.
    Always keeps structural tokens: <SOS>, <EOS>, clef-*, key-*, time-*, barline*.
    """
    ALWAYS_KEEP = {"<SOS>", "<EOS>", "<PAD>", "<UNK>"}
    filtered = []
    for tok in tokens:
        if tok in ALWAYS_KEEP:
            filtered.append(tok)
        elif tok in allowed:
            filtered.append(tok)
        # else: silently drop; the score XML may still contain the mark
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
#  Round 4 grand staff (대보표) 생성
# ─────────────────────────────────────────────────────────────────────────────

def _split_tokens_by_measure(tokens: list) -> tuple:
    """
    토큰 시퀀스를 (헤더, [(마디_토큰_리스트, 바라인_토큰), ...]) 로 분리.

    헤더 = <SOS>, clef-*, key-*, time-* (첫 note/rest/chord/barline 직전까지)
    바라인 종결 토큰 = barline / barline-final / barline-end-repeat
    """
    BARLINE_ENDS = {"barline", "barline-final", "barline-end-repeat"}

    # 헤더 수집
    header, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if (tok.startswith("note-") or tok.startswith("rest-") or
                tok.startswith("chord-") or tok in BARLINE_ENDS or tok == "<EOS>"):
            break
        header.append(tok)
        i += 1

    # 마디별 분리
    measures, current = [], []
    while i < len(tokens):
        tok = tokens[i]
        if tok == "<EOS>":
            break
        if tok in BARLINE_ENDS:
            measures.append((list(current), tok))
            current = []
        else:
            current.append(tok)
        i += 1

    if current:
        measures.append((current, "barline-final"))

    return header, measures


def _build_grand_staff(gen_mod, score_id: int) -> tuple:
    """
    Round 4 전용: 높은음자리표 + 낮은음자리표 2파트 대보표 악보 생성.

    토큰 구조 (마디별):
        [treble 음표] staff-bass [bass 음표] barline

    낮은음자리표가 전체 쉼표 마디면 staff-bass + bass 음표 생략.
    """
    import random as _rnd
    from music21 import clef as m21_clef, key as m21_key, meter as m21_meter
    from music21.stream import Score as M21Score, Part, Measure
    from music21.note import Note, Rest
    from music21.chord import Chord as M21Chord
    from music21.pitch import Pitch as M21Pitch

    # ── 높은음자리표 파트 생성 (기존 build_score 활용) ────────────────────────
    old_bass_prob = gen_mod.BASS_CLEF_PROB
    gen_mod.BASS_CLEF_PROB = 0.0
    treble_score, treble_tokens = gen_mod.build_score(score_id)
    gen_mod.BASS_CLEF_PROB = old_bass_prob

    treble_part   = list(treble_score.parts)[0]
    treble_meas   = list(treble_part.getElementsByClass(Measure))
    n_measures    = len(treble_meas)

    # 조성·박자를 높은음자리표 파트에서 읽기
    m0     = treble_meas[0]
    ks_obj = m0.recurse().getElementsByClass(m21_key.KeySignature).first()
    ts_obj = m0.recurse().getElementsByClass(m21_meter.TimeSignature).first()
    ks_sharps  = ks_obj.sharps if ks_obj else 0
    ts_str     = f"{ts_obj.numerator}/{ts_obj.denominator}" if ts_obj else "4/4"
    measure_ql = (ts_obj.numerator * 4.0 / ts_obj.denominator) if ts_obj else 4.0

    # ── 낮은음자리표 파트 생성 ────────────────────────────────────────────────
    bass_part = Part()
    bass_part.insert(0, m21_clef.BassClef())
    bass_part.insert(0, m21_key.KeySignature(ks_sharps))
    bass_part.insert(0, m21_meter.TimeSignature(ts_str))

    bass_toks_per_measure = []

    for m_idx in range(n_measures):
        m      = Measure(number=m_idx + 1)
        m_toks = []
        remaining = measure_ql

        while remaining > 0.001:
            ql, dur_tok = gen_mod._choose_duration(remaining)
            ql = min(float(ql), remaining)

            r = _rnd.random()
            if r < gen_mod.REST_PROB:
                m.append(Rest(quarterLength=ql))
                m_toks.append(f"rest-{dur_tok}")
            elif r < gen_mod.REST_PROB + 0.30 and abs(remaining - measure_ql) < 0.001:
                # 첫 박자에 화음
                n_notes = _rnd.randint(2, 3)
                pitches = _rnd.sample(gen_mod.BASS_PITCHES,
                                      min(n_notes, len(gen_mod.BASS_PITCHES)))
                pitches.sort(key=lambda p: M21Pitch(p).midi)
                m.append(M21Chord(pitches, quarterLength=ql))
                first = gen_mod._normalize_pitch_name(M21Pitch(pitches[0]).nameWithOctave)
                m_toks.append(f"note-{first}-{dur_tok}")
                for p_str in pitches[1:]:
                    m_toks.append(
                        f"chord-{gen_mod._normalize_pitch_name(M21Pitch(p_str).nameWithOctave)}"
                    )
            else:
                p_str = _rnd.choice(gen_mod.BASS_PITCHES)
                m.append(Note(p_str, quarterLength=ql))
                name = gen_mod._normalize_pitch_name(M21Pitch(p_str).nameWithOctave)
                m_toks.append(f"note-{name}-{dur_tok}")

            remaining -= ql

        bass_part.append(m)
        bass_toks_per_measure.append(m_toks)

    # ── 두 파트를 하나의 Score 로 합치기 ─────────────────────────────────────
    combined_score = M21Score()
    combined_score.append(treble_part)
    combined_score.append(bass_part)

    # ── 토큰 시퀀스 병합: [treble] staff-bass [bass] barline ──────────────────
    header, treble_measures = _split_tokens_by_measure(treble_tokens)
    n_use = min(len(treble_measures), len(bass_toks_per_measure))

    combined_tokens = list(header)
    bass_clef_emitted = False   # 아래 오선 클레프 첫 등장 여부

    for i in range(n_use):
        m_toks, barline_tok = treble_measures[i]
        bass_m  = bass_toks_per_measure[i]

        combined_tokens.extend(m_toks)

        # 전체 쉼표가 아닌 경우에만 staff-bass + bass 토큰 추가
        non_rest = [t for t in bass_m if not t.startswith("rest-")]
        if non_rest or (bass_m and len(bass_m) > 1):
            combined_tokens.append("staff-bass")
            # 아래 오선 클레프 첫 등장 시 명시 (합성 데이터는 항상 낮은음자리표)
            if not bass_clef_emitted:
                combined_tokens.append("clef-F")
                bass_clef_emitted = True
            combined_tokens.extend(bass_m)

        combined_tokens.append(barline_tok)

    combined_tokens.append("<EOS>")
    return combined_score, combined_tokens


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Round-aware random sheet music generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--round",     type=int, required=True, choices=[1, 2, 3, 4],
                        help="Training round (1=simplest … 4=most complex)")
    parser.add_argument("--count",     type=int, default=100,
                        help="Number of scores to generate (default: 100)")
    parser.add_argument("--output",    type=str, default=None,
                        help="Output directory (default: data/Round{N})")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--musescore", type=str, default=None,
                        help="Path to MuseScore executable for PNG rendering")
    parser.add_argument("--no-png",    action="store_true",
                        help="Skip PNG rendering (output XML + JSON only)")
    parser.add_argument("--start-idx", type=int, default=1,
                        help="Starting index for output file names (default: 1)")
    args = parser.parse_args()

    # ── Defaults ─────────────────────────────────────────────────────────────
    if args.output is None:
        args.output = str(_REPO_ROOT / "data" / f"Round{args.round}")

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed : {args.seed}")

    # ── Load the generator module ─────────────────────────────────────────────
    if not _GEN_MODULE.exists():
        sys.exit(f"ERROR: generator module not found: {_GEN_MODULE}")

    gen_mod = _load_gen_module()

    # ── Apply per-round probability overrides ─────────────────────────────────
    _apply_round_config(gen_mod, args.round)

    # ── Prepare output directory ──────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve MuseScore path ────────────────────────────────────────────────
    musescore_path = None
    if not args.no_png:
        musescore_path = gen_mod.find_musescore(args.musescore)
        if musescore_path:
            print(f"MuseScore   : {musescore_path}")
        else:
            print("WARNING: MuseScore not found — PNG rendering will be skipped.")

    # ── Allowed token set for this round ──────────────────────────────────────
    allowed_tokens = _load_round_token_set(args.round)
    print(f"Round       : {args.round}  ({len(allowed_tokens)} allowed token types)")
    print(f"Generating  : {args.count} scores into {out_dir.resolve()}")

    # ── Generation loop ───────────────────────────────────────────────────────
    try:
        from tqdm import tqdm as _tqdm
        _iter = lambda x: _tqdm(x, desc=f"Round {args.round}", unit="score")
    except ImportError:
        _iter = lambda x: x

    ok_xml = ok_png = 0

    for i in _iter(range(args.start_idx, args.start_idx + args.count)):
        stem     = f"num{i}"
        xml_path = out_dir / f"{stem}.musicxml"
        png_path = out_dir / f"{stem}.png"
        lbl_path = out_dir / f"{stem}.json"

        try:
            if args.round == 4:
                score, tokens = _build_grand_staff(gen_mod, i)
            else:
                score, tokens = gen_mod.build_score(i)
        except Exception as exc:
            print(f"  [ERROR] {stem}: {exc}")
            continue

        # ── Filter tokens to round vocabulary ─────────────────────────────────
        tokens = _filter_tokens(tokens, allowed_tokens)

        try:
            score.write("musicxml", fp=str(xml_path))
            ok_xml += 1
        except Exception as exc:
            print(f"  [ERROR] {stem} XML write: {exc}")
            continue

        lbl_path.write_text(
            __import__("json").dumps({"id": stem, "tokens": tokens},
                                     ensure_ascii=False),
            encoding="utf-8",
        )

        if musescore_path:
            ok = gen_mod.render_png(musescore_path, xml_path, png_path)
            if ok:
                ok_png += 1

    print(f"\n{'─'*50}")
    print(f"Round       : {args.round}")
    print(f"Generated   : {ok_xml}/{args.count} MusicXML files")
    if not args.no_png:
        if musescore_path:
            print(f"Rendered    : {ok_png}/{ok_xml} PNG files")
        else:
            print("Rendered    : 0  (MuseScore not available)")
    print(f"Labels      : {ok_xml} JSON files")
    print(f"Output dir  : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
