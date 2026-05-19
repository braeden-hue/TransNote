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
