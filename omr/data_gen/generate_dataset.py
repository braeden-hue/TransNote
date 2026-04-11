#!/usr/bin/env python3
"""
generate_dataset.py

Generates N random sheet music pairs for OMR training.
Each pair shares the same filename stem:
    data/train/num1.musicxml  ← music notation data
    data/train/num1.png       ← rendered image (via MuseScore)
    data/train/num1.json      ← DeepScore token sequence label

DeepScore token format examples:
    <SOS>  clef-G  key-C  time-4/4
    note-C4-1/4  note-E4-1/4  chord-G4  rest-1/8  note-A4-1/8
    barline  ...  barline-final  <EOS>

Usage:
    # install dependencies first
    pip install -r scripts/requirements.txt

    # generate 100 pairs (default)
    python scripts/generate_dataset.py

    # custom count / output / seed
    python scripts/generate_dataset.py -n 500 -o data/train --seed 42

    # skip PNG rendering (MusicXML + labels only)
    python scripts/generate_dataset.py --no-png

    # specify MuseScore path manually
    python scripts/generate_dataset.py --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

# ── music21 ──────────────────────────────────────────────────────────────────
try:
    from music21 import (
        bar, chord, clef, key, meter, note, stream, tempo
    )
    from music21.bar import Barline
    from music21.chord import Chord
    from music21.note import Note, Rest
    from music21.pitch import Pitch
    from music21.stream import Measure, Part, Score
except ImportError:
    print("ERROR: music21 not found.\n  Run: pip install music21")
    sys.exit(1)

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
#  Static configuration
# ─────────────────────────────────────────────────────────────────────────────

# Common MuseScore install paths (Windows / Linux / macOS)
MUSESCORE_CANDIDATES = [
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
    "/usr/bin/mscore4",
    "/usr/bin/mscore3",
    "/usr/bin/mscore",
    "/usr/bin/musescore4",
    "/usr/bin/musescore3",
    "/usr/bin/musescore",
    "/usr/local/bin/musescore",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
]

# Rendered PNG resolution - A4 @ 150 DPI ≈ 1240 × 1754 px
# Matches the lower end of smartphone capture after preprocessing.
PNG_DPI = 150

# ── Time signatures ───────────────────────────────────────────────────────────
# (numerator, denominator, sampling_weight)
TIME_SIGS = [
    (4, 4, 0.50),
    (3, 4, 0.25),
    (2, 4, 0.15),
    (6, 8, 0.10),
]
_TS_WEIGHTS = [w for *_, w in TIME_SIGS]

# ── Key signatures ────────────────────────────────────────────────────────────
# (sharps_count, major_key_name)
#   sharps_count > 0 -> sharps,  < 0 -> flats
KEY_SIGS = [
    ( 0, "C"),
    ( 1, "G"),
    (-1, "F"),
    ( 2, "D"),
    (-2, "Bb"),
    ( 3, "A"),
    (-3, "Eb"),
]
_KS_WEIGHTS = [0.30, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]

# ── Note durations ────────────────────────────────────────────────────────────
# (quarterLength, token_string, sampling_weight)
DURATIONS = [
    (4.000, "1/1",  0.04),
    (3.000, "3/4",  0.04),   # dotted half  = 3/4 of whole note
    (2.000, "1/2",  0.14),
    (1.500, "3/8",  0.06),   # dotted quarter
    (1.000, "1/4",  0.38),
    (0.750, "3/16", 0.04),   # dotted eighth
    (0.500, "1/8",  0.22),
    (0.250, "1/16", 0.08),
]
_DUR_WEIGHTS = [w for *_, w in DURATIONS]
# Quick lookup: quarterLength -> token
QL_TO_TOKEN = {round(ql, 6): tok for ql, tok, _ in DURATIONS}

# Probability settings
REST_PROB  = 0.15   # chance any beat-slot becomes a rest
CHORD_PROB = 0.10   # chance any note becomes a 2–3 note chord

# Measures per score
MEASURES_MIN = 4
MEASURES_MAX = 8

# Bass clef appearance probability
BASS_CLEF_PROB = 0.20

# ── Pitch pools ───────────────────────────────────────────────────────────────
_PITCH_CLASSES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]


def _pitch_pool(lo: str, hi: str) -> list:
    lo_midi = Pitch(lo).midi
    hi_midi = Pitch(hi).midi
    pool = []
    for oct in range(2, 7):
        for pc in _PITCH_CLASSES:
            try:
                p = Pitch(f"{pc}{oct}")
                if lo_midi <= p.midi <= hi_midi:
                    pool.append(p.nameWithOctave)
            except Exception:
                pass
    return pool


TREBLE_PITCHES = _pitch_pool("C4", "B5")   # comfortable treble range
BASS_PITCHES   = _pitch_pool("C2", "B3")   # comfortable bass range


# ─────────────────────────────────────────────────────────────────────────────
#  Duration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _choose_duration(max_ql: float) -> tuple:
    """Return (quarterLength, token) for a random duration ≤ max_ql."""
    possible = [(ql, tok, w) for ql, tok, w in DURATIONS if ql <= max_ql + 1e-9]
    if not possible:
        # Fall back to the shortest available duration
        ql, tok, _ = min(DURATIONS, key=lambda x: x[0])
        return ql, tok
    weights = [w for *_, w in possible]
    ql, tok, _ = random.choices(possible, weights=weights)[0]
    return min(ql, max_ql), tok


def _ql_to_token(ql: float) -> str:
    """Convert a quarterLength to a fraction token string (e.g. 1.5 -> '3/8')."""
    t = QL_TO_TOKEN.get(round(ql, 6))
    if t:
        return t
    f = Fraction(ql).limit_denominator(32)
    # Normalise to fraction of a whole note: multiply by 1/4
    normalised = Fraction(f.numerator, f.denominator * 4)
    return f"{normalised.numerator}/{normalised.denominator}"


def _normalize_pitch_name(name: str) -> str:
    """Convert music21 pitch name to token-safe form: 'D-4' -> 'Db4'."""
    return name.replace("-", "b")


# ─────────────────────────────────────────────────────────────────────────────
#  Element factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_element(pitch_pool: list, ql: float):
    """Create a random Note / Rest / Chord with the given quarterLength."""
    r = random.random()
    if r < REST_PROB:
        return Rest(quarterLength=ql)
    if r < REST_PROB + CHORD_PROB:
        n_notes = random.randint(2, 3)
        chosen = random.sample(pitch_pool, min(n_notes, len(pitch_pool)))
        chosen.sort(key=lambda p: Pitch(p).midi)
        return Chord(chosen, quarterLength=ql)
    return Note(random.choice(pitch_pool), quarterLength=ql)


# ─────────────────────────────────────────────────────────────────────────────
#  Token generation
# ─────────────────────────────────────────────────────────────────────────────

def _element_to_tokens(el) -> list:
    """
    Convert a single music21 element to a list of DeepScore tokens.

    Chord encoding:
        The lowest note gets the full  note-{pitch}-{dur}  token.
        Higher chord notes get          chord-{pitch}       tokens
        (they share the duration of the preceding note token).
    """
    if isinstance(el, Rest):
        tok = _ql_to_token(el.duration.quarterLength)
        return [f"rest-{tok}"]

    if isinstance(el, Chord):
        tok = _ql_to_token(el.duration.quarterLength)
        tokens = []
        sorted_pitches = sorted(el.pitches, key=lambda p: p.midi)
        for i, p in enumerate(sorted_pitches):
            name = _normalize_pitch_name(p.nameWithOctave)
            if i == 0:
                tokens.append(f"note-{name}-{tok}")
            else:
                tokens.append(f"chord-{name}")
        return tokens

    if isinstance(el, Note):
        tok = _ql_to_token(el.duration.quarterLength)
        name = _normalize_pitch_name(el.pitch.nameWithOctave)
        return [f"note-{name}-{tok}"]

    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Score builder
# ─────────────────────────────────────────────────────────────────────────────

def build_score(score_id: int) -> tuple:
    """
    Build a random music21 Score and its DeepScore token sequence.
    Returns (Score, list[str]).
    """
    # ── random choices ────────────────────────────────────────────────────────
    ts_num, ts_den, _ = random.choices(TIME_SIGS, weights=_TS_WEIGHTS)[0]
    ks_sharps, ks_name = random.choices(KEY_SIGS, weights=_KS_WEIGHTS)[0]
    use_bass = random.random() < BASS_CLEF_PROB
    pitch_pool = BASS_PITCHES if use_bass else TREBLE_PITCHES
    n_measures = random.randint(MEASURES_MIN, MEASURES_MAX)
    bpm = random.randint(60, 130)

    clef_obj = clef.BassClef() if use_bass else clef.TrebleClef()
    clef_tok = "clef-F" if use_bass else "clef-G"

    # Total quarter lengths per measure
    measure_ql = ts_num * (4.0 / ts_den)

    # ── build music21 Part ────────────────────────────────────────────────────
    part = Part()
    part.insert(0, clef_obj)
    part.insert(0, key.KeySignature(ks_sharps))
    part.insert(0, meter.TimeSignature(f"{ts_num}/{ts_den}"))
    part.insert(0, tempo.MetronomeMark(number=bpm))

    # ── build token list ──────────────────────────────────────────────────────
    tokens = [
        "<SOS>",
        clef_tok,
        f"key-{ks_name}",
        f"time-{ts_num}/{ts_den}",
    ]

    for m_idx in range(n_measures):
        m = Measure(number=m_idx + 1)

        # Fill measure to exactly measure_ql
        remaining = measure_ql
        while remaining > 1e-9:
            ql, _ = _choose_duration(remaining)
            ql = round(min(ql, remaining), 6)
            el = _make_element(pitch_pool, ql)
            m.append(el)
            tokens.extend(_element_to_tokens(el))
            remaining = round(remaining - ql, 6)

        # Barline tokens
        if m_idx < n_measures - 1:
            tokens.append("barline")
        else:
            m.rightBarline = Barline("final")
            tokens.append("barline-final")

        part.append(m)

    tokens.append("<EOS>")

    score = Score()
    score.insert(0, part)
    return score, tokens


# ─────────────────────────────────────────────────────────────────────────────
#  MuseScore rendering
# ─────────────────────────────────────────────────────────────────────────────

def find_musescore(override: str = None) -> str:
    """Locate the MuseScore executable. Returns path string or None."""
    if override:
        p = Path(override)
        if p.exists():
            return str(p)
        print(f"WARNING: Specified MuseScore path not found: {override}")

    for candidate in MUSESCORE_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    for cmd in ["musescore4", "musescore3", "musescore", "mscore4", "mscore3", "mscore"]:
        found = shutil.which(cmd)
        if found:
            return found

    return None


def render_png(musescore: str, xml_path: Path, png_path: Path) -> bool:
    """
    Render xml_path to png_path using MuseScore CLI.

    MuseScore appends "-1" to the stem for page 1 (e.g. tmp-1.png).
    We rename that to png_path after export.

    Returns True on success.
    """
    # Use a temp name so the "-1" suffix is predictable
    tmp_stem = png_path.with_name(png_path.stem + "_ms_tmp.png")

    cmd = [
        musescore,
        "-o", str(tmp_stem),
        "-r", str(PNG_DPI),
        str(xml_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        print(f"  [WARN] MuseScore timed out: {xml_path.name}")
        return False
    except Exception as exc:
        print(f"  [WARN] MuseScore subprocess error: {exc}")
        return False

    # MuseScore 3/4 creates {stem}-1.png for the first (and usually only) page
    page1 = tmp_stem.with_name(tmp_stem.stem + "-1.png")
    if page1.exists():
        page1.rename(png_path)
        return True
    # Some builds produce the file without the suffix
    if tmp_stem.exists():
        tmp_stem.rename(png_path)
        return True

    print(f"  [WARN] PNG not found after rendering {xml_path.name}")
    if result.stderr:
        print(f"         stderr: {result.stderr[:400]}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer vocabulary builder
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer() -> dict:
    """
    Build a comprehensive DeepScore-style vocabulary.

    Token naming conventions:
      note-{pitch}{octave}-{dur}   e.g.  note-C4-1/4,  note-F#5-1/8,  note-Bb3-1/2
      chord-{pitch}{octave}        e.g.  chord-E4,  chord-G4
      rest-{dur}                   e.g.  rest-1/4
      clef-G / clef-F / clef-C
      key-{name}                   e.g.  key-C,  key-G,  key-Bb
      time-{num}/{den}             e.g.  time-4/4
      barline / barline-final / ...
      <PAD> / <SOS> / <EOS> / <UNK>

    Duration tokens (fraction of a whole note):
      1/1  3/4  1/2  3/8  1/4  3/16  1/8  3/32  1/16  1/32
    """
    tokens = []

    # ── special ───────────────────────────────────────────────────────────────
    tokens += ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    # ── clefs ─────────────────────────────────────────────────────────────────
    tokens += ["clef-G", "clef-F", "clef-C"]

    # ── key signatures ────────────────────────────────────────────────────────
    key_names = [
        "C", "G", "D", "A", "E", "B", "F#",   # sharp keys
        "F", "Bb", "Eb", "Ab", "Db", "Gb",     # flat keys
    ]
    tokens += [f"key-{k}" for k in key_names]

    # ── time signatures ───────────────────────────────────────────────────────
    tokens += [
        "time-2/4", "time-3/4", "time-4/4", "time-6/8",
        "time-3/8", "time-12/8", "time-2/2", "time-5/4",
    ]

    # ── duration tokens ───────────────────────────────────────────────────────
    dur_tokens = ["1/1", "3/4", "1/2", "3/8", "1/4", "3/16", "1/8", "3/32", "1/16", "1/32"]

    # ── pitch class set (includes enharmonic pairs for robustness) ────────────
    pitch_classes = [
        "C", "C#", "Db",
        "D", "D#", "Eb",
        "E",
        "F", "F#", "Gb",
        "G", "G#", "Ab",
        "A", "A#", "Bb",
        "B",
    ]
    octaves = [2, 3, 4, 5, 6]

    # ── note tokens ───────────────────────────────────────────────────────────
    for oct in octaves:
        for pc in pitch_classes:
            for dur in dur_tokens:
                tokens.append(f"note-{pc}{oct}-{dur}")

    # ── chord continuation tokens (no duration; shares with preceding note) ───
    for oct in octaves:
        for pc in pitch_classes:
            tokens.append(f"chord-{pc}{oct}")

    # ── rest tokens ───────────────────────────────────────────────────────────
    for dur in dur_tokens:
        tokens.append(f"rest-{dur}")

    # ── barlines ──────────────────────────────────────────────────────────────
    tokens += [
        "barline",
        "barline-double",
        "barline-final",
        "barline-start-repeat",
        "barline-end-repeat",
    ]

    # Deduplicate while preserving insertion order
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return {tok: idx for idx, tok in enumerate(unique)}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate random OMR training data (MusicXML + PNG pairs)"
    )
    parser.add_argument(
        "-n", "--count", type=int, default=100,
        help="Number of pairs to generate (default: 100)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="data/train",
        help="Output directory (default: data/train)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--musescore", type=str, default=None,
        help="Path to MuseScore executable (auto-detected if omitted)",
    )
    parser.add_argument(
        "--no-png", action="store_true",
        help="Skip PNG rendering - output MusicXML + label JSON only",
    )
    parser.add_argument(
        "--tokenizer-out", type=str, default="data/tokenizer.json",
        help="Where to save the tokenizer vocabulary (default: data/tokenizer.json)",
    )
    args = parser.parse_args()

    # ── seed ──────────────────────────────────────────────────────────────────
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # ── output directories ────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── tokenizer vocabulary ──────────────────────────────────────────────────
    tok_path = Path(args.tokenizer_out)
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    vocab = build_tokenizer()
    tok_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Tokenizer saved -> {tok_path}  ({len(vocab)} tokens)")

    # ── MuseScore ─────────────────────────────────────────────────────────────
    musescore_path = None
    if not args.no_png:
        musescore_path = find_musescore(args.musescore)
        if musescore_path:
            print(f"MuseScore found -> {musescore_path}")
        else:
            print(
                "WARNING: MuseScore not found - PNG rendering will be skipped.\n"
                "  Install MuseScore 3 or 4, or pass --musescore <path>, or use --no-png."
            )

    # ── generate loop ─────────────────────────────────────────────────────────
    indices = range(1, args.count + 1)
    iterator = _tqdm(indices, desc="Generating", unit="score") if HAS_TQDM else indices

    ok_xml = 0
    ok_png = 0

    for i in iterator:
        stem = f"num{i}"
        xml_path = out_dir / f"{stem}.musicxml"
        png_path = out_dir / f"{stem}.png"
        lbl_path = out_dir / f"{stem}.json"

        # Build and save MusicXML
        try:
            score, tokens = build_score(i)
            score.write("musicxml", fp=str(xml_path))
            ok_xml += 1
        except Exception as exc:
            print(f"  [ERROR] {stem}: score generation failed - {exc}")
            continue

        # Save DeepScore label sequence
        lbl_path.write_text(
            json.dumps({"id": stem, "tokens": tokens}, ensure_ascii=False),
            encoding="utf-8",
        )

        # Render PNG
        if musescore_path:
            ok = render_png(musescore_path, xml_path, png_path)
            if ok:
                ok_png += 1
            elif not HAS_TQDM:
                print(f"  [{i}/{args.count}] {stem}: XML OK  PNG FAIL")
        elif not HAS_TQDM and not args.no_png:
            # No MuseScore available
            pass

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'-'*50}")
    print(f"Generated  : {ok_xml}/{args.count} MusicXML files")
    if not args.no_png:
        if musescore_path:
            print(f"Rendered   : {ok_png}/{ok_xml} PNG files")
        else:
            print(f"Rendered   : 0  (MuseScore not available)")
    print(f"Labels     : {ok_xml} JSON label files")
    print(f"Output dir : {out_dir.resolve()}")
    print(f"Tokenizer  : {tok_path.resolve()}  ({len(vocab)} tokens)")


if __name__ == "__main__":
    main()
