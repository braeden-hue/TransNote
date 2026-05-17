#!/usr/bin/env python3
"""
generate_dataset.py

Generates N random sheet music pairs for OMR training.
Each pair shares the same filename stem:
    data/train/num1.musicxml  ← music notation data
    data/train/num1.png       ← rendered image (via MuseScore)
    data/train/num1.json      ← DeepScore token sequence label

Token format:
    <SOS> clef-G key-C time-4/4
    dynamic-f
    barline-start-repeat
    hairpin-cresc-start
    note-C4-1/4 artic-staccato
    note-E4-1/4 ornament-trill
    note-G4-1/2 fermata
    hairpin-cresc-end dynamic-ff
    barline-end-repeat barline-final
    <EOS>

Token placement rules:
  - dynamic-*         : standalone token BEFORE the notes it applies to
  - hairpin-*-start/end : wraps the note group it spans
  - artic-*           : immediately AFTER the note/chord token(s)
  - ornament-*        : immediately AFTER the note token
  - fermata           : immediately AFTER the note token
  - slur-start/end    : wraps the slurred note group

Usage:
    pip install -r omr/data_gen/requirements.txt

    python omr/data_gen/generate_dataset.py --count 100 --out_dir data/train \\
      --musescore "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe"
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
        articulations as m21_artic,
        bar, chord, clef,
        dynamics as m21_dyn,
        expressions as m21_expr,
        key, meter, note,
        spanner as m21_spanner,
        stream, tempo,
    )
    from music21.bar import Barline, Repeat
    from music21.chord import Chord
    from music21.note import Note, Rest
    from music21.pitch import Accidental, Pitch
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
#  MuseScore install paths
# ─────────────────────────────────────────────────────────────────────────────

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

PNG_DPI = 150   # A4 @ 150 DPI ≈ 1240 × 1754 px


# ─────────────────────────────────────────────────────────────────────────────
#  Time / Key / Duration tables
# ─────────────────────────────────────────────────────────────────────────────

TIME_SIGS = [
    (4, 4, 0.50),
    (3, 4, 0.25),
    (2, 4, 0.15),
    (6, 8, 0.10),
]
_TS_WEIGHTS = [w for *_, w in TIME_SIGS]

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

DURATIONS = [
    (4.000, "1/1",  0.04),
    (3.000, "3/4",  0.04),
    (2.000, "1/2",  0.14),
    (1.500, "3/8",  0.06),
    (1.000, "1/4",  0.38),
    (0.750, "3/16", 0.04),
    (0.500, "1/8",  0.22),
    (0.250, "1/16", 0.08),
]
_DUR_WEIGHTS = [w for *_, w in DURATIONS]
QL_TO_TOKEN  = {round(ql, 6): tok for ql, tok, _ in DURATIONS}

REST_PROB  = 0.15
CHORD_PROB = 0.10

MEASURES_MIN = 4
MEASURES_MAX = 8
BASS_CLEF_PROB = 0.20


# ─────────────────────────────────────────────────────────────────────────────
#  New symbol tables
# ─────────────────────────────────────────────────────────────────────────────

# ── Dynamics (셈여림) ─────────────────────────────────────────────────────────
DYNAMIC_MARKS = ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "fp", "sf", "sfz"]
_DYN_WEIGHTS  = [0.08, 0.09, 0.10, 0.10, 0.10, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07]
DYNAMIC_PROB  = 0.40   # probability of dynamic marking at measure start

# ── Hairpins (크레센도/디미누엔도) ────────────────────────────────────────────
HAIRPIN_PROB  = 0.25   # probability of hairpin in a measure

# ── Articulations (아티큘레이션) ──────────────────────────────────────────────
# (token_suffix, music21_class)
ARTIC_CHOICES = [
    ("staccato",       m21_artic.Staccato),
    ("accent",         m21_artic.Accent),
    ("tenuto",         m21_artic.Tenuto),
    ("marcato",        m21_artic.StrongAccent),
    ("staccatissimo",  m21_artic.Staccatissimo),
]
ARTIC_PROB = 0.20   # probability any pitched note gets an articulation

# ── Ornaments (꾸밈음) ────────────────────────────────────────────────────────
ORNAMENT_CHOICES = [
    ("trill",   m21_expr.Trill),
    ("mordent", m21_expr.Mordent),
    ("turn",    m21_expr.Turn),
]
ORNAMENT_PROB = 0.08   # probability any pitched note gets an ornament

# ── Fermata (페르마타) ────────────────────────────────────────────────────────
# Applied to the last note before a barline or the final note
FERMATA_PROB = 0.20

# ── Slur (이음줄) ─────────────────────────────────────────────────────────────
SLUR_PROB = 0.15   # probability of a slur over 2–4 consecutive notes in a measure

# ── Repeat (도돌이표) ─────────────────────────────────────────────────────────
REPEAT_PROB = 0.25   # probability the score uses a repeat section

# ── Round 2 new symbols ───────────────────────────────────────────────────────
TUPLET_PROB             = 0.20   # prob of an eighth-note triplet group in a measure
TRILL_EXT_PROB          = 0.10   # prob of a trill extension (tr~~~) span in a measure
OTTAVA_PROB             = 0.10   # prob of 8va bracket in a measure (treble only)
DOUBLE_ACCIDENTAL_PROB  = 0.08   # per-note prob of double sharp/flat
NATURAL_IN_MEASURE_PROB = 0.15   # per-note prob of forced natural (contradicts key)


# ─────────────────────────────────────────────────────────────────────────────
#  Pitch pools
# ─────────────────────────────────────────────────────────────────────────────

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


TREBLE_PITCHES = _pitch_pool("C4", "B5")
BASS_PITCHES   = _pitch_pool("C2", "B3")


# ─────────────────────────────────────────────────────────────────────────────
#  Duration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _choose_duration(max_ql: float) -> tuple:
    possible = [(ql, tok, w) for ql, tok, w in DURATIONS if ql <= max_ql + 1e-9]
    if not possible:
        ql, tok, _ = min(DURATIONS, key=lambda x: x[0])
        return ql, tok
    weights = [w for *_, w in possible]
    ql, tok, _ = random.choices(possible, weights=weights)[0]
    return min(ql, max_ql), tok


def _ql_to_token(ql: float) -> str:
    t = QL_TO_TOKEN.get(round(ql, 6))
    if t:
        return t
    f = Fraction(ql).limit_denominator(32)
    normalised = Fraction(f.numerator, f.denominator * 4)
    return f"{normalised.numerator}/{normalised.denominator}"


def _normalize_pitch_name(name: str) -> str:
    return name.replace("-", "b")


def _element_to_tokens_with_dur(el, dur_tok: str) -> list:
    """Like _element_to_tokens but uses an explicit duration token (for tuplets)."""
    if isinstance(el, Rest):
        return [f"rest-{dur_tok}"]
    if isinstance(el, Chord):
        toks = []
        for i, p in enumerate(sorted(el.pitches, key=lambda p: p.midi)):
            name = _normalize_pitch_name(p.nameWithOctave)
            toks.append(f"note-{name}-{dur_tok}" if i == 0 else f"chord-{name}")
        return toks
    if isinstance(el, Note):
        name = _normalize_pitch_name(el.pitch.nameWithOctave)
        return [f"note-{name}-{dur_tok}"]
    return []


def _element_to_tokens_sounding_up(el) -> list:
    """Like _element_to_tokens but shifts pitch token up one octave (8va sounding pitch)."""
    dur_tok = _ql_to_token(el.duration.quarterLength)
    if isinstance(el, Rest):
        return [f"rest-{dur_tok}"]
    if isinstance(el, Chord):
        toks = []
        for i, p in enumerate(sorted(el.pitches, key=lambda p: p.midi)):
            shifted = Pitch(p.midi + 12)
            name = _normalize_pitch_name(shifted.nameWithOctave)
            toks.append(f"note-{name}-{dur_tok}" if i == 0 else f"chord-{name}")
        return toks
    if isinstance(el, Note):
        shifted = Pitch(el.pitch.midi + 12)
        name = _normalize_pitch_name(shifted.nameWithOctave)
        return [f"note-{name}-{dur_tok}"]
    return []


# Sharp/flat order for key signatures
_SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
_FLAT_ORDER  = ['B', 'E', 'A', 'D', 'G', 'C', 'F']


def _natural_contradiction_pool(ks_sharps: int) -> set:
    """Return pitch class names that have accidentals in this key signature."""
    if ks_sharps > 0:
        return set(_SHARP_ORDER[:ks_sharps])
    elif ks_sharps < 0:
        return set(_FLAT_ORDER[:-ks_sharps])
    return set()


# ─────────────────────────────────────────────────────────────────────────────
#  Element factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_element(pitch_pool: list, ql: float):
    """Create a random Note / Rest / Chord."""
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
#  Token helpers
# ─────────────────────────────────────────────────────────────────────────────

def _element_to_tokens(el) -> list:
    """
    Convert a music21 element to base DeepScore tokens (note/rest/chord).
    Does NOT include articulation/ornament/fermata — those are added by the caller.
    """
    if isinstance(el, Rest):
        return [f"rest-{_ql_to_token(el.duration.quarterLength)}"]

    if isinstance(el, Chord):
        tok = _ql_to_token(el.duration.quarterLength)
        tokens = []
        for i, p in enumerate(sorted(el.pitches, key=lambda p: p.midi)):
            name = _normalize_pitch_name(p.nameWithOctave)
            tokens.append(f"note-{name}-{tok}" if i == 0 else f"chord-{name}")
        return tokens

    if isinstance(el, Note):
        tok = _ql_to_token(el.duration.quarterLength)
        name = _normalize_pitch_name(el.pitch.nameWithOctave)
        return [f"note-{name}-{tok}"]

    return []


def _add_note_modifiers(el, tokens: list) -> list:
    """
    Optionally attach articulation / ornament to el and append matching tokens.
    Returns the (possibly extended) token list.
    Only applies to pitched elements (Note / Chord).
    """
    if isinstance(el, Rest):
        return tokens

    # Articulation
    if random.random() < ARTIC_PROB:
        tok_suffix, artic_cls = random.choice(ARTIC_CHOICES)
        try:
            el.articulations.append(artic_cls())
            tokens.append(f"artic-{tok_suffix}")
        except Exception:
            pass

    # Ornament
    if random.random() < ORNAMENT_PROB:
        tok_suffix, orn_cls = random.choice(ORNAMENT_CHOICES)
        try:
            el.expressions.append(orn_cls())
            tokens.append(f"ornament-{tok_suffix}")
        except Exception:
            pass

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
#  Score builder
# ─────────────────────────────────────────────────────────────────────────────

def build_score(score_id: int) -> tuple:
    """
    Build a random music21 Score and its DeepScore token sequence.
    Returns (Score, list[str]).
    """
    ts_num, ts_den, _ = random.choices(TIME_SIGS, weights=_TS_WEIGHTS)[0]
    ks_sharps, ks_name = random.choices(KEY_SIGS,  weights=_KS_WEIGHTS)[0]
    use_bass   = random.random() < BASS_CLEF_PROB
    pitch_pool = BASS_PITCHES if use_bass else TREBLE_PITCHES
    n_measures = random.randint(MEASURES_MIN, MEASURES_MAX)
    bpm        = random.randint(60, 130)

    clef_obj = clef.BassClef() if use_bass else clef.TrebleClef()
    clef_tok = "clef-F" if use_bass else "clef-G"

    measure_ql = ts_num * (4.0 / ts_den)

    part = Part()
    part.insert(0, clef_obj)
    part.insert(0, key.KeySignature(ks_sharps))
    part.insert(0, meter.TimeSignature(f"{ts_num}/{ts_den}"))
    part.insert(0, tempo.MetronomeMark(number=bpm))

    tokens = ["<SOS>", clef_tok, f"key-{ks_name}", f"time-{ts_num}/{ts_den}"]

    # ── Repeat section decision ───────────────────────────────────────────────
    use_repeat  = random.random() < REPEAT_PROB
    repeat_start = 0
    repeat_end   = n_measures

    # ── Current score offset tracker ─────────────────────────────────────────
    current_offset = 0.0

    # Key-signature contradiction pool (for natural-in-measure generation)
    nat_pool = _natural_contradiction_pool(ks_sharps)

    for m_idx in range(n_measures):
        m = Measure(number=m_idx + 1)

        # ── Per-measure new symbol decisions ─────────────────────────────────
        do_tuplet    = random.random() < TUPLET_PROB
        do_ottava    = (random.random() < OTTAVA_PROB) and (not use_bass)
        do_trill_ext = random.random() < TRILL_EXT_PROB
        ottava_max         = random.randint(2, 4) if do_ottava else 0
        ottava_started     = False
        ottava_ended       = False
        ottava_emitted     = 0
        ottava_note_objs: list = []
        tuplet_done  = False

        # ── Repeat barlines ───────────────────────────────────────────────────
        if use_repeat and m_idx == repeat_start:
            m.leftBarline = Repeat(direction='start')
            tokens.append("barline-start-repeat")

        # ── Dynamic at measure start ──────────────────────────────────────────
        dyn_threshold = DYNAMIC_PROB if m_idx == 0 else DYNAMIC_PROB * 0.35
        if random.random() < dyn_threshold:
            dyn_name = random.choices(DYNAMIC_MARKS, weights=_DYN_WEIGHTS)[0]
            try:
                part.insert(current_offset, m21_dyn.Dynamic(dyn_name))
            except Exception:
                pass
            tokens.append(f"dynamic-{dyn_name}")

        # ── Hairpin decision ──────────────────────────────────────────────────
        do_hairpin   = random.random() < HAIRPIN_PROB
        hairpin_type = random.choice(["cresc", "dim"])
        if do_hairpin:
            tokens.append(f"hairpin-{hairpin_type}-start")

        # ── Slur decision ─────────────────────────────────────────────────────
        do_slur       = random.random() < SLUR_PROB
        slur_note_objs: list = []

        # ── Fill measure ──────────────────────────────────────────────────────
        remaining = measure_ql
        measure_elements: list = []

        while remaining > 1e-9:

            # ── Tuplet group: eighth-note triplet (3 notes in 1 quarter) ─────
            if do_tuplet and not tuplet_done and remaining >= 1.0:
                tokens.append("tuplet-3-start")
                for _ in range(3):
                    tql = round(1.0 / 3, 6)
                    tel = _make_element(pitch_pool, tql)
                    m.append(tel)
                    ttoks = _element_to_tokens_with_dur(tel, "1/8")
                    ttoks = _add_note_modifiers(tel, ttoks)
                    tokens.extend(ttoks)
                    measure_elements.append(tel)
                    if not isinstance(tel, Rest):
                        slur_note_objs.append(tel)
                tokens.append("tuplet-3-end")
                remaining = round(remaining - 1.0, 6)
                tuplet_done = True
                continue

            ql, _ = _choose_duration(remaining)
            ql    = round(min(ql, remaining), 6)
            el    = _make_element(pitch_pool, ql)
            m.append(el)

            # ── Token generation (ottava / double-accidental / normal) ────────
            if (isinstance(el, Note) and do_ottava
                    and ottava_emitted < ottava_max):
                # 8va: token encodes sounding pitch (written + 1 octave)
                if not ottava_started:
                    tokens.append("ottava-8va-start")
                    ottava_started = True
                el_tokens = _element_to_tokens_sounding_up(el)
                ottava_note_objs.append(el)
                ottava_emitted += 1

            elif (isinstance(el, Note)
                  and el.pitch.accidental is None
                  and random.random() < DOUBLE_ACCIDENTAL_PROB):
                # Double accidental: written with ## or bb, token = sounding pitch
                use_dsharp = random.random() < 0.5
                try:
                    el.pitch.accidental = Accidental(
                        'double-sharp' if use_dsharp else 'double-flat')
                except Exception:
                    pass
                sounding_name = _normalize_pitch_name(
                    Pitch(el.pitch.midi).nameWithOctave)
                el_tokens = [f"note-{sounding_name}-{_ql_to_token(ql)}"]

            else:
                # Natural-in-measure: force natural for a key-signature pitch class
                if (isinstance(el, Note) and nat_pool
                        and el.pitch.accidental is None
                        and el.pitch.name in nat_pool
                        and random.random() < NATURAL_IN_MEASURE_PROB):
                    try:
                        el.pitch.accidental = Accidental('natural')
                    except Exception:
                        pass
                el_tokens = _element_to_tokens(el)

            el_tokens = _add_note_modifiers(el, el_tokens)
            tokens.extend(el_tokens)
            measure_elements.append(el)
            if not isinstance(el, Rest):
                slur_note_objs.append(el)

            # Close ottava span when max pitched notes reached
            if (do_ottava and ottava_started and not ottava_ended
                    and ottava_emitted >= ottava_max):
                tokens.append("ottava-8va-end")
                ottava_ended = True
                if len(ottava_note_objs) >= 2:
                    try:
                        ott = m21_spanner.Ottava(type='8va')
                        ott.addSpannedElements(
                            [ottava_note_objs[0], ottava_note_objs[-1]])
                        part.insert(current_offset, ott)
                    except Exception:
                        pass

            remaining = round(remaining - ql, 6)

        # Close ottava if measure ended before ottava_max notes were generated
        if do_ottava and ottava_started and not ottava_ended:
            tokens.append("ottava-8va-end")
            if len(ottava_note_objs) >= 2:
                try:
                    ott = m21_spanner.Ottava(type='8va')
                    ott.addSpannedElements(
                        [ottava_note_objs[0], ottava_note_objs[-1]])
                    part.insert(current_offset, ott)
                except Exception:
                    pass

        # ── Fermata on last pitched note ──────────────────────────────────────
        if random.random() < FERMATA_PROB:
            for el in reversed(measure_elements):
                if not isinstance(el, Rest):
                    try:
                        el.expressions.append(m21_expr.Fermata())
                    except Exception:
                        pass
                    tokens.append("fermata")
                    break

        # ── Close hairpin ─────────────────────────────────────────────────────
        if do_hairpin:
            tokens.append(f"hairpin-{hairpin_type}-end")
            if len(slur_note_objs) >= 2:
                try:
                    if hairpin_type == "cresc":
                        hp = m21_dyn.Crescendo()
                    else:
                        hp = m21_dyn.Diminuendo()
                    hp.addSpannedElements([slur_note_objs[0], slur_note_objs[-1]])
                    part.insert(current_offset, hp)
                except Exception:
                    pass

        # ── Trill extension span ──────────────────────────────────────────────
        if do_trill_ext and len(slur_note_objs) >= 2:
            trill_len = random.randint(2, min(4, len(slur_note_objs)))
            trill_els = slur_note_objs[:trill_len]
            tokens.append("trill-start")
            tokens.append("trill-end")
            try:
                te = m21_expr.TrillExtension(trill_els[0], trill_els[-1])
                part.insert(current_offset, te)
            except Exception:
                try:
                    sp = m21_spanner.TrillExtension()
                    sp.addSpannedElements([trill_els[0], trill_els[-1]])
                    part.insert(current_offset, sp)
                except Exception:
                    pass

        # ── Slur spanner ──────────────────────────────────────────────────────
        if do_slur and len(slur_note_objs) >= 2:
            max_slur = min(4, len(slur_note_objs))
            slur_len = random.randint(2, max_slur)
            slur_els = slur_note_objs[:slur_len]
            tokens.append("slur-start")
            tokens.append("slur-end")
            try:
                sl = m21_spanner.Slur(slur_els[0], slur_els[-1])
                part.insert(current_offset, sl)
            except Exception:
                pass

        # ── Barline ───────────────────────────────────────────────────────────
        if use_repeat and m_idx == repeat_end - 1:
            m.rightBarline = Repeat(direction='end')
            tokens.append("barline-end-repeat")
        elif m_idx < n_measures - 1:
            tokens.append("barline")
        else:
            m.rightBarline = Barline("final")
            tokens.append("barline-final")

        part.append(m)
        current_offset += measure_ql

    tokens.append("<EOS>")

    score = Score()
    score.insert(0, part)
    return score, tokens


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer() -> dict:
    """
    Build the complete DeepScore vocabulary including all symbol types.
    """
    tokens = []

    # ── Special ───────────────────────────────────────────────────────────────
    tokens += ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    # ── Clefs ─────────────────────────────────────────────────────────────────
    tokens += ["clef-G", "clef-F", "clef-C"]

    # ── Key signatures ────────────────────────────────────────────────────────
    tokens += [f"key-{k}" for k in [
        "C", "G", "D", "A", "E", "B", "F#",   # sharp keys
        "F", "Bb", "Eb", "Ab", "Db", "Gb",     # flat keys
    ]]

    # ── Time signatures ───────────────────────────────────────────────────────
    tokens += [
        "time-2/4", "time-3/4", "time-4/4", "time-6/8",
        "time-3/8", "time-12/8", "time-2/2", "time-5/4",
    ]

    # ── Note / Chord / Rest ───────────────────────────────────────────────────
    dur_tokens = ["1/1", "3/4", "1/2", "3/8", "1/4", "3/16", "1/8", "3/32", "1/16", "1/32"]
    pitch_classes = [
        "C", "C#", "Db", "D", "D#", "Eb", "E",
        "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
    ]
    octaves = [2, 3, 4, 5, 6]

    for oct in octaves:
        for pc in pitch_classes:
            for dur in dur_tokens:
                tokens.append(f"note-{pc}{oct}-{dur}")

    for oct in octaves:
        for pc in pitch_classes:
            tokens.append(f"chord-{pc}{oct}")

    for dur in dur_tokens:
        tokens.append(f"rest-{dur}")

    # ── Barlines (세로줄 / 도돌이표) ──────────────────────────────────────────
    tokens += [
        "barline",
        "barline-double",
        "barline-final",
        "barline-start-repeat",
        "barline-end-repeat",
    ]

    # ── Dynamics (셈여림) ──────────────────────────────────────────────────────
    tokens += [f"dynamic-{d}" for d in DYNAMIC_MARKS]

    # ── Hairpins (크레센도 / 디미누엔도) ────────────────────────────────────────
    tokens += [
        "hairpin-cresc-start", "hairpin-cresc-end",
        "hairpin-dim-start",   "hairpin-dim-end",
    ]

    # ── Articulations (아티큘레이션) ──────────────────────────────────────────
    tokens += [f"artic-{s}" for s, _ in ARTIC_CHOICES]

    # ── Ornaments (꾸밈음) ────────────────────────────────────────────────────
    tokens += [f"ornament-{s}" for s, _ in ORNAMENT_CHOICES]

    # ── Fermata (페르마타) ────────────────────────────────────────────────────
    tokens += ["fermata"]

    # ── Slur (이음줄) ─────────────────────────────────────────────────────────
    tokens += ["slur-start", "slur-end"]

    # ── Tuplet (셋잇단음표) ────────────────────────────────────────────────────
    tokens += ["tuplet-3-start", "tuplet-3-end"]

    # ── Ottava / 8va (한 옥타브 올림/내림) ─────────────────────────────────────
    tokens += [
        "ottava-8va-start", "ottava-8va-end",
        "ottava-8vb-start", "ottava-8vb-end",
    ]

    # ── Trill extension (긴 트릴) ──────────────────────────────────────────────
    tokens += ["trill-start", "trill-end"]

    # Deduplicate while preserving insertion order
    seen, unique = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return {tok: idx for idx, tok in enumerate(unique)}


# ─────────────────────────────────────────────────────────────────────────────
#  MuseScore rendering
# ─────────────────────────────────────────────────────────────────────────────

def find_musescore(override: str = None) -> str:
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
    tmp_stem = png_path.with_name(png_path.stem + "_ms_tmp.png")
    cmd = [musescore, "-o", str(tmp_stem), "-r", str(PNG_DPI), str(xml_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] MuseScore timed out: {xml_path.name}")
        return False
    except Exception as exc:
        print(f"  [WARN] MuseScore subprocess error: {exc}")
        return False

    page1 = tmp_stem.with_name(tmp_stem.stem + "-1.png")
    if page1.exists():
        page1.rename(png_path)
        return True
    if tmp_stem.exists():
        tmp_stem.rename(png_path)
        return True

    print(f"  [WARN] PNG not found after rendering {xml_path.name}")
    if result.stderr:
        print(f"         stderr: {result.stderr[:400]}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate random OMR training data (MusicXML + PNG + label JSON)"
    )
    parser.add_argument("-n", "--count",   type=int, default=100)
    parser.add_argument("-o", "--output",  type=str, default="data/Round2")
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--musescore",     type=str, default=None)
    parser.add_argument("--no-png",        action="store_true")
    parser.add_argument("--tokenizer-out", type=str, default="data/tokenizer.json")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = Path(args.tokenizer_out)
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    vocab = build_tokenizer()
    tok_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Tokenizer saved → {tok_path}  ({len(vocab)} tokens)")

    # ── MuseScore ─────────────────────────────────────────────────────────────
    musescore_path = None
    if not args.no_png:
        musescore_path = find_musescore(args.musescore)
        if musescore_path:
            print(f"MuseScore found → {musescore_path}")
        else:
            print("WARNING: MuseScore not found — PNG rendering will be skipped.")

    # ── Generate loop ─────────────────────────────────────────────────────────
    indices  = range(1, args.count + 1)
    iterator = _tqdm(indices, desc="Generating", unit="score") if HAS_TQDM else indices

    ok_xml = ok_png = 0

    for i in iterator:
        stem     = f"num{i}"
        xml_path = out_dir / f"{stem}.musicxml"
        png_path = out_dir / f"{stem}.png"
        lbl_path = out_dir / f"{stem}.json"

        try:
            score, tokens = build_score(i)
            score.write("musicxml", fp=str(xml_path))
            ok_xml += 1
        except Exception as exc:
            print(f"  [ERROR] {stem}: {exc}")
            continue

        lbl_path.write_text(
            json.dumps({"id": stem, "tokens": tokens}, ensure_ascii=False),
            encoding="utf-8",
        )

        if musescore_path:
            ok = render_png(musescore_path, xml_path, png_path)
            if ok:
                ok_png += 1
            elif not HAS_TQDM:
                print(f"  [{i}/{args.count}] {stem}: XML OK  PNG FAIL")

    print(f"\n{'-'*50}")
    print(f"Generated  : {ok_xml}/{args.count} MusicXML files")
    if not args.no_png:
        print(f"Rendered   : {ok_png}/{ok_xml} PNG files" if musescore_path
              else "Rendered   : 0  (MuseScore not available)")
    print(f"Labels     : {ok_xml} JSON label files")
    print(f"Output dir : {out_dir.resolve()}")
    print(f"Tokenizer  : {tok_path.resolve()}  ({len(vocab)} tokens)")


if __name__ == "__main__":
    main()
