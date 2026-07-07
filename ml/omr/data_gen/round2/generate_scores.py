"""
round2train/generate_scores.py  –  Round 2 데이터 생성

Round 1 기호 포함:
  clef-G/F, key-C/G/D/F/Bb, time-4/4·3/4·2/4
  note/rest/chord, barline/barline-final
  barline-start-repeat/barline-end-repeat

Round 2 추가 기호:
  dynamic-pp/p/mp/mf/f/ff/fp
  hairpin-cresc-start/end, hairpin-dim-start/end
  artic-staccato/accent/tenuto/marcato
  ornament-trill/mordent/turn
  fermata
  slur-start/end
  tuplet-3-start/end  (3연음: 3음이 1박 채움)
  ottava-8va-start/end, ottava-8vb-start/end

MEASURES_MAX=4: 단일 행에 맞도록 제한

사용법:
    python round2train/generate_scores.py ^
        --count 4000 ^
        --output round2train/Round2 ^
        --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"

    python round2train/generate_scores.py ^
        --count 300 --start-idx 4001 ^
        --output round2train/Round2_test ^
        --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
"""

import argparse
import json
import random
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from music21 import (
    articulations, bar, chord as m21chord, clef, dynamics,
    expressions, key, meter, note as m21note, spanner, stream
)
from music21.note import Note, Rest
from music21.pitch import Pitch
from music21.stream import Measure, Part, Score
from music21.chord import Chord

_MUSESCORE_CANDIDATES = [
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    "/usr/bin/musescore4", "/usr/bin/musescore3", "/usr/bin/musescore",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
]

def find_musescore(override=None):
    if override and Path(override).exists():
        return str(Path(override))
    for c in _MUSESCORE_CANDIDATES:
        if Path(c).exists():
            return c
    for cmd in ["musescore4", "musescore3", "musescore", "mscore4", "mscore3"]:
        found = shutil.which(cmd)
        if found:
            return found
    return None

def render_png(musescore, xml_path, png_path):
    tmp = png_path.with_name(png_path.stem + "_ms_tmp.png")
    try:
        subprocess.run([musescore, "-o", str(tmp), "-r", "150", str(xml_path)],
                       capture_output=True, text=True, timeout=90)
    except Exception:
        return False
    page1 = tmp.with_name(tmp.stem + "-1.png")
    if page1.exists():
        page1.rename(png_path); return True
    if tmp.exists():
        tmp.rename(png_path); return True
    return False


# ─── 설정 ─────────────────────────────────────────────────────────────────────

KEY_SIGS   = [(0,'C'),(1,'G'),(-1,'F'),(2,'D'),(-2,'Bb')]
KS_WEIGHTS = [0.30, 0.20, 0.20, 0.15, 0.15]

TIME_SIGS  = [(4,4),(3,4),(2,4)]
TS_WEIGHTS = [0.50, 0.30, 0.20]

DURATIONS  = [
    (4.0,'1/1',0.03),(2.0,'1/2',0.12),(1.5,'3/8',0.04),
    (1.0,'1/4',0.42),(0.5,'1/8',0.28),(0.25,'1/16',0.09),(0.75,'3/16',0.02),
]

DYNAMICS_LIST = ['pp','p','mp','mf','f','ff','fp']
ARTICS        = ['staccato','accent','tenuto','marcato']
ORNAMENTS     = ['trill','mordent','turn']

REST_PROB     = 0.15
CHORD_PROB    = 0.15
DYNAMIC_PROB  = 0.40
HAIRPIN_PROB  = 0.25
ARTIC_PROB    = 0.20
ORNAMENT_PROB = 0.06
FERMATA_PROB  = 0.04
SLUR_PROB     = 0.15
TUPLET_PROB   = 0.15
OTTAVA_PROB   = 0.10
REPEAT_PROB   = 0.15
BASS_PROB     = 0.20


def _pitch_pool(lo, hi):
    lo_m = Pitch(lo).midi; hi_m = Pitch(hi).midi
    pcs  = ["C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"]
    pool = []
    for oct in range(2, 7):
        for pc in pcs:
            try:
                p = Pitch(f"{pc}{oct}")
                if lo_m <= p.midi <= hi_m:
                    pool.append(p.nameWithOctave)
            except Exception:
                pass
    return pool


TREBLE_PITCHES = _pitch_pool("C4", "B5")
BASS_PITCHES   = _pitch_pool("C2", "B3")


def _np(name: str) -> str:
    return name.replace('-', 'b')


def _choose_dur(max_ql: float):
    possible = [(ql, tok, w) for ql, tok, w in DURATIONS if ql <= max_ql + 1e-9]
    if not possible:
        return 0.25, '1/16'
    weights = [w for *_, w in possible]
    ql, tok, _ = random.choices(possible, weights=weights)[0]
    return round(min(ql, max_ql), 6), tok


def _dur_tok(ql: float) -> str:
    for q, t, _ in DURATIONS:
        if abs(q - ql) < 1e-6:
            return t
    f = Fraction(ql).limit_denominator(32)
    n = Fraction(f.numerator, f.denominator * 4)
    return f"{n.numerator}/{n.denominator}"


def _add_artic(el, name: str):
    amap = {
        'staccato':  articulations.Staccato(),
        'accent':    articulations.Accent(),
        'tenuto':    articulations.Tenuto(),
        'marcato':   articulations.StrongAccent(),
    }
    obj = amap.get(name)
    if obj is None:
        return
    targets = list(el.notes) if isinstance(el, Chord) else [el]
    for t in targets:
        t.articulations = [obj]


def _add_ornament(el, name: str):
    omap = {
        'trill':   expressions.Trill(),
        'mordent': expressions.Mordent(),
        'turn':    expressions.Turn(),
    }
    obj = omap.get(name)
    if obj is None:
        return
    try:
        el.expressions.append(obj)
    except Exception:
        el.expressions = [obj]


def build_score_r2(score_id: int) -> tuple:
    ts_num, ts_den = random.choices(TIME_SIGS, weights=TS_WEIGHTS)[0]
    ks_sharps, ks_name = random.choices(KEY_SIGS, weights=KS_WEIGHTS)[0]
    use_bass  = random.random() < BASS_PROB
    pitch_pool = BASS_PITCHES if use_bass else TREBLE_PITCHES
    n_measures = random.randint(2, 4)
    measure_ql = ts_num * (4.0 / ts_den)

    clef_tok = 'clef-F' if use_bass else 'clef-G'
    tokens   = ['<SOS>', clef_tok, f'key-{ks_name}', f'time-{ts_num}/{ts_den}']

    part = Part()
    part.insert(0, clef.BassClef() if use_bass else clef.TrebleClef())
    part.insert(0, key.KeySignature(ks_sharps))
    part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    # ── 스팬 사전 결정 ─────────────────────────────────────────────────────────
    use_hairpin  = random.random() < HAIRPIN_PROB
    hp_type      = random.choice(['cresc', 'dim'])
    hp_start_m   = random.randint(0, max(0, n_measures - 2)) if use_hairpin else -1
    hp_end_m     = min(hp_start_m + random.randint(1, 2), n_measures - 1) if use_hairpin else -1

    use_repeat   = random.random() < REPEAT_PROB and n_measures >= 2
    rp_start_m   = random.randint(0, n_measures // 2) if use_repeat else -1
    rp_end_m     = random.randint(max(rp_start_m, 1), n_measures - 1) if use_repeat else -1

    use_ottava   = (not use_bass) and random.random() < OTTAVA_PROB
    ott_type     = random.choice(['8va', '8vb'])
    ott_start_m  = random.randint(0, max(0, n_measures - 2)) if use_ottava else -1
    ott_end_m    = min(ott_start_m + 1, n_measures - 1) if use_ottava else -1

    hp_notes  = []
    ott_notes = []

    for m_idx in range(n_measures):
        m = Measure(number=m_idx + 1)

        # ── 반복기호 시작 ────────────────────────────────────────────────────
        if use_repeat and m_idx == rp_start_m:
            m.leftBarline = bar.Barline('start-repeat')
            tokens.append('barline-start-repeat')

        # ── 다이나믹 ─────────────────────────────────────────────────────────
        if random.random() < DYNAMIC_PROB:
            dyn = random.choice(DYNAMICS_LIST)
            m.insert(0, dynamics.Dynamic(dyn))
            tokens.append(f'dynamic-{dyn}')

        # ── 헤어핀 시작 ──────────────────────────────────────────────────────
        if use_hairpin and m_idx == hp_start_m:
            tokens.append(f'hairpin-{hp_type}-start')

        # ── 오타바 시작 ──────────────────────────────────────────────────────
        if use_ottava and m_idx == ott_start_m:
            tokens.append(f'ottava-{ott_type}-start')

        # ── 슬러 결정 (마디 단위) ─────────────────────────────────────────────
        use_slur  = random.random() < SLUR_PROB
        slur_ns   = []
        slur_open = False

        # ── 3연음 결정 (마디 단위) ────────────────────────────────────────────
        use_tuplet = random.random() < TUPLET_PROB
        tuplet_done = False

        # ── 음표 채우기 ──────────────────────────────────────────────────────
        remaining = measure_ql

        while remaining > 1e-9:
            # 3연음 그룹 삽입
            if use_tuplet and not tuplet_done and remaining >= 1.0 - 1e-9:
                from music21 import duration as dur_mod
                tokens.append('tuplet-3-start')
                for _ in range(3):
                    p = random.choice(pitch_pool)
                    n_obj = Note(p)
                    n_obj.duration.quarterLength = 1.0 / 3.0
                    n_obj.duration.appendTuplet(dur_mod.Tuplet(3, 2))
                    m.append(n_obj)
                    tokens.append(f"note-{_np(n_obj.pitch.nameWithOctave)}-1/8")
                    if hp_start_m <= m_idx <= hp_end_m:
                        hp_notes.append(n_obj)
                    if use_ottava and ott_start_m <= m_idx <= ott_end_m:
                        ott_notes.append(n_obj)
                tokens.append('tuplet-3-end')
                remaining -= 1.0
                tuplet_done = True
                continue

            ql, dtok = _choose_dur(remaining)
            r = random.random()

            if r < REST_PROB:
                el = Rest(quarterLength=ql)
                m.append(el)
                tokens.append(f'rest-{dtok}')
                remaining -= ql
                continue

            if r < REST_PROB + CHORD_PROB:
                n_notes = random.randint(2, 3)
                chosen  = sorted(random.sample(pitch_pool, min(n_notes, len(pitch_pool))),
                                 key=lambda p: Pitch(p).midi)
                el = Chord(chosen, quarterLength=ql)
                el_toks = [f"note-{_np(chosen[0])}-{dtok}"]
                el_toks += [f"chord-{_np(p)}" for p in chosen[1:]]
            else:
                p  = random.choice(pitch_pool)
                el = Note(p, quarterLength=ql)
                el_toks = [f"note-{_np(el.pitch.nameWithOctave)}-{dtok}"]

            # 슬러 시작
            if use_slur and not slur_open and remaining > ql + 1e-9:
                tokens.append('slur-start')
                slur_open = True

            tokens.extend(el_toks)
            m.append(el)

            # 아티큘레이션
            if random.random() < ARTIC_PROB:
                artic = random.choice(ARTICS)
                _add_artic(el, artic)
                tokens.append(f'artic-{artic}')

            # 꾸밈음
            if random.random() < ORNAMENT_PROB:
                orn = random.choice(ORNAMENTS)
                _add_ornament(el, orn)
                tokens.append(f'ornament-{orn}')

            # 페르마타 (마디 마지막 음)
            if random.random() < FERMATA_PROB and remaining <= ql + 1e-9:
                try:
                    el.expressions.append(expressions.Fermata())
                except Exception:
                    el.expressions = [expressions.Fermata()]
                tokens.append('fermata')

            # 스패너 음표 등록
            if hp_start_m <= m_idx <= hp_end_m:
                hp_notes.append(el)
            if use_ottava and ott_start_m <= m_idx <= ott_end_m:
                ott_notes.append(el)
            if slur_open:
                slur_ns.append(el)

            remaining -= ql

        # ── 슬러 닫기 ────────────────────────────────────────────────────────
        if slur_open and len(slur_ns) >= 2:
            tokens.append('slur-end')
            try:
                sl = spanner.Slur()
                sl.addSpannedElements([slur_ns[0], slur_ns[-1]])
                part.insert(0, sl)
            except Exception:
                pass

        # ── 헤어핀 끝 ────────────────────────────────────────────────────────
        if use_hairpin and m_idx == hp_end_m:
            tokens.append(f'hairpin-{hp_type}-end')

        # ── 오타바 끝 ────────────────────────────────────────────────────────
        if use_ottava and m_idx == ott_end_m:
            tokens.append(f'ottava-{ott_type}-end')

        # ── 마디선 ──────────────────────────────────────────────────────────
        if use_repeat and m_idx == rp_end_m:
            m.rightBarline = bar.Barline('end-repeat')
            tokens.append('barline-end-repeat')
        elif m_idx < n_measures - 1:
            tokens.append('barline')
        else:
            m.rightBarline = bar.Barline('final')
            tokens.append('barline-final')

        part.append(m)

    # ── 헤어핀 스패너 연결 ─────────────────────────────────────────────────────
    if len(hp_notes) >= 2:
        try:
            hp_obj = dynamics.Crescendo() if hp_type == 'cresc' else dynamics.Diminuendo()
            hp_obj.addSpannedElements([hp_notes[0], hp_notes[-1]])
            part.insert(0, hp_obj)
        except Exception:
            pass

    # ── 오타바 스패너 연결 ─────────────────────────────────────────────────────
    if len(ott_notes) >= 2:
        try:
            ott_obj = spanner.Ottava(type=ott_type)
            ott_obj.addSpannedElements([ott_notes[0], ott_notes[-1]])
            part.insert(0, ott_obj)
        except Exception:
            pass

    tokens.append('<EOS>')
    score = Score()
    score.insert(0, part)
    return score, tokens


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Round 2 데이터 생성')
    p.add_argument('--count',     type=int, default=4000)
    p.add_argument('--output',    default='round2train/Round2')
    p.add_argument('--seed',      type=int, default=None)
    p.add_argument('--musescore', default=None)
    p.add_argument('--no-png',    action='store_true')
    p.add_argument('--start-idx', type=int, default=1)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    musescore_path = None
    if not args.no_png:
        musescore_path = find_musescore(args.musescore)
        if musescore_path:
            print(f"MuseScore: {musescore_path}")
        else:
            print("WARNING: MuseScore not found — XML+JSON만 생성")

    print(f"Round 2 생성: {args.count}개 → {out_dir.resolve()}")
    print(f"추가 기호: dynamics, hairpin, artic, ornament, slur, tuplet, ottava, repeat")

    try:
        from tqdm import tqdm as _tqdm
        _iter = lambda x: _tqdm(x, desc="Round2", unit="score")
    except ImportError:
        _iter = lambda x: x

    ok_xml = ok_png = 0
    for i in _iter(range(args.start_idx, args.start_idx + args.count)):
        stem     = f"num{i}"
        xml_path = out_dir / f"{stem}.musicxml"
        png_path = out_dir / f"{stem}.png"
        lbl_path = out_dir / f"{stem}.json"

        try:
            score, tokens = build_score_r2(i)
        except Exception as exc:
            print(f"  [ERROR] {stem}: {exc}")
            continue

        try:
            score.write("musicxml", fp=str(xml_path))
            ok_xml += 1
        except Exception as exc:
            print(f"  [ERROR] {stem} XML: {exc}")
            continue

        lbl_path.write_text(
            json.dumps({"id": stem, "tokens": tokens}, ensure_ascii=False),
            encoding='utf-8'
        )

        if musescore_path:
            if render_png(musescore_path, xml_path, png_path):
                ok_png += 1

    print(f"\nRound 2 완료: XML={ok_xml}/{args.count}, PNG={ok_png}/{ok_xml}")
    print(f"출력: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
