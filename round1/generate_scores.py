"""
round1train/generate_scores.py  –  Round 1 디지털 악보 데이터 생성

Round 1 기호:
  clef-G / clef-F (bass 20%)
  key-C/G/D/F/Bb
  time-4/4 · 3/4 · 2/4
  note / rest  (음가: 1/1·1/2·3/8·1/4·1/8·1/16)
  barline / barline-final
  barline-start-repeat / barline-end-repeat  (15% 확률)

※ chord, dynamics, articulation 등은 Round 2부터 추가됨

사용법:
    python round1train/generate_scores.py ^
        --count 4000 ^
        --output round1train/Round1 ^
        --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"

    python round1train/generate_scores.py ^
        --count 300 --start-idx 9001 ^
        --output round1train/Round1_test ^
        --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
"""

import argparse
import json
import random
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from music21 import bar, clef, key, meter
from music21.note import Note, Rest
from music21.pitch import Pitch
from music21.stream import Measure, Part, Score

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
    (4.0,'1/1',0.04),(2.0,'1/2',0.14),(1.5,'3/8',0.05),
    (1.0,'1/4',0.42),(0.5,'1/8',0.27),(0.25,'1/16',0.08),
]

REST_PROB   = 0.15
REPEAT_PROB = 0.15
BASS_PROB   = 0.20


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


def build_score_r1(score_id: int) -> tuple:
    ts_num, ts_den = random.choices(TIME_SIGS, weights=TS_WEIGHTS)[0]
    ks_sharps, ks_name = random.choices(KEY_SIGS, weights=KS_WEIGHTS)[0]
    use_bass   = random.random() < BASS_PROB
    pitch_pool = BASS_PITCHES if use_bass else TREBLE_PITCHES
    n_measures = random.randint(4, 8)
    measure_ql = ts_num * (4.0 / ts_den)

    clef_tok = 'clef-F' if use_bass else 'clef-G'
    tokens   = ['<SOS>', clef_tok, f'key-{ks_name}', f'time-{ts_num}/{ts_den}']

    part = Part()
    part.insert(0, clef.BassClef() if use_bass else clef.TrebleClef())
    part.insert(0, key.KeySignature(ks_sharps))
    part.insert(0, meter.TimeSignature(f'{ts_num}/{ts_den}'))

    # 반복기호 결정
    use_repeat = random.random() < REPEAT_PROB and n_measures >= 3
    rp_start_m = random.randint(0, n_measures // 2) if use_repeat else -1
    rp_end_m   = random.randint(max(rp_start_m + 1, 1), n_measures - 1) if use_repeat else -1

    for m_idx in range(n_measures):
        m = Measure(number=m_idx + 1)

        if use_repeat and m_idx == rp_start_m:
            m.leftBarline = bar.Barline('start-repeat')
            tokens.append('barline-start-repeat')

        remaining = measure_ql
        while remaining > 1e-9:
            ql, dtok = _choose_dur(remaining)
            if random.random() < REST_PROB:
                m.append(Rest(quarterLength=ql))
                tokens.append(f'rest-{dtok}')
            else:
                p   = random.choice(pitch_pool)
                n   = Note(p, quarterLength=ql)
                m.append(n)
                tokens.append(f"note-{_np(n.pitch.nameWithOctave)}-{dtok}")
            remaining -= ql

        if use_repeat and m_idx == rp_end_m:
            m.rightBarline = bar.Barline('end-repeat')
            tokens.append('barline-end-repeat')
        elif m_idx < n_measures - 1:
            tokens.append('barline')
        else:
            m.rightBarline = bar.Barline('final')
            tokens.append('barline-final')

        part.append(m)

    tokens.append('<EOS>')
    score = Score()
    score.insert(0, part)
    return score, tokens


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Round 1 데이터 생성')
    p.add_argument('--count',     type=int, default=4000)
    p.add_argument('--output',    default='round1train/Round1')
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

    print(f"Round 1 생성: {args.count}개 → {out_dir.resolve()}")
    print(f"기호: clef-G/F, key 5종, time 3종, note/rest, repeat barline")

    try:
        from tqdm import tqdm as _tqdm
        _iter = lambda x: _tqdm(x, desc="Round1", unit="score")
    except ImportError:
        _iter = lambda x: x

    ok_xml = ok_png = 0
    for i in _iter(range(args.start_idx, args.start_idx + args.count)):
        stem     = f"num{i}"
        xml_path = out_dir / f"{stem}.musicxml"
        png_path = out_dir / f"{stem}.png"
        lbl_path = out_dir / f"{stem}.json"

        try:
            score, tokens = build_score_r1(i)
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

    print(f"\nRound 1 완료: XML={ok_xml}/{args.count}, PNG={ok_png}/{ok_xml}")
    print(f"출력: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
