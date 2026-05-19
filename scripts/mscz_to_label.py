#!/usr/bin/env python3
"""
mscz_to_label.py — .mscz / .musicxml / .mscx 파일을 JSON 토큰 레이블로 변환.

generate_dataset.py 와 동일한 토큰 규칙을 사용합니다.
Round 5 도메인 적응용 실제 촬영 이미지의 정답 레이블 생성에 사용합니다.

Usage:
    # 폴더 일괄 변환
    python scripts/mscz_to_label.py --input-dir data/test --output-dir data/test

    # 단일 파일
    python scripts/mscz_to_label.py --input data/test/chopin_waltz.mscz

    # 파트 인덱스 지정 (0=RH, 1=LH)
    python scripts/mscz_to_label.py --input-dir data/test --part 0
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

try:
    import music21 as m21
    from music21 import articulations as m21_artic
    from music21 import bar, clef, dynamics as m21_dyn, expressions as m21_expr
    from music21 import key, meter
    from music21.note import Note, Rest
    from music21.chord import Chord
    from music21.pitch import Pitch
    from music21.stream import Measure, Part, Score
except ImportError:
    sys.exit("ERROR: music21 not installed.\n  Run: pip install music21")


# ─────────────────────────────────────────────────────────────────────────────
#  MuseScore 경로
# ─────────────────────────────────────────────────────────────────────────────

MUSESCORE_CANDIDATES = [
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
    "/usr/bin/mscore4", "/usr/bin/mscore3", "/usr/bin/mscore",
    "/usr/bin/musescore4", "/usr/bin/musescore3",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
]


def find_musescore() -> str | None:
    for p in MUSESCORE_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Duration 매핑 (generate_dataset.py 와 동일)
# ─────────────────────────────────────────────────────────────────────────────

QL_TO_TOKEN = {
    4.000: "1/1",
    3.000: "3/4",
    2.000: "1/2",
    1.500: "3/8",
    1.000: "1/4",
    0.750: "3/16",
    0.500: "1/8",
    0.250: "1/16",
}


def ql_to_token(ql: float) -> str:
    key_v = round(ql, 6)
    if key_v in QL_TO_TOKEN:
        return QL_TO_TOKEN[key_v]
    # 표준 목록에 없는 경우 Fraction 근사
    f = Fraction(ql).limit_denominator(32)
    norm = Fraction(f.numerator, f.denominator * 4)
    return f"{norm.numerator}/{norm.denominator}"


# ─────────────────────────────────────────────────────────────────────────────
#  조성 매핑
# ─────────────────────────────────────────────────────────────────────────────

SHARPS_TO_KEY = {
    0: "C",   1: "G",   2: "D",   3: "A",   4: "E",   5: "B",   6: "F#",  7: "C#",
    -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb", -7: "Cb",
}


# ─────────────────────────────────────────────────────────────────────────────
#  피치 이름 정규화
# ─────────────────────────────────────────────────────────────────────────────

def normalize_pitch(name: str) -> str:
    """music21 피치명 → 토큰 형식. 예: 'E-4' → 'Eb4'"""
    return name.replace("-", "b")


# ─────────────────────────────────────────────────────────────────────────────
#  아티큘레이션 / 꾸밈음 토큰
# ─────────────────────────────────────────────────────────────────────────────

def artic_token(a) -> str | None:
    name = type(a).__name__.lower()
    if "staccatissimo" in name:  return "artic-staccatissimo"
    if "staccato"      in name:  return "artic-staccato"
    if "strongaccent"  in name:  return "artic-marcato"
    if "marcato"       in name:  return "artic-marcato"
    if "accent"        in name:  return "artic-accent"
    if "tenuto"        in name:  return "artic-tenuto"
    return None


def expr_token(e) -> str | None:
    name = type(e).__name__.lower()
    if "fermata" in name:  return "fermata"
    if "trill"   in name:  return "ornament-trill"
    if "mordent" in name:  return "ornament-mordent"
    if "turn"    in name:  return "ornament-turn"
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  mscz → MusicXML 변환 (MuseScore CLI)
# ─────────────────────────────────────────────────────────────────────────────

def mscz_to_musicxml(mscz_path: str, musescore_exe: str) -> str | None:
    tmp = tempfile.mktemp(suffix=".musicxml")
    try:
        r = subprocess.run(
            [musescore_exe, "-o", tmp, mscz_path],
            capture_output=True, timeout=90,
        )
        if r.returncode == 0 and os.path.isfile(tmp):
            return tmp
        stderr = r.stderr.decode(errors="replace").strip()
        print(f"    MuseScore stderr: {stderr[:200]}" if stderr else "")
        return None
    except Exception as exc:
        print(f"    MuseScore error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  토큰 추출 (music21 Score → list[str])
# ─────────────────────────────────────────────────────────────────────────────

def extract_tokens(score: Score, part_index: int = 0) -> list[str]:
    tokens: list[str] = ["<SOS>"]

    parts = list(score.parts)
    if not parts:
        return tokens + ["<EOS>"]

    part = parts[min(part_index, len(parts) - 1)]
    part = part.stripTies()   # 이음줄로 연결된 음표 합치기

    measures = list(part.getElementsByClass(Measure))
    if not measures:
        return tokens + ["<EOS>"]

    # ── 헤더 토큰 (첫 마디에서 읽기) ────────────────────────────────────────
    m0 = measures[0]

    clef_obj = m0.recurse().getElementsByClass(clef.Clef).first()
    if clef_obj is None:
        clef_obj = part.recurse().getElementsByClass(clef.Clef).first()
    if isinstance(clef_obj, clef.BassClef):
        tokens.append("clef-F")
    else:
        tokens.append("clef-G")

    ks = m0.recurse().getElementsByClass(key.KeySignature).first()
    if ks is None:
        ks = part.recurse().getElementsByClass(key.KeySignature).first()
    if ks is not None:
        key_name = SHARPS_TO_KEY.get(ks.sharps, f"C")   # 범위 초과 시 C 대체
        tokens.append(f"key-{key_name}")
    else:
        tokens.append("key-C")

    ts = m0.recurse().getElementsByClass(meter.TimeSignature).first()
    if ts is None:
        ts = part.recurse().getElementsByClass(meter.TimeSignature).first()
    if ts is not None:
        tokens.append(f"time-{ts.numerator}/{ts.denominator}")
    else:
        tokens.append("time-4/4")

    # ── 마디별 처리 ──────────────────────────────────────────────────────────
    for measure in measures:
        # 반복 시작 바라인
        lb = measure.leftBarline
        if isinstance(lb, bar.Repeat) and lb.direction == "start":
            tokens.append("barline-start-repeat")

        # 마디 레벨 다이나믹
        for dyn in measure.getElementsByClass(m21_dyn.Dynamic):
            tokens.append(f"dynamic-{dyn.value}")

        # 같은 오프셋의 음표를 화음으로 묶기 위해 오프셋별 그룹화
        offset_map: dict[float, list] = defaultdict(list)
        for el in measure.flatten().notesAndRests:
            # 꾸밈음 제외
            if hasattr(el.duration, "isGrace") and el.duration.isGrace:
                continue
            offset_map[round(float(el.offset), 6)].append(el)

        for offset in sorted(offset_map):
            group = offset_map[offset]

            # 그룹이 Rest 하나만 → rest 토큰
            if len(group) == 1 and isinstance(group[0], Rest):
                el = group[0]
                tokens.append(f"rest-{ql_to_token(float(el.duration.quarterLength))}")
                continue

            # 피치 있는 요소만 수집 (같은 오프셋 = 화음으로 취급)
            pitches_ql: list[tuple] = []   # (midi, pitch_name, ql)
            has_artic:  list = []
            has_expr:   list = []

            for el in group:
                if isinstance(el, Rest):
                    # 다른 음표와 동시에 등장하는 쉼표는 화음 내에서 무시
                    continue
                if isinstance(el, Chord):
                    ql = float(el.duration.quarterLength)
                    for p in sorted(el.pitches, key=lambda x: x.midi):
                        pitches_ql.append((p.midi, normalize_pitch(p.nameWithOctave), ql))
                    has_artic.extend(el.articulations)
                    has_expr.extend(el.expressions)
                elif isinstance(el, Note):
                    ql = float(el.duration.quarterLength)
                    pitches_ql.append((el.pitch.midi, normalize_pitch(el.pitch.nameWithOctave), ql))
                    has_artic.extend(el.articulations)
                    has_expr.extend(el.expressions)

            if not pitches_ql:
                continue

            # 중복 피치 제거 (두 성부에서 같은 음이 겹칠 경우)
            seen_midi: set[int] = set()
            deduped = []
            for midi, name, ql in sorted(pitches_ql, key=lambda x: x[0]):
                if midi not in seen_midi:
                    seen_midi.add(midi)
                    deduped.append((midi, name, ql))

            # 첫 번째 음에 duration, 나머지는 chord- 접두사
            first_midi, first_name, first_ql = deduped[0]
            dur_tok = ql_to_token(first_ql)
            tokens.append(f"note-{first_name}-{dur_tok}")
            for _, name, _ in deduped[1:]:
                tokens.append(f"chord-{name}")

            # 아티큘레이션 (note 뒤에)
            seen_artic: set[str] = set()
            for a in has_artic:
                tok = artic_token(a)
                if tok and tok not in seen_artic:
                    tokens.append(tok)
                    seen_artic.add(tok)

            # 꾸밈음 / 페르마타 (아티큘레이션 뒤에)
            seen_expr: set[str] = set()
            for e in has_expr:
                tok = expr_token(e)
                if tok and tok not in seen_expr:
                    tokens.append(tok)
                    seen_expr.add(tok)

        # 마디 종료 바라인
        rb = measure.rightBarline
        if isinstance(rb, bar.Repeat) and rb.direction == "end":
            tokens.append("barline-end-repeat")
        elif rb is not None and rb.type in ("final", "light-heavy", "double"):
            tokens.append("barline-final")
        else:
            tokens.append("barline")

    # 마지막 barline → barline-final 로 변경
    if tokens[-1] == "barline":
        tokens[-1] = "barline-final"
    elif tokens[-1] not in ("barline-final", "barline-end-repeat", "<EOS>"):
        tokens.append("barline-final")

    tokens.append("<EOS>")
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
#  단일 파일 처리
# ─────────────────────────────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str,
                 musescore_exe: str, part_index: int) -> bool:
    ext = Path(input_path).suffix.lower()
    tmp_xml = None

    if ext in (".mscz", ".mscx"):
        print(f"  MuseScore 변환 중 ... ", end="", flush=True)
        tmp_xml = mscz_to_musicxml(input_path, musescore_exe)
        if not tmp_xml:
            print("실패 (MuseScore 변환 오류)")
            return False
        print("완료")
        xml_path = tmp_xml
    elif ext in (".musicxml", ".xml"):
        xml_path = input_path
    else:
        print(f"  건너뜀 (지원하지 않는 확장자: {ext})")
        return False

    try:
        print(f"  music21 파싱 중 ... ", end="", flush=True)
        score = m21.converter.parse(xml_path)
        tokens = extract_tokens(score, part_index)
        note_count = sum(1 for t in tokens if t.startswith("note-"))
        print(f"완료 (토큰 {len(tokens)}개, 음표 {note_count}개)")
    except Exception as exc:
        print(f"실패 ({exc})")
        return False
    finally:
        if tmp_xml and os.path.isfile(tmp_xml):
            os.unlink(tmp_xml)

    stem = Path(input_path).stem
    out_data = {"id": stem, "tokens": tokens}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    print(f"  저장 완료 → {output_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=".mscz/.musicxml 파일을 JSON 토큰 레이블로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir",  default=None,
                   help=".mscz 파일이 있는 폴더")
    p.add_argument("--output-dir", default=None,
                   help="JSON 출력 폴더 (기본: input-dir 과 동일)")
    p.add_argument("--input",      default=None,
                   help="단일 .mscz 파일 경로")
    p.add_argument("--output",     default=None,
                   help="단일 .json 출력 경로")
    p.add_argument("--musescore",  default=None,
                   help="MuseScore 실행 파일 경로")
    p.add_argument("--part",       type=int, default=0,
                   help="악보 파트 인덱스 (0=첫번째 파트/RH, 1=LH, 기본: 0)")
    args = p.parse_args()

    musescore = args.musescore or find_musescore()
    if not musescore:
        sys.exit("ERROR: MuseScore를 찾을 수 없습니다. --musescore 옵션으로 경로를 지정하세요.")
    print(f"MuseScore : {musescore}\n")

    if args.input:
        out = args.output or str(Path(args.input).with_suffix(".json"))
        success = process_file(args.input, out, musescore, args.part)
        sys.exit(0 if success else 1)

    if args.input_dir:
        in_dir  = args.input_dir
        out_dir = args.output_dir or in_dir
        os.makedirs(out_dir, exist_ok=True)

        exts = {".mscz", ".mscx", ".musicxml", ".xml"}
        files = sorted(f for f in os.listdir(in_dir)
                       if Path(f).suffix.lower() in exts)
        if not files:
            sys.exit(f"ERROR: {in_dir} 에서 .mscz/.musicxml 파일을 찾을 수 없습니다.")

        ok, fail = 0, 0
        for fname in files:
            stem     = Path(fname).stem
            in_path  = os.path.join(in_dir,  fname)
            out_path = os.path.join(out_dir, stem + ".json")
            print(f"[{fname}]")
            if process_file(in_path, out_path, musescore, args.part):
                ok += 1
            else:
                fail += 1
            print()

        print(f"완료: {ok}개 성공, {fail}개 실패")
        sys.exit(0 if fail == 0 else 1)

    p.print_help()


if __name__ == "__main__":
    main()
