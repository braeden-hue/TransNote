"""
실사 촬영용으로 손으로 만든 .mscz 정답 악보를 tokenizer258.json 규격의 토큰 JSON으로
변환한다. generate_scores.py는 스스로 악보를 만들면서 토큰을 함께 뽑지만, 이건 반대로
"이미 존재하는" 악보(MuseScore CLI로 musicxml 변환 후 music21로 파싱)에서 같은 토큰
규격을 역산하는 변환기다.

사용법:
    python mscz_to_tokens.py <mscz_dir_or_file> [--musescore <path>] [--out <json_path>]

    <mscz_dir_or_file>가 디렉토리면 그 안의 첫 *.mscz 파일 하나를 변환해서
    같은 디렉토리에 <디렉토리이름>.json으로 저장한다(파일이 이미 있으면 skip).
"""
import argparse
import json
import subprocess
import sys
import warnings as _warnings
from pathlib import Path

import music21 as m21
from music21 import bar

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import generate_scores as gs  # find_musescore(), _np() 재사용

QL_TO_TOKEN = {
    4.0: '1/1', 3.0: '3/4', 2.0: '1/2', 1.5: '3/8', 1.0: '1/4',
    0.75: '3/16', 0.5: '1/8', 0.375: '3/32', 0.25: '1/16', 0.125: '1/32',
}
SHARPS_TO_KEYNAME = {0: 'C', 1: 'G', -1: 'F', 2: 'D', -2: 'Bb', 3: 'A', -3: 'Eb',
                     4: 'E', -4: 'Ab', 5: 'B', -5: 'Db', 6: 'F#', -6: 'Gb'}
VALID_TIME_SIGS = {'2/4', '3/4', '4/4', '6/8', '3/8', '12/8', '2/2', '5/4'}
ARTIC_MAP = {'Staccato': 'staccato', 'Accent': 'accent', 'Tenuto': 'tenuto',
             'StrongAccent': 'marcato', 'Staccatissimo': 'staccatissimo'}
ORNAMENT_MAP = {'Trill': 'trill', 'Mordent': 'mordent', 'InvertedMordent': 'mordent', 'Turn': 'turn'}
DYNAMIC_ALIAS = {'fz': 'f'}  # 어휘에 없는 표기를 가까운 값으로 치환(사용자 확인 완료)
PC_MIN, PC_MAX = m21.pitch.Pitch('C2').midi, m21.pitch.Pitch('B6').midi

# tokenizer258.json이 지원하는 피치 철자만 (Fb/Cb/E#/B# 같은 이명동음 철자는 없음 --
# music21이 조표 문맥상 그런 철자를 고를 수 있어서 어휘에 맞게 재철자해야 함)
PC_TO_SHARP_NAME = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
PC_TO_FLAT_NAME = {0: 'C', 1: 'Db', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F',
                   6: 'Gb', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'}


def dur_token(ql, warn_list, ctx):
    ql = float(ql)
    best = min(QL_TO_TOKEN.keys(), key=lambda k: abs(k - ql))
    if abs(best - ql) > 1e-6:
        warn_list.append(f"{ctx}: duration {ql} -> 가장 가까운 {QL_TO_TOKEN[best]}로 근사")
    return QL_TO_TOKEN[best]


def pitch_token(p, warn_list, ctx):
    midi = p.midi
    name = gs._np(p.name)  # 옥타브 뺀 철자(스텝+임시표), 예: 'C#', 'Db', 'Fb'
    pc = midi % 12
    octave = midi // 12 - 1
    if name not in (PC_TO_SHARP_NAME[pc], PC_TO_FLAT_NAME[pc]):
        # Fb/Cb/E#/B# 같은 어휘 밖 철자 -- 원래 임시표가 샵 계열이면 샵 철자로,
        # 그 외(플랫/자연)엔 플랫 철자로 재철자. 옥타브도 midi 기준으로 다시 계산해서
        # Cb5(=B4)/B#3(=C4) 같은 옥타브 경계 이동을 정확히 반영.
        acc = p.accidental
        use_sharp = acc is not None and acc.modifier == '#'
        name = PC_TO_SHARP_NAME[pc] if use_sharp else PC_TO_FLAT_NAME[pc]
        warn_list.append(f"{ctx}: 어휘 밖 철자 {gs._np(p.nameWithOctave)} -> {name}{octave}로 재철자")
    tok = f"{name}{octave}"
    if not (PC_MIN <= midi <= PC_MAX):
        warn_list.append(f"{ctx}: 음높이 {tok}가 어휘 범위(C2~B6) 밖")
    return tok


def barline_token(measure, is_last, warn_list, ctx):
    if is_last:
        return 'barline-final'
    rb = measure.rightBarline
    if rb is None:
        return 'barline'
    if isinstance(rb, bar.Repeat):
        return 'barline-end-repeat'
    style = getattr(rb, 'type', None) or getattr(rb, 'style', None)
    if style in ('light-light', 'double'):
        return 'barline-double'
    if style in ('light-heavy', 'final'):
        return 'barline-final'
    return 'barline'


def has_repeat_start(measure):
    lb = measure.leftBarline
    return isinstance(lb, bar.Repeat)


def _is_triplet(el):
    """3연음(3:2, ql=1/3)만 지원 -- generate_scores.py의 tuplet-3-start/end 관례와 동일.
    다른 비율(5연음 등)은 어휘에 없어서 여기서 감지 안 하고 일반 duration 근사로 넘어감."""
    if not el.duration.tuplets:
        return False
    t = el.duration.tuplets[0]
    return t.numberNotesActual == 3 and t.numberNotesNormal == 2


def _emit_one(el, dtok, toks, warn_list, ctx):
    if el.isRest:
        toks.append(f"rest-{dtok}")
        return
    if el.isChord:
        pitches = sorted(el.pitches, key=lambda p: p.midi)
        toks.append(f"note-{pitch_token(pitches[0], warn_list, ctx)}")
        toks.append(f"dur-{dtok}")
        for p in pitches[1:]:
            toks.append(f"chord-{pitch_token(p, warn_list, ctx)}")
    else:
        toks.append(f"note-{pitch_token(el.pitch, warn_list, ctx)}")
        toks.append(f"dur-{dtok}")

    for a in getattr(el, 'articulations', []):
        name = ARTIC_MAP.get(type(a).__name__)
        if name:
            toks.append(f"artic-{name}")
    for e in getattr(el, 'expressions', []):
        cname = type(e).__name__
        if cname == 'Fermata':
            toks.append('fermata')
        elif cname in ORNAMENT_MAP:
            toks.append(f"ornament-{ORNAMENT_MAP[cname]}")

    # 붙임줄(tie) -- start/continue는 "바로 다음 음표로 이어진다"는 뜻의 `tie` 토큰 1개로
    # 표현(체인 길이 3+인 continue도 동일 토큰 재사용). stop(도착점)에는 아무것도 안 붙임
    # -- generate_scores.py의 TIE_PROB 표기 관례와 동일(docs/music-notation-rule-designer.md
    # 대신 이쪽은 round3train 쪽 관례라 TrainingStep.md에 정리돼 있음). MusicXML의 <tie>는
    # 정의상 항상 같은 음높이만 잇는 표기라 여기서 음높이 일치를 별도로 검증할 필요가 없음.
    t = getattr(el, 'tie', None)
    if t is not None and t.type in ('start', 'continue'):
        toks.append('tie')


def emit_measure_tokens(measure, warn_list, ctx):
    toks = []
    elements = sorted(measure.notesAndRests, key=lambda e: e.offset)
    i = 0
    while i < len(elements):
        el = elements[i]
        if _is_triplet(el):
            group = [el]
            j = i + 1
            while j < len(elements) and _is_triplet(elements[j]) and len(group) < 3:
                group.append(elements[j])
                j += 1
            if len(group) != 3:
                warn_list.append(f"{ctx}: 3연음 그룹이 3개가 아닌 {len(group)}개로 감지됨 -- 일반 duration으로 근사")
                _emit_one(el, dur_token(el.duration.quarterLength, warn_list, ctx), toks, warn_list, ctx)
                i += 1
                continue
            toks.append('tuplet-3-start')
            for g in group:
                _emit_one(g, '1/8', toks, warn_list, ctx)
            toks.append('tuplet-3-end')
            i = j
            continue
        dtok = dur_token(el.duration.quarterLength, warn_list, ctx)
        _emit_one(el, dtok, toks, warn_list, ctx)
        i += 1

    dyn_toks = []
    for d in measure.getElementsByClass('Dynamic'):
        value = DYNAMIC_ALIAS.get(d.value, d.value)
        dyn_toks.append((d.offset, f"dynamic-{value}"))
    for off, tok in sorted(dyn_toks, key=lambda x: x[0]):
        toks.insert(0, tok) if off <= 1e-6 else None

    return toks


def convert_part_pair(score, warn_list):
    """2파트(대보표 -- 위=치, 아래=베이스) 전제로 토큰 생성."""
    parts = list(score.parts)
    treble_part, bass_part = parts[0], parts[1]
    t_measures = list(treble_part.getElementsByClass('Measure'))
    b_measures = list(bass_part.getElementsByClass('Measure'))
    n = min(len(t_measures), len(b_measures))
    if len(t_measures) != len(b_measures):
        warn_list.append(f"치/베이스 마디 수 불일치({len(t_measures)} vs {len(b_measures)}) -- 짧은 쪽 기준({n}개)으로 자름")

    m0 = t_measures[0]
    ks = m0.getElementsByClass('KeySignature').first()
    ts = m0.getElementsByClass('TimeSignature').first()
    ks_name = SHARPS_TO_KEYNAME.get(ks.sharps if ks else 0, 'C')
    ts_str = f"{ts.numerator}/{ts.denominator}" if ts else "4/4"
    if ts_str not in VALID_TIME_SIGS:
        warn_list.append(f"박자 {ts_str}가 어휘에 없는 박자")

    # 2026-08-03: 치/베이스 클렙을 'G'/'F'로 무조건 고정했었는데, 편곡에 따라 베이스보표를
    # 치클렙(G)으로 표기하는 경우가 실제로 있음(newage19 held-out 실측에서 발견 -- mscx에
    # <defaultClef>F</defaultClef>가 있어도 첫 마디에 <Clef><concertClefType>G</...>로
    # 오버라이드된 채 렌더링되는 케이스, 사진과도 일치). convert_single_part()와 동일하게
    # 각 파트의 실제 첫 Clef 객체를 읽어서 반영한다.
    t_clef_obj = m0.getElementsByClass('Clef').first()
    b_clef_obj = b_measures[0].getElementsByClass('Clef').first()
    treble_clef_tok = 'clef-F' if t_clef_obj and t_clef_obj.sign == 'F' else 'clef-G'
    bass_clef_tok = 'clef-F' if b_clef_obj and b_clef_obj.sign == 'F' else 'clef-G'

    tokens = ['<SOS>', treble_clef_tok, f'key-{ks_name}', f'time-{ts_str}']
    bass_clef_emitted = False
    for i in range(n):
        tm, bm = t_measures[i], b_measures[i]
        if has_repeat_start(tm):
            tokens.append('barline-start-repeat')
        tokens.extend(emit_measure_tokens(tm, warn_list, f"마디{i+1}(치)"))
        tokens.append('staff-bass')
        if not bass_clef_emitted:
            tokens.append(bass_clef_tok)
            bass_clef_emitted = True
        tokens.extend(emit_measure_tokens(bm, warn_list, f"마디{i+1}(베이스)"))
        tokens.append(barline_token(tm, i == n - 1, warn_list, f"마디{i+1}"))
    tokens.append('<EOS>')
    return tokens


def convert_single_part(score, warn_list):
    part = list(score.parts)[0]
    measures = list(part.getElementsByClass('Measure'))
    m0 = measures[0]
    clef_obj = m0.getElementsByClass('Clef').first()
    clef_tok = 'clef-F' if clef_obj and clef_obj.sign == 'F' else 'clef-G'
    ks = m0.getElementsByClass('KeySignature').first()
    ts = m0.getElementsByClass('TimeSignature').first()
    ks_name = SHARPS_TO_KEYNAME.get(ks.sharps if ks else 0, 'C')
    ts_str = f"{ts.numerator}/{ts.denominator}" if ts else "4/4"
    if ts_str not in VALID_TIME_SIGS:
        warn_list.append(f"박자 {ts_str}가 어휘에 없는 박자")

    tokens = ['<SOS>', clef_tok, f'key-{ks_name}', f'time-{ts_str}']
    for i, m in enumerate(measures):
        if has_repeat_start(m):
            tokens.append('barline-start-repeat')
        tokens.extend(emit_measure_tokens(m, warn_list, f"마디{i+1}"))
        tokens.append(barline_token(m, i == len(measures) - 1, warn_list, f"마디{i+1}"))
    tokens.append('<EOS>')
    return tokens


def convert_mscz(mscz_path: Path, musescore_path: str, tmp_xml: Path):
    subprocess.run([musescore_path, "-o", str(tmp_xml), str(mscz_path)],
                   capture_output=True, text=True, timeout=90)
    if not tmp_xml.exists():
        raise RuntimeError(f"MuseScore 변환 실패: {mscz_path}")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        score = m21.converter.parse(str(tmp_xml))
    warn_list = []
    n_parts = len(score.parts)
    if n_parts >= 2:
        tokens = convert_part_pair(score, warn_list)
    elif n_parts == 1:
        tokens = convert_single_part(score, warn_list)
    else:
        raise RuntimeError(f"파트가 없음: {mscz_path}")
    return tokens, warn_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', help='.mscz 파일 또는 그 파일이 들어있는 디렉토리')
    ap.add_argument('--musescore', default=None)
    ap.add_argument('--out', default=None, help='출력 json 경로(미지정 시 대상 폴더에 폴더이름.json)')
    ap.add_argument('--force', action='store_true', help='이미 json이 있어도 덮어씀')
    args = ap.parse_args()

    musescore_path = gs.find_musescore(args.musescore)
    if not musescore_path:
        print("ERROR: MuseScore를 찾을 수 없음", file=sys.stderr)
        sys.exit(1)

    target = Path(args.target)
    if target.is_dir():
        mscz_files = sorted(target.glob('*.mscz'))
        if not mscz_files:
            print(f"SKIP(mscz 없음): {target}")
            return
        mscz_path = mscz_files[0]
        out_path = Path(args.out) if args.out else target / f"{target.name}.json"
    else:
        mscz_path = target
        out_path = Path(args.out) if args.out else target.with_suffix('.json')

    if out_path.exists() and not args.force:
        print(f"SKIP(이미 존재): {out_path}")
        return

    tmp_xml = out_path.with_suffix('.musicxml')
    try:
        tokens, warn_list = convert_mscz(mscz_path, musescore_path, tmp_xml)
    except Exception as exc:
        print(f"ERROR: {mscz_path} -- {exc}", file=sys.stderr)
        return
    finally:
        if tmp_xml.exists():
            tmp_xml.unlink()

    stem = out_path.stem
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"id": stem, "tokens": tokens}, f, ensure_ascii=False)

    print(f"OK: {out_path} (토큰 {len(tokens)}개)")
    for w in warn_list:
        print(f"  경고: {w}")


if __name__ == '__main__':
    main()
