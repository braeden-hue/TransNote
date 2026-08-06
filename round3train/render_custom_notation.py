"""exactPicture 곡 하나를 커스텀 악보(자체 표기법) SVG/PNG로 렌더링.

online_webpage/js/notation.js(실제 웹 서비스 렌더러)와는 별개의 Python 구현 —
웹페이지에는 아직 반영하지 않은 신규 규칙을 여기서 먼저 검증한다:

  규칙 1 (붙임줄): tie로 이어진 동일 음높이 두 음을 합쳐 하나의 셀(합산 박자)로 표현.
                  8분음표 두 개가 tie로 이어지면 4분음표 한 칸으로 인식.
  규칙 2 (화음):    화음을 세로 스택이 아니라 가로 나열(도~시 오름차순)로 표현하고,
                  옥타브가 다르면 각 음을 자기 옥타브 존(행)에 따로 적는다. 정의된
                  3존 범위를 벗어나는 음(트레블 3옥 이하 / 베이스 4옥 이상)은
                  최하단 칸 바로 밑 / 최상단 칸 바로 위의 예외 행에 적되, 그 행은
                  정식 칸(배경/구분선/라벨)을 그리지 않는 빈 여백으로 둔다. 여러 행에
                  걸치는 화음도 관련 행을 모두 감싸는 박스 하나로 묶어 표현한다.
  규칙 3 (색상):    박자는 셀 너비(음 길이)만으로 표현. 왼쪽 존 라벨/마디 번호 등 텍스트는
                  전부 없애고, 대신 클렙마다 다른 색 테마(트레블=하늘색 계열, 베이스=주황
                  계열)를 오선 배경·음표·화음 박스·쉼표까지 전부에 준다. 음표 색은 그
                  클렙 색 계열을 쓰되 옥타브 행이 높을수록 진하고 낮을수록 옅어지는 농담
                  단계로 구별한다. 마디 중간에 clef 토큰으로 클렙이 바뀌면 그 구간만
                  반대쪽 클렙의 색 테마로 바뀐다.
  규칙 4 (마디):    한 줄에 모든 마디를 이어 쓰지 않고 마디마다 한 행(row)으로 끊어서
                  세로로 쌓는다(스크롤로 마디 단위 이동하는 인터랙션의 정적 프리뷰 —
                  실제 스크롤/페이징 동작은 웹페이지 쪽 구현이 필요해 여기선 다루지 않음).
  규칙 5 (쉼표):    점선 사각형 박스로 박자 길이만큼 표현("쉼표" 라벨, 자기 클렙 색).
  규칙 6 (화음 박스): 채움 없이 테두리 선으로만 감싸되, 자기 클렙 색 테마를 쓴다.
  규칙 7 (최소 너비): 8/16분음표 등 짧은 음도 최소 셀 너비를 보장(너무 좁아 안 보이는 것 방지).

zoneLabels/pitchToZone 등 시각 규칙은 online_webpage/js/samples.js를 참고했지만
코드는 공유하지 않음(수정 없이 그대로 둠).

사용법:
    python render_custom_notation.py newage14 --out_png <path> [--musescore <exe>]
"""
import argparse
import json
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SRC_DIR = _HERE / 'data' / 'local_pools' / 'exactPicture'

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# ── online_webpage/js/samples.js와 동일 상수 (읽기 전용 참고) ────────────────
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# 규칙 3: 클렙마다 다른 색 테마 — 트레블=하늘색 계열, 베이스=주황 계열. 같은 계열 안에서
# 옥타브 행(row 0=최상단)이 높을수록 진하고 낮을수록 옅어지는 4단계 농담.
CLEF_ROW_COLORS = {
    'treble': ['#0284C7', '#0EA5E9', '#38BDF8', '#7DD3FC'],
    'bass':   ['#EA580C', '#F97316', '#FB923C', '#FDBA74'],
}
# 화음 박스 테두리 / 쉼표 점선 테두리 — 클렙 색 테마의 대표 톤(각 행 색 농담 중 3번째,
# row_colors[2])을 그대로 재사용해 팔레트를 늘리지 않는다.
CHORD_BOX_COLOR = {c: CLEF_ROW_COLORS[c][2] for c in CLEF_ROW_COLORS}
REST_COLOR = CHORD_BOX_COLOR
# 오선 배경(짝수 존 fill, 홀수 존 fill) — 클렙 테마를 배경에도 옅게 반영.
CLEF_ZONE_BG = {
    'treble': ('#EAF4FF', '#F5FAFF'),
    'bass':   ('#FFF1E4', '#FFF8F2'),
}
BLACK_LABEL = {'C#': '1', 'D#': '2', 'F#': '3', 'G#': '4', 'A#': '5'}
_FLAT_TO_SHARP = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}


def format_note_name(pitch: str) -> str:
    name = pitch[:-1]
    return BLACK_LABEL.get(name, name)


def pitch_class(pitch: str) -> int:
    return NOTE_NAMES.index(pitch[:-1])


def normalize_pitch(pitch: str) -> str:
    if not pitch:
        return pitch
    name, octave = pitch[:-1], pitch[-1]
    return _FLAT_TO_SHARP.get(name, name) + octave


def fraction_to_quarters(frac: str) -> float:
    n, d = frac.split('/')
    return float(n) / float(d) * 4.0


# ── 규칙 2: 4행 존 매핑 (기존 3존 + 예외 1행) ──────────────────────────────
def row_index(pitch: str, clef: str) -> int:
    oct_ = int(pitch[-1])
    if clef == 'bass':
        if oct_ >= 4: return 0   # 예외: 최상단 바로 위 칸
        if oct_ == 3: return 1
        if oct_ == 2: return 2
        return 3                 # 1옥 이하 (기존 낮음 존 그대로)
    # treble
    if oct_ >= 6: return 0       # 기존 높음 존 그대로 (6옥+)
    if oct_ == 5: return 1
    if oct_ == 4: return 2
    return 3                     # 예외: 최하단 바로 밑 칸 (3옥 이하)


# ── 토큰 파싱: token_to_notes.py의 tokens_to_score 로직 + 규칙 1(붙임줄 병합) ──
# 'beat'(박자 카운터)는 색상에도 안 쓰고, 마디 표시도 실제 barline 토큰을 직접 추적하는
# measureStart 플래그로 하므로 더 이상 필요 없음 — beat 카운터의 반올림 누적 오차로
# 마디 중간에 beat==1이 되는(따라서 가짜 마디선이 찍히는) 문제를 피하기 위함.
def tokens_to_score_tie_aware(tokens):
    treble, bass = [], []
    on_bass = False
    treble_mstart = True  # 다음에 push될 노트가 각 보표의 "마디 시작 첫 음"인지
    bass_mstart = True
    # 규칙 3: 보표별로 따로 추적 — clef-F는 토큰 스트림에 보통 베이스 보표 진입 시
    # 딱 한 번만 나오고 이후 마디들은 반복하지 않으므로, 단일 전역 변수로 두면
    # 그 값이 트레블 쪽까지 그대로 새어나가 버린다(실제 겪은 버그). "어느 보표가
    # 지금 어떤 클렙으로 읽히는지"를 보표별로 유지해야 진짜 마디 중간 클렙 전환만 잡힌다.
    treble_clef = 'treble'
    bass_clef = 'bass'

    pending_pitch = None
    pending_duration = 1.0
    pending_chord = []
    has_pending = False
    tie_carry = 0.0       # 규칙 1: 이전 붙임줄 음의 박자를 누적
    expect_tie = False    # 직전에 'tie' 토큰을 봤음 -> 다음 note가 이어붙을 대상

    def push(pitch, duration, chord=None, is_rest=False):
        nonlocal treble_mstart, bass_mstart
        target = bass if on_bass else treble
        note = {'pitch': normalize_pitch(pitch), 'duration': duration,
                 'clef': bass_clef if on_bass else treble_clef}
        if chord:
            note['chordNotes'] = [normalize_pitch(c) for c in chord]
        if is_rest:
            note['isRest'] = True
        if on_bass and bass_mstart:
            note['measureStart'] = True
            bass_mstart = False
        elif not on_bass and treble_mstart:
            note['measureStart'] = True
            treble_mstart = False
        target.append(note)

    def finalize_pending():
        nonlocal pending_pitch, pending_duration, pending_chord, has_pending, tie_carry
        if not has_pending:
            return
        push(pending_pitch, tie_carry + pending_duration, chord=list(pending_chord))
        pending_pitch, pending_duration, pending_chord = None, 1.0, []
        has_pending, tie_carry = False, 0.0

    for tok in tokens:
        if tok.startswith('clef-'):
            # 규칙 3: clef-G -> 트레블 색 테마, clef-F -> 베이스 색 테마. 지금 활성인
            # 보표(on_bass)에만 적용 — 이게 헤더의 1회성 선언인지 진짜 마디 중간
            # 클렙 전환인지는 토큰 스트림만으로 구분 안 되므로, "지금 이 보표는 이
            # 클렙으로 읽는다"를 보표별로 계속 갱신해두는 것으로 둘 다 커버한다.
            finalize_pending()
            letter = tok[5:]
            new_clef = 'treble' if letter == 'G' else 'bass' if letter == 'F' else None
            if new_clef:
                if on_bass:
                    bass_clef = new_clef
                else:
                    treble_clef = new_clef
        elif tok.startswith('key-') or tok.startswith('time-'):
            pass
        elif tok == 'staff-bass':
            finalize_pending()
            on_bass = True
        elif tok == 'tie':
            if has_pending:
                expect_tie = True
        elif tok.startswith('note-'):
            new_pitch = tok[5:]
            if expect_tie and has_pending and normalize_pitch(new_pitch) == normalize_pitch(pending_pitch):
                # 규칙 1: 동일 음높이로 이어붙음 -> 별도 셀로 push하지 않고 박자를 이월
                tie_carry += pending_duration
                pending_pitch, pending_duration, pending_chord = new_pitch, 1.0, []
                has_pending = True
            else:
                finalize_pending()
                pending_pitch, pending_duration, pending_chord = new_pitch, 1.0, []
                has_pending = True
            expect_tie = False
        elif tok.startswith('dur-'):
            if has_pending:
                pending_duration = fraction_to_quarters(tok[4:])
        elif tok.startswith('chord-'):
            if has_pending:
                pending_chord.append(tok[6:])
        elif tok.startswith('rest-'):
            finalize_pending()
            push('', fraction_to_quarters(tok[5:]), is_rest=True)
        elif tok in ('barline', 'barline-double', 'barline-final', 'barline-end-repeat', 'barline-start-repeat'):
            finalize_pending()
            treble_mstart, bass_mstart, on_bass = True, True, False
        # dynamic-*/artic-*/fermata/ornament-*/slur-*/trill-*/tuplet-*/ottava-*/hairpin-*: 무시

    finalize_pending()
    return treble, bass


# ── SVG 생성 (규칙 2: 화음을 행별로 분리, 같은 행은 가로 나열 오름차순) ────────
UNIT_W, CELL_H, ZONE_H = 80, 46, 56
MARGIN_L, MARGIN_Y, INDICATOR_H = 16, 8, 18
N_ROWS = 4
MIN_SLOT_W = 44          # 규칙 7: 짧은 음(8/16분음표)도 이 너비 밑으로는 안 좁아짐
REST_ROW = 2              # 규칙 5: 쉼표는 pitch가 없으므로 고정 행(가온다/2옥 자리)에 표기
MEASURE_GAP = 24           # 규칙 4: 마디별 행 사이 세로 간격


def slot_width(duration):
    return max(duration * UNIT_W, MIN_SLOT_W)


def split_measures(notes):
    measures, cur = [], []
    for n in notes:
        if n.get('measureStart') and cur:
            measures.append(cur)
            cur = []
        cur.append(n)
    if cur:
        measures.append(cur)
    return measures


def clef_runs(ms):
    """마디 안에서 note['clef']가 바뀌는 구간을 (x시작, 폭, clef) 리스트로 묶는다 —
    마디 중간 클렙 전환 시 그 구간만 반대쪽 클렙 배경색을 쓰기 위함(규칙 3)."""
    runs = []
    x = MARGIN_L + 4
    seg_start = x
    cur = ms[0]['clef'] if ms else 'treble'
    for n in ms:
        if n['clef'] != cur:
            runs.append((seg_start, x - seg_start, cur))
            seg_start, cur = x, n['clef']
        x += slot_width(n['duration'])
    runs.append((seg_start, x - seg_start, cur))
    return runs


def build_stave_svg(notes, clef):
    measures = split_measures(notes) or [[]]
    row_block_h = INDICATOR_H + ZONE_H * N_ROWS + MARGIN_Y * 2

    def measure_content_w(ms):
        return MARGIN_L + 4 + sum(slot_width(n['duration']) for n in ms) + 36

    svg_w = max((measure_content_w(ms) for ms in measures), default=MARGIN_L + 100)
    svg_h = row_block_h * len(measures) + MEASURE_GAP * (len(measures) - 1)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" height="{svg_h:.0f}" '
             f'viewBox="0 0 {svg_w:.0f} {svg_h:.0f}">',
             f'<rect x="0" y="0" width="{svg_w:.0f}" height="{svg_h:.0f}" fill="#fff"/>']

    for mi, ms in enumerate(measures):
        y_base = mi * (row_block_h + MEASURE_GAP)
        content_y = y_base + INDICATOR_H + MARGIN_Y
        runs = clef_runs(ms)

        # 트레블 예외 행("3옥 이하", z==3)은 정식 칸으로 취급하지 않음 — 배경/구분선을
        # 그리지 않고 빈 여백으로 둔다(음이 있을 때만 그 자리에 텍스트/박스가 나타남).
        # 왼쪽 라벨 텍스트는 전부 제거 — 클렙별 배경색 테마로 대신 구분한다.
        for z in range(N_ROWS):
            if clef == 'treble' and z == 3:
                continue
            zy = content_y + z * ZONE_H
            for rx, rw, run_clef in runs:
                fill = CLEF_ZONE_BG[run_clef][z % 2]
                parts.append(f'<rect x="{rx:.1f}" y="{zy}" width="{rw:.1f}" height="{ZONE_H}" fill="{fill}"/>')
            if z > 0:
                parts.append(f'<line x1="{MARGIN_L}" y1="{zy}" x2="{svg_w - 8:.0f}" y2="{zy}" '
                              f'stroke="#C5D8EC" stroke-width="1.5" stroke-dasharray="6,4"/>')

        x = MARGIN_L + 4
        for note in ms:
            note_clef = note.get('clef', clef)
            row_colors = CLEF_ROW_COLORS[note_clef]
            sw = slot_width(note['duration'])
            w = sw - 4

            if note.get('isRest'):
                # 규칙 5: 쉼표 = 점선 사각형 박스(박자 길이만큼) + "쉼표" 라벨. 색은 자기
                # 클렙 테마의 대표 톤(화음 박스와 동일 색)을 써서 팔레트를 통일한다.
                rest_color = REST_COLOR[note_clef]
                y = content_y + REST_ROW * ZONE_H + 5
                h = CELL_H
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="5" '
                              f'fill="none" stroke="{rest_color}" stroke-width="2" stroke-dasharray="5,4"/>')
                fs = 9 if w < 26 else 11 if w < 42 else 13
                parts.append(f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 5:.1f}" text-anchor="middle" '
                              f'fill="{rest_color}" font-size="{fs}" font-weight="700" font-family="system-ui">쉼표</text>')
                x += sw
                continue

            pitches = [note['pitch']] + note.get('chordNotes', [])
            rows = {}
            for p in pitches:
                rows.setdefault(row_index(p, note_clef), []).append(p)
            for plist in rows.values():
                plist.sort(key=pitch_class)

            if len(pitches) == 1:
                # 단음 표기: 박스 없이 텍스트 + 밑줄. 색은 자기 클렙 색 계열의 옥타브 행 농담.
                ridx = next(iter(rows))
                color = row_colors[ridx]
                y = content_y + ridx * ZONE_H + 5
                h = CELL_H
                parts.append(f'<line x1="{x:.1f}" y1="{y + h}" x2="{x + w:.1f}" y2="{y + h}" '
                              f'stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>')
                fs = 10 if w < 26 else 12 if w < 42 else 14
                parts.append(f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 5:.1f}" text-anchor="middle" '
                              f'fill="{color}" font-size="{fs}" font-weight="700" font-family="system-ui">'
                              f'{format_note_name(pitches[0])}</text>')
            else:
                # 화음: 몇 개 행에 걸치든 하나의 셀로 묶어서 표현 — 관련된 행 전체를
                # 감싸는 박스 하나를 그리고, 그 안에서 각 음을 자기 행 높이·자기 행 색으로
                # 배치한다(같은 행을 공유하는 음은 가로로 오름차순 나열).
                # 규칙 6: 박스는 채움 없이 테두리 선으로만 감싸되, 자기 클렙 테마 색을 쓴다.
                min_row, max_row = min(rows), max(rows)
                y_top = content_y + min_row * ZONE_H + 5
                y_bot = content_y + max_row * ZONE_H + 5 + CELL_H
                parts.append(f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{w:.1f}" height="{y_bot - y_top:.1f}" rx="5" '
                              f'fill="none" stroke="{CHORD_BOX_COLOR[note_clef]}" stroke-width="2"/>')
                for ridx, plist in rows.items():
                    color = row_colors[ridx]
                    y = content_y + ridx * ZONE_H + 5
                    h = CELL_H
                    n_sub = len(plist)
                    sub_w = w / n_sub
                    fs = 9 if sub_w < 26 else 11 if sub_w < 42 else 13
                    for i, p in enumerate(plist):
                        cx = x + sub_w * (i + 0.5)
                        parts.append(f'<text x="{cx:.1f}" y="{y + h / 2 + 5:.1f}" text-anchor="middle" '
                                      f'fill="{color}" font-size="{fs}" font-weight="700" font-family="system-ui">'
                                      f'{format_note_name(p)}</text>')
            x += sw

    parts.append('</svg>')
    return '\n'.join(parts), svg_w, svg_h


def svg_to_png(svg_path: Path, png_path: Path, width: float, height: float, scale=3, chrome=CHROME):
    subprocess.run([
        chrome, '--headless', '--disable-gpu', '--force-color-profile=srgb',
        f'--force-device-scale-factor={scale}',
        f'--window-size={int(width)},{int(height)}',
        f'--screenshot={png_path}',
        svg_path.as_uri(),
    ], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('song', help='exactPicture 하위 폴더명 (예: newage14)')
    ap.add_argument('--out_png', required=True)
    ap.add_argument('--max_measures', type=int, default=None, help='앞 N마디만 렌더링 (미지정 시 전체)')
    ap.add_argument('--musescore_chrome', default=CHROME)
    args = ap.parse_args()

    song_dir = SRC_DIR / args.song
    json_paths = sorted(song_dir.glob(f'{args.song}.json'))
    if not json_paths:
        raise SystemExit(f"{song_dir}에 {args.song}.json 없음")
    tokens = json.loads(json_paths[0].read_text(encoding='utf-8'))['tokens']

    if args.max_measures:
        from prepare_exactpicture_test import relabel, trim_to_measures, strip_out_of_scope
        tokens = strip_out_of_scope(trim_to_measures(relabel(tokens), args.max_measures))

    treble, bass = tokens_to_score_tie_aware(tokens)

    import tempfile
    work_dir = Path(tempfile.gettempdir()) / f'{args.song}_custom_work'
    work_dir.mkdir(parents=True, exist_ok=True)

    tr_svg, tr_w, tr_h = build_stave_svg(treble, 'treble')
    ba_svg, ba_w, ba_h = build_stave_svg(bass, 'bass')
    tr_svg_path, ba_svg_path = work_dir / 'treble.svg', work_dir / 'bass.svg'
    tr_svg_path.write_text(tr_svg, encoding='utf-8')
    ba_svg_path.write_text(ba_svg, encoding='utf-8')

    tr_png_path, ba_png_path = work_dir / 'treble.png', work_dir / 'bass.png'
    svg_to_png(tr_svg_path, tr_png_path, tr_w, tr_h, chrome=args.musescore_chrome)
    svg_to_png(ba_svg_path, ba_png_path, ba_w, ba_h, chrome=args.musescore_chrome)

    from PIL import Image, ImageDraw, ImageFont
    tr_img = Image.open(tr_png_path).convert('RGB')
    ba_img = Image.open(ba_png_path).convert('RGB')
    font_bold = ImageFont.truetype(r'C:\Windows\Fonts\malgunbd.ttf', 34)

    PAD, LABEL_H, GAP = 40, 50, 24
    W = max(tr_img.width, ba_img.width) + PAD * 2
    H = PAD + LABEL_H + tr_img.height + GAP + LABEL_H + ba_img.height + PAD
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    y = PAD
    draw.text((PAD, y), '높은음자리 (오른손)', font=font_bold, fill=(0, 118, 206))
    y += LABEL_H
    canvas.paste(tr_img, (PAD, y))
    y += tr_img.height + GAP
    draw.text((PAD, y), '낮은음자리 (왼손)', font=font_bold, fill=(91, 184, 245))
    y += LABEL_H
    canvas.paste(ba_img, (PAD, y))

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"완료: {out_path} ({canvas.size})")


if __name__ == '__main__':
    main()
