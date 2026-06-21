#!/usr/bin/env python3
"""
render_notation.py — DeepScore JSON 토큰 → 커스텀 악보 이미지 렌더러

사용법:
    python omr/utils/render_notation.py data/train/num1.json
    python omr/utils/render_notation.py data/train/num1.json data/train/num3.json --out output/
    python omr/utils/render_notation.py data/train/num1.json --show
    python omr/utils/render_notation.py data/train/num1.json --cell 32 --font-size 22

규칙 (project.md Section 5 기준):
  - 보표: 직사각형 박스, 수직 3등분 = 옥타브 존
  - 흰 건반 (CDEFGAB): 흰 글자 + 검은 외곽선
  - 검은 건반 (C#=1, D#=2, F#=3, G#=4, A#=5): 검은 글자
  - 음가: 16분음표 기준 셀, 음표 글자는 점유 셀 정중앙 배치
  - 빈 셀: 쉼표 (별도 기호 없음)
  - 음표 색상 테두리: 주황 → 연두 → 하늘 순환
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("pillow 가 설치되어 있지 않습니다.  pip install pillow  를 실행하세요.")

# ── 상수 ──────────────────────────────────────────────────────────────────────

DEFAULT_CELL_W  = 24    # 16분음표 1개 = 24 px 너비
DEFAULT_ZONE_H  = 52    # 옥타브 존 1개 높이 (px)
DEFAULT_FONT_SZ = 18    # 음표 글자 폰트 크기 (px)

HEADER_H   = 30   # 박자표 레이블 영역 높이
FOOTER_H   = 10   # 하단 여백
CLEF_W     = 36   # 좌측 음자리표 레이블 너비
BARLINE_W  = 2    # 마디선 두께
MEASURE_PAD = 4   # 마디 좌우 안쪽 여백

# 음표 셀 테두리 색 (주황 → 연두 → 하늘, 순환)
NOTE_COLORS: List[Tuple[int, int, int]] = [
    (255, 140,   0),   # 주황
    (100, 200,  50),   # 연두
    (  0, 191, 255),   # 하늘
]

WHITE_KEYS = set("CDEFGAB")
BLACK_KEY_MAP = {
    "C#": "1", "Db": "1",
    "D#": "2", "Eb": "2",
    "F#": "3", "Gb": "3",
    "G#": "4", "Ab": "4",
    "A#": "5", "Bb": "5",
}
# 이명동음 중 흰 건반에 해당하는 케이스 (B# = C, E# = F, Cb = B, Fb = E)
ENHARMONIC_WHITE = {"B#": "C", "E#": "F", "Cb": "B", "Fb": "E"}

# duration 토큰 → quarterLength
DURATION_QL = {
    "1/1": 4.0,  "3/4": 3.0,  "1/2": 2.0,  "3/8": 1.5,
    "1/4": 1.0,  "3/16": 0.75, "1/8": 0.5,  "3/32": 0.375,
    "1/16": 0.25, "1/32": 0.125,
}

# 셈여림 한국어 변환
DYN_KO = {
    "ppp": "여리여리여리게", "pp": "여리여리게", "p": "여리게",
    "mp": "조금 여리게",    "mf": "조금 세게",  "f": "세게",
    "ff": "매우 세게",      "fff": "아주 세게",
    "fp": "세게→여리게",    "sf": "특히 세게",  "sfz": "특히 세게",
}

# 정규식
_NOTE_RE  = re.compile(r"^note-([A-G][#b]?)(\d+)-(.+)$")
_CHORD_RE = re.compile(r"^chord-([A-G][#b]?)(\d+)$")
_DYN_RE   = re.compile(r"^dynamic-(.+)$")

# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class PitchLabel:
    """한 옥타브 존에 표시할 텍스트 정보."""
    text: str         # 표시 문자 (예: "C", "1", "CEG", "C3E")
    zone: int         # 0=top, 1=mid, 2=bottom
    has_black: bool   # True → 검은 건반 포함 → 검은 텍스트


@dataclass
class NoteEvent:
    kind: str                          # "note" | "rest" | "dynamic"
    labels: List[PitchLabel] = field(default_factory=list)
    sub_cells: int = 4                 # 16분음표 단위 점유 셀 수
    dynamic_text: str = ""             # 셈여림 한국어 텍스트


@dataclass
class Header:
    clef: str     = "G"
    time_num: int = 4
    time_den: int = 4


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def _ql_to_sub(ql: float) -> int:
    """quarterLength → 16분음표 단위 셀 수 (최소 1)."""
    return max(1, round(ql * 4))


def _pitch_display(pitch: str) -> Tuple[str, bool]:
    """'C#' → ('1', True),  'C' → ('C', False),  'B#' → ('C', False)"""
    if pitch in WHITE_KEYS:
        return pitch, False
    if pitch in ENHARMONIC_WHITE:
        return ENHARMONIC_WHITE[pitch], False
    return BLACK_KEY_MAP.get(pitch, "?"), True


def _zone(octave: int, clef: str) -> int:
    """옥타브 번호 → 존 인덱스 (0=top, 1=mid, 2=bottom)."""
    if clef == "G":
        if octave >= 6: return 0
        if octave == 5: return 1
        return 2
    else:  # F (bass)
        if octave >= 3: return 0
        if octave == 2: return 1
        return 2


def parse_tokens(tokens: list) -> Tuple[Header, List[List[NoteEvent]]]:
    hdr = Header()
    measures: List[List[NoteEvent]] = []
    cur: List[NoteEvent] = []

    for tok in tokens:
        if tok in ("<SOS>", "<EOS>", "<PAD>", "<UNK>"):
            continue

        if tok.startswith("clef-"):
            hdr.clef = tok[5:6]  # 'G', 'F', 'C'
            continue

        if tok.startswith("key-"):
            continue  # 조표는 표시하지 않음

        if tok.startswith("time-"):
            n, d = tok[5:].split("/")
            hdr.time_num, hdr.time_den = int(n), int(d)
            continue

        if tok in ("barline", "barline-double", "barline-final",
                   "barline-start-repeat", "barline-end-repeat"):
            if cur:
                measures.append(cur)
                cur = []
            continue

        # chord: 앞 note 이벤트에 추가
        m = _CHORD_RE.match(tok)
        if m and cur:
            pitch, oct_ = m.group(1), int(m.group(2))
            disp, is_blk = _pitch_display(pitch)
            z = _zone(oct_, hdr.clef)
            last = cur[-1]
            if last.kind == "note":
                for lbl in last.labels:
                    if lbl.zone == z:
                        lbl.text += "/" + disp   # "/" 구분자로 혼동 방지
                        if is_blk:
                            lbl.has_black = True
                        break
                else:
                    last.labels.append(PitchLabel(disp, z, is_blk))
            continue

        # note
        m = _NOTE_RE.match(tok)
        if m:
            pitch, oct_, dur = m.group(1), int(m.group(2)), m.group(3)
            ql   = DURATION_QL.get(dur, 1.0)
            disp, is_blk = _pitch_display(pitch)
            z = _zone(oct_, hdr.clef)
            cur.append(NoteEvent(
                kind="note",
                labels=[PitchLabel(disp, z, is_blk)],
                sub_cells=_ql_to_sub(ql),
            ))
            continue

        # rest
        if tok.startswith("rest-"):
            ql = DURATION_QL.get(tok[5:], 1.0)
            cur.append(NoteEvent(kind="rest", sub_cells=_ql_to_sub(ql)))
            continue

        # dynamic
        m = _DYN_RE.match(tok)
        if m:
            ko = DYN_KO.get(m.group(1), m.group(1))
            cur.append(NoteEvent(kind="dynamic", dynamic_text=ko, sub_cells=0))
            continue

        # hairpin, artic, ornament, fermata, slur → 무시 (렌더링 미구현)

    if cur:
        measures.append(cur)

    return hdr, measures


# ── 렌더링 헬퍼 ───────────────────────────────────────────────────────────────

def _tint(color: Tuple[int, int, int], alpha: float = 0.18) -> Tuple[int, int, int]:
    """color 를 alpha 비율로 흰색과 혼합한 연한 배경색."""
    return tuple(int(255 * (1 - alpha) + c * alpha) for c in color)  # type: ignore


def _outlined_text(draw: ImageDraw.ImageDraw,
                   xy: Tuple[int, int], text: str,
                   font: ImageFont.ImageFont,
                   fill: Tuple[int, int, int],
                   outline: Tuple[int, int, int],
                   stroke: int = 2) -> None:
    """텍스트에 외곽선(outline)을 그린다. 흰 건반 글자에 사용."""
    x, y = xy
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx * dx + dy * dy <= stroke * stroke + 1:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(xy, text, font=font, fill=fill)


# ── 메인 렌더러 ───────────────────────────────────────────────────────────────

def render(hdr: Header,
           measures: List[List[NoteEvent]],
           cell_w: int = DEFAULT_CELL_W,
           zone_h: int = DEFAULT_ZONE_H,
           font_size: int = DEFAULT_FONT_SZ) -> Image.Image:
    """파싱된 헤더 + 마디 리스트 → PIL Image."""

    staff_h = zone_h * 3

    # 폰트 로드 (실패 시 기본 폰트로 대체)
    def _load_font(size: int) -> ImageFont.ImageFont:
        for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
        return ImageFont.load_default()

    font    = _load_font(font_size)
    font_sm = _load_font(max(10, font_size - 5))

    # 마디당 기준 서브셀 수 (박자표 기준)
    total_ql = hdr.time_num * (4.0 / hdr.time_den)
    base_sub = max(4, round(total_ql * 4))

    def _measure_w(events: List[NoteEvent]) -> int:
        used = sum(e.sub_cells for e in events if e.kind != "dynamic")
        return max(base_sub, used) * cell_w + MEASURE_PAD * 2

    m_widths = [_measure_w(m) for m in measures]

    # 캔버스 크기
    total_w = (CLEF_W
               + sum(m_widths)
               + (len(m_widths) + 1) * BARLINE_W   # 마디선들
               + 20)                                 # 우측 여백
    total_h = HEADER_H + staff_h + FOOTER_H

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    staff_top = HEADER_H
    staff_bot = staff_top + staff_h

    # ── 존 구분선 (점선, 밝은 회색) ──────────────────────────────────────────
    for z in (1, 2):
        y = staff_top + z * zone_h
        for x in range(CLEF_W, total_w - 20, 8):
            draw.line([(x, y), (x + 4, y)], fill=(210, 210, 210), width=1)

    # ── 보표 외곽 직사각형 ────────────────────────────────────────────────────
    draw.rectangle(
        [(CLEF_W, staff_top), (total_w - 20, staff_bot)],
        outline=(30, 30, 30), width=2,
    )

    # ── 음자리표 레이블 (T / B) ───────────────────────────────────────────────
    clef_label = "T" if hdr.clef == "G" else "B"
    draw.text((8, staff_top + staff_h // 2 - font_size // 2),
              clef_label, font=font, fill=(30, 30, 30))

    # ── 박자표 레이블 ─────────────────────────────────────────────────────────
    draw.text((CLEF_W + 6, 4),
              f"{hdr.time_num}/{hdr.time_den}", font=font_sm, fill=(80, 80, 80))

    # ── 마디 렌더링 ───────────────────────────────────────────────────────────
    cur_x = CLEF_W + BARLINE_W  # 첫 번째 마디 시작 x

    for mi, events in enumerate(measures):
        mw = m_widths[mi]

        # 마디 내부 세로선 (첫 마디 제외 — 보표 테두리가 첫 줄)
        if mi > 0:
            draw.line([(cur_x, staff_top), (cur_x, staff_bot)],
                      fill=(30, 30, 30), width=BARLINE_W)
            cur_x += BARLINE_W

        note_events  = [e for e in events if e.kind != "dynamic"]
        dyn_events   = [e for e in events if e.kind == "dynamic"]
        color_idx    = 0
        ex           = cur_x + MEASURE_PAD  # 이벤트 배치 시작 x

        for ev in note_events:
            color = NOTE_COLORS[color_idx % len(NOTE_COLORS)]
            ew    = ev.sub_cells * cell_w  # 이벤트 픽셀 너비

            if ev.kind == "note":
                tint_col = _tint(color, 0.22)

                # 배경 tint (각 존별)
                for lbl in ev.labels:
                    zy = staff_top + lbl.zone * zone_h
                    draw.rectangle(
                        [(ex + 1, zy + 1), (ex + ew - 1, zy + zone_h - 1)],
                        fill=tint_col,
                    )

                # 컬러 테두리 (점유 존 전체를 하나로 묶음)
                zones = sorted(lbl.zone for lbl in ev.labels)
                rect_y1 = staff_top + min(zones) * zone_h + 1
                rect_y2 = staff_top + (max(zones) + 1) * zone_h - 1
                draw.rectangle(
                    [(ex, rect_y1), (ex + ew - 1, rect_y2)],
                    outline=color, width=2,
                )

                # 음표 글자 (존별, 셀 정중앙 배치)
                for lbl in ev.labels:
                    zy = staff_top + lbl.zone * zone_h
                    bb = draw.textbbox((0, 0), lbl.text, font=font)
                    tw, th = bb[2] - bb[0], bb[3] - bb[1]
                    tx = ex + (ew - tw) // 2
                    ty = zy + (zone_h - th) // 2

                    if lbl.has_black:
                        # 검은 건반 → 검은 숫자
                        draw.text((tx, ty), lbl.text, font=font, fill=(15, 15, 15))
                    else:
                        # 흰 건반 → 흰 글자 + 검은 외곽선
                        _outlined_text(draw, (tx, ty), lbl.text,
                                       font=font,
                                       fill=(255, 255, 255),
                                       outline=(15, 15, 15),
                                       stroke=2)

                color_idx += 1

            elif ev.kind == "rest":
                # 빈 셀 (회색 점선 테두리, 중간 존에 표시)
                zy = staff_top + 1 * zone_h  # 중간 존
                draw.rectangle(
                    [(ex + 2, zy + 4), (ex + ew - 2, zy + zone_h - 4)],
                    outline=(180, 180, 180), width=1,
                )
                color_idx += 1

            ex += ew  # 다음 이벤트 x 위치로 전진

        # 셈여림 (마디 상단 좌측)
        if dyn_events:
            dyn_str = "  ".join(e.dynamic_text for e in dyn_events)
            draw.text((cur_x + MEASURE_PAD, staff_top - HEADER_H + 4),
                      dyn_str, font=font_sm, fill=(180, 50, 50))

        cur_x += mw

    # 마지막 세로선
    draw.line([(cur_x, staff_top), (cur_x, staff_bot)],
              fill=(30, 30, 30), width=2)

    return img


# ── Grand Staff 지원 ──────────────────────────────────────────────────────────

STAFF_GAP = 20  # 두 보표 사이 여백 (px)


def split_staves(tokens: list) -> Tuple[list, list]:
    """flat grand staff 토큰 → (treble_tokens, bass_tokens) 분리.

    각 마디 내에서 staff-bass 이전 = treble, 이후 = bass.
    barline 토큰은 양쪽 모두에 복사한다.
    key-/time- 헤더 토큰은 bass에도 복사해 박자표가 올바르게 표시되도록 한다.
    """
    treble: list = []
    bass:   list = []
    in_bass = False

    for tok in tokens:
        if tok == 'staff-bass':
            in_bass = True
            continue
        if tok in ('barline', 'barline-double', 'barline-final',
                   'barline-start-repeat', 'barline-end-repeat'):
            in_bass = False
            treble.append(tok)
            bass.append(tok)
        elif tok in ('<SOS>', '<EOS>', '<PAD>', '<UNK>'):
            treble.append(tok)
        elif tok.startswith('key-') or tok.startswith('time-'):
            # 박자표·조표는 두 보표 모두에 적용
            treble.append(tok)
            bass.append(tok)
        elif in_bass:
            bass.append(tok)
        else:
            treble.append(tok)

    return treble, bass


def render_grand_staff(
        treble_hdr: Header, treble_measures: List[List[NoteEvent]],
        bass_hdr:   Header, bass_measures:   List[List[NoteEvent]],
        cell_w: int  = DEFAULT_CELL_W,
        zone_h: int  = DEFAULT_ZONE_H,
        font_size: int = DEFAULT_FONT_SZ) -> Image.Image:
    """두 보표(treble 위, bass 아래)를 수직으로 합성한 이미지를 반환."""
    img_t = render(treble_hdr, treble_measures, cell_w, zone_h, font_size)
    img_b = render(bass_hdr,   bass_measures,   cell_w, zone_h, font_size)

    w = max(img_t.width, img_b.width)

    def _pad(img: Image.Image) -> Image.Image:
        if img.width == w:
            return img
        canvas = Image.new("RGB", (w, img.height), (255, 255, 255))
        canvas.paste(img, (0, 0))
        return canvas

    img_t = _pad(img_t)
    img_b = _pad(img_b)

    combined = Image.new("RGB", (w, img_t.height + STAFF_GAP + img_b.height), (255, 255, 255))
    combined.paste(img_t, (0, 0))
    combined.paste(img_b, (0, img_t.height + STAFF_GAP))
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="DeepScore JSON 토큰 파일을 커스텀 악보 이미지로 렌더링합니다."
    )
    ap.add_argument("inputs",    nargs="+", help="JSON 파일 경로 (여러 개 가능)")
    ap.add_argument("--out",     default=".",    help="출력 디렉토리 (기본: 현재 디렉토리)")
    ap.add_argument("--show",    action="store_true", help="렌더링 결과를 즉시 화면에 표시")
    ap.add_argument("--cell",    type=int, default=DEFAULT_CELL_W,  help=f"16분음표 셀 너비 px (기본 {DEFAULT_CELL_W})")
    ap.add_argument("--zone-h",  type=int, default=DEFAULT_ZONE_H,  help=f"옥타브 존 높이 px (기본 {DEFAULT_ZONE_H})")
    ap.add_argument("--font-size", type=int, default=DEFAULT_FONT_SZ, help=f"폰트 크기 px (기본 {DEFAULT_FONT_SZ})")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"[건너뜀] 파일 없음: {inp}", file=sys.stderr)
            continue

        data   = json.loads(p.read_text(encoding="utf-8"))
        tokens = data.get("tokens", [])

        if 'staff-bass' in tokens:
            # Grand staff: treble + bass 분리 후 합성
            t_toks, b_toks = split_staves(tokens)
            t_hdr, t_measures = parse_tokens(t_toks)
            b_hdr, b_measures = parse_tokens(b_toks)
            if not t_measures and not b_measures:
                print(f"[건너뜀] 마디를 찾을 수 없음: {inp}", file=sys.stderr)
                continue
            img = render_grand_staff(t_hdr, t_measures, b_hdr, b_measures,
                                     cell_w=args.cell,
                                     zone_h=args.zone_h,
                                     font_size=args.font_size)
        else:
            hdr, measures = parse_tokens(tokens)
            if not measures:
                print(f"[건너뜀] 마디를 찾을 수 없음: {inp}", file=sys.stderr)
                continue
            img = render(hdr, measures,
                         cell_w=args.cell,
                         zone_h=args.zone_h,
                         font_size=args.font_size)

        out_path = out_dir / (p.stem + "_notation.png")
        img.save(str(out_path))
        n_measures = len(t_measures) if 'staff-bass' in tokens else len(measures)
        print(f"저장: {out_path}  ({img.width}×{img.height} px, {n_measures}마디)")

        if args.show:
            img.show()


if __name__ == "__main__":
    main()
