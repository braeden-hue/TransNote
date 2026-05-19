#!/usr/bin/env python3
"""
compare_musicxml.py — OMR 인식 결과 vs Ground Truth 비교
사용법:
  python compare_musicxml.py <ground_truth.xml> <recognized.xml>
  python compare_musicxml.py num2_gt.musicxml num2.musicxml
"""

import xml.etree.ElementTree as ET
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ─── 자료구조 ──────────────────────────────────────────────────────────────────

@dataclass
class Note:
    step:     str   # C D E F G A B  (쉼표면 'R')
    octave:   int   # 4 = 중간 옥타브
    alter:    int   # -1=flat  0=natural  1=sharp
    dur_type: str   # whole half quarter eighth sixteenth ...
    dots:     int   # 점음표 개수
    is_rest:  bool
    measure:  int   # 몇 번째 마디

    def pitch_str(self) -> str:
        if self.is_rest: return "rest"
        acc = {-2:"bb", -1:"b", 0:"", 1:"#", 2:"##"}.get(self.alter, "")
        return f"{self.step}{acc}{self.octave}"

    def rhythm_str(self) -> str:
        dot_str = "." * self.dots
        return f"{self.dur_type}{dot_str}"

    def full_str(self) -> str:
        return f"{self.pitch_str()}_{self.rhythm_str()}"


# ─── MusicXML 파서 ─────────────────────────────────────────────────────────────

def parse_musicxml(filepath: str) -> List[Note]:
    """MusicXML 파일에서 음표/쉼표 시퀀스 추출"""
    notes: List[Note] = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        # namespace 처리 (MusicXML 4.0은 namespace 없음, 구버전은 있을 수 있음)
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        measure_num = 0
        for measure in root.iter(f"{ns}measure"):
            measure_num += 1
            for elem in measure:
                if elem.tag != f"{ns}note":
                    continue
                # grace note 제외
                if elem.find(f"{ns}grace") is not None:
                    continue
                # chord 음표 (동시에 울리는 음) — 피치 비교에서 첫 음만 사용
                if elem.find(f"{ns}chord") is not None:
                    continue

                is_rest = elem.find(f"{ns}rest") is not None

                type_elem = elem.find(f"{ns}type")
                dur_type  = type_elem.text.strip() if type_elem is not None else "quarter"
                dots      = len(list(elem.findall(f"{ns}dot")))

                if is_rest:
                    notes.append(Note("R", 0, 0, dur_type, dots, True, measure_num))
                else:
                    pitch = elem.find(f"{ns}pitch")
                    if pitch is None:
                        continue
                    step_e  = pitch.find(f"{ns}step")
                    oct_e   = pitch.find(f"{ns}octave")
                    alter_e = pitch.find(f"{ns}alter")

                    step   = step_e.text.strip()  if step_e  is not None else "C"
                    octave = int(oct_e.text)       if oct_e   is not None else 4
                    alter  = int(float(alter_e.text)) if alter_e is not None else 0

                    notes.append(Note(step, octave, alter, dur_type, dots, False, measure_num))

    except FileNotFoundError:
        print(f"[error] 파일 없음: {filepath}")
    except ET.ParseError as e:
        print(f"[error] XML 파싱 실패 ({filepath}): {e}")
    return notes


# ─── 마디별 분리 ───────────────────────────────────────────────────────────────

def split_by_measure(notes: List[Note]):
    measures = {}
    for n in notes:
        measures.setdefault(n.measure, []).append(n)
    return measures


# ─── 비교 ─────────────────────────────────────────────────────────────────────

def compare(gt: List[Note], rec: List[Note]) -> dict:
    n = min(len(gt), len(rec))
    if n == 0:
        return {}

    exact = pitch = rhythm = rest_ok = 0
    mismatches = []

    for i, (g, r) in enumerate(zip(gt[:n], rec[:n])):
        p_ok = (g.is_rest == r.is_rest and
                g.step == r.step and g.octave == r.octave and g.alter == r.alter)
        r_ok = (g.dur_type == r.dur_type and g.dots == r.dots)

        if p_ok: pitch += 1
        if r_ok: rhythm += 1
        if p_ok and r_ok: exact += 1
        if g.is_rest and r.is_rest: rest_ok += 1

        if not (p_ok and r_ok):
            mismatches.append((i + 1, g.measure, g.full_str(), r.full_str()))

    return {
        "gt_total":    len(gt),
        "rec_total":   len(rec),
        "compare_n":   n,
        "exact":       exact,
        "pitch":       pitch,
        "rhythm":      rhythm,
        "rest_ok":     rest_ok,
        "exact_pct":   exact  / n * 100,
        "pitch_pct":   pitch  / n * 100,
        "rhythm_pct":  rhythm / n * 100,
        "count_pct":   len(rec) / len(gt) * 100,
        "mismatches":  mismatches,
    }


# ─── 보고서 출력 ───────────────────────────────────────────────────────────────

def grade(pct: float) -> str:
    if pct >= 85: return "✅ 우수"
    if pct >= 65: return "⚠  보통"
    if pct >= 40: return "❌ 미흡"
    return "💥 불량"

def print_report(gt_path: str, rec_path: str, r: dict):
    W = 54
    bar = "─" * W
    print(bar)
    print(f"  GT  : {gt_path}")
    print(f"  인식: {rec_path}")
    print(bar)
    print(f"  음표 수   GT={r['gt_total']}  인식={r['rec_total']}  "
          f"비율={r['count_pct']:.0f}%")
    print(f"  비교 대상 {r['compare_n']}개")
    print(bar)
    print(f"  완전 일치 (피치+리듬)  {r['exact']:3d}/{r['compare_n']}  "
          f"{r['exact_pct']:5.1f}%  {grade(r['exact_pct'])}")
    print(f"  피치만 일치            {r['pitch']:3d}/{r['compare_n']}  "
          f"{r['pitch_pct']:5.1f}%")
    print(f"  리듬만 일치            {r['rhythm']:3d}/{r['compare_n']}  "
          f"{r['rhythm_pct']:5.1f}%")
    print(bar)

    if r["mismatches"]:
        print(f"  불일치 목록 (최대 15개):")
        for idx, meas, g_str, r_str in r["mismatches"][:15]:
            print(f"    #{idx:3d} [마디{meas}]  GT={g_str:<18}  인식={r_str}")
        if len(r["mismatches"]) > 15:
            print(f"    ... 외 {len(r['mismatches']) - 15}개")
    else:
        print("  ✅ 불일치 없음")
    print(bar)

    # 마디별 요약
    gt_notes  = []   # 이미 parse된 목록을 못 넘겨서 재출력은 생략
    print(f"  종합: {grade(r['exact_pct'])}  ({r['exact_pct']:.1f}%)")
    print(bar)


# ─── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    gt_path  = sys.argv[1]
    rec_path = sys.argv[2]

    gt_notes  = parse_musicxml(gt_path)
    rec_notes = parse_musicxml(rec_path)

    if not gt_notes:
        print("[error] Ground truth 음표를 읽지 못했습니다.")
        sys.exit(1)
    if not rec_notes:
        print("[error] 인식 결과 음표를 읽지 못했습니다.")
        sys.exit(1)

    result = compare(gt_notes, rec_notes)
    print_report(gt_path, rec_path, result)

    # 마디별 정확도
    gt_m  = split_by_measure(gt_notes)
    rec_m = split_by_measure(rec_notes)
    all_m = sorted(set(gt_m) | set(rec_m))
    if len(all_m) > 1:
        print("  마디별 완전 일치율:")
        for m in all_m:
            gm = gt_m.get(m, [])
            rm = rec_m.get(m, [])
            if not gm: continue
            sub = compare(gm, rm)
            bar_str = "█" * int(sub.get("exact_pct", 0) / 10)
            print(f"    마디 {m:2d}: {sub.get('exact_pct', 0):5.1f}%  {bar_str}")
        print("─" * 54)

    sys.exit(0 if result.get("exact_pct", 0) >= 65 else 1)


if __name__ == "__main__":
    main()
