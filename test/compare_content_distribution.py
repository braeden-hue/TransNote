"""라운드3 합성 학습 풀(r3_density_register_clef, 5000장) vs exactPicture(실사 86곡)
라벨의 평균적 콘텐츠 분포 비교. 두 실측 정확도 격차(합성 94.2% vs 실사 클린렌더링 72.2%)가
순수 노이즈 문제가 아니라 콘텐츠 분포(OOD) 차이 때문인지 진단하기 위한 일회성 분석 스크립트.
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYNTH_DIR = ROOT / 'data' / 'local_pools' / 'r3_density_register_clef'
REAL_DIR = ROOT / 'data' / 'local_pools' / 'exactPicture'

_PITCH_RE = re.compile(r'^(?:note|chord)-([A-G])(#{1,2}|b{1,2})?(\d)$')


def load_synth_tokens():
    out = []
    for p in glob.glob(str(SYNTH_DIR / 'num*.json')):
        if p.endswith('_staffs.json'):
            continue
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        out.append(d['tokens'])
    return out


def load_real_tokens():
    out = []
    for song_dir in sorted(d for d in REAL_DIR.iterdir() if d.is_dir()):
        jsons = list(song_dir.glob('*.json'))
        if not jsons:
            continue
        with open(jsons[0], encoding='utf-8') as f:
            d = json.load(f)
        out.append(d['tokens'])
    return out


def analyze(name, all_tokens):
    n_scores = len(all_tokens)
    total_toks = sum(len(t) for t in all_tokens)
    n_notes = sum(sum(1 for x in t if x.startswith('note-')) for t in all_tokens)
    n_chords = sum(sum(1 for x in t if x.startswith('chord-')) for t in all_tokens)
    n_barlines = sum(sum(1 for x in t if x.startswith('barline')) for t in all_tokens)
    n_grand = sum(1 for t in all_tokens if 'staff-bass' in t)

    def presence_rate(prefix_or_tok):
        cnt = sum(1 for t in all_tokens if any(
            x == prefix_or_tok or x.startswith(prefix_or_tok) for x in t))
        return cnt / n_scores * 100

    def per_1000_notes(prefix_or_tok):
        cnt = sum(sum(1 for x in t if x == prefix_or_tok or (isinstance(prefix_or_tok, str) and x.startswith(prefix_or_tok))) for t in all_tokens)
        return cnt / max(n_notes, 1) * 1000

    dur_counter = Counter()
    octave_counter = Counter()
    accidental = Counter()
    key_counter = Counter()
    time_counter = Counter()
    clef_mid_count = 0  # 시작(첫 clef) 이후 등장하는 clef 전환 수
    for t in all_tokens:
        seen_first_clef = False
        for x in t:
            if x.startswith('dur-'):
                dur_counter[x] += 1
            m = _PITCH_RE.match(x)
            if m:
                octave_counter[m.group(3)] += 1
                acc = m.group(2) or ''
                if acc.startswith('#'):
                    accidental['sharp'] += 1
                elif acc.startswith('b'):
                    accidental['flat'] += 1
                else:
                    accidental['natural'] += 1
            if x.startswith('key-'):
                key_counter[x] += 1
            if x.startswith('time-'):
                time_counter[x] += 1
            if x.startswith('clef-'):
                if seen_first_clef:
                    clef_mid_count += 1
                seen_first_clef = True

    print(f"\n{'='*70}\n{name}  (n={n_scores}곡, 총 토큰 {total_toks}, 음표 {n_notes}, 마디 {n_barlines})\n{'='*70}")
    print(f"  대보표 비율            : {n_grand/n_scores*100:.1f}%")
    print(f"  곡당 평균 토큰 수       : {total_toks/n_scores:.1f}")
    print(f"  곡당 평균 음표 수       : {n_notes/n_scores:.1f}")
    print(f"  곡당 평균 마디 수       : {n_barlines/n_scores:.2f}")
    print(f"  화음(chord-) 비율       : {n_chords/max(n_notes,1)*100:.1f}%  (음표 대비)")
    print(f"  --- 곡 단위 등장률(presence rate) ---")
    for label, tok in [
        ('셋잇단음표(tuplet-3-start)', 'tuplet-3-start'),
        ('붙임줄(tie)', 'tie'),
        ('페르마타(fermata)', 'fermata'),
        ('다이나믹(dynamic-)', 'dynamic-'),
        ('헤어핀(hairpin-)', 'hairpin-'),
        ('슬러(slur-)', 'slur-'),
        ('아티큘레이션(artic-)', 'artic-'),
        ('오나먼트(ornament-)', 'ornament-'),
        ('옥타브(ottava-)', 'ottava-'),
        ('반복표(repeat)', 'barline-start-repeat'),
    ]:
        print(f"    {label:28s}: {presence_rate(tok):5.1f}%")
    print(f"  --- 1000음표당 등장 횟수(밀도) ---")
    for label, tok in [
        ('셋잇단음표(tuplet-3-start)', 'tuplet-3-start'),
        ('붙임줄(tie)', 'tie'),
        ('페르마타(fermata)', 'fermata'),
    ]:
        print(f"    {label:28s}: {per_1000_notes(tok):6.1f} / 1000음표")
    print(f"  clef 전환(곡당 평균, 첫 clef 제외): {clef_mid_count/n_scores:.2f}회")
    print(f"  key 서명 분포(top5)     : {key_counter.most_common(5)}")
    print(f"  time 서명 분포          : {sorted(time_counter.items(), key=lambda x:-x[1])}")
    total_pitch = sum(accidental.values())
    if total_pitch:
        print(f"  임시표 비율             : sharp {accidental['sharp']/total_pitch*100:.1f}% / "
              f"flat {accidental['flat']/total_pitch*100:.1f}% / natural {accidental['natural']/total_pitch*100:.1f}%")
    print(f"  옥타브(음역) 분포        : {dict(sorted(octave_counter.items()))}")
    dur_total = sum(dur_counter.values())
    print(f"  duration 분포(%)        :")
    for d, c in sorted(dur_counter.items(), key=lambda x: -x[1]):
        print(f"    {d:12s}: {c/dur_total*100:5.1f}%")

    return {
        'n_scores': n_scores, 'n_notes': n_notes,
        'tuplet_presence': presence_rate('tuplet-3-start'),
        'tuplet_density': per_1000_notes('tuplet-3-start'),
    }


def main():
    synth_tokens = load_synth_tokens()
    real_tokens = load_real_tokens()
    s = analyze('합성(Round3 학습 풀, r3_density_register_clef)', synth_tokens)
    r = analyze('실사(exactPicture, mscz 정답 라벨)', real_tokens)

    print(f"\n{'='*70}\n요약 비교\n{'='*70}")
    print(f"  셋잇단음표 곡당 등장률: 합성 {s['tuplet_presence']:.1f}% vs 실사 {r['tuplet_presence']:.1f}%")
    print(f"  셋잇단음표 1000음표당 밀도: 합성 {s['tuplet_density']:.1f} vs 실사 {r['tuplet_density']:.1f}")


if __name__ == '__main__':
    main()
