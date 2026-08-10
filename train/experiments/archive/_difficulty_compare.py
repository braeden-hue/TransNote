"""클래식 실사(100곡)/뉴에이지(20곡)/r7b 합성 학습데이터의 콘텐츠 난이도를 같은 지표로
측정해서 비교한다. 이미지가 아니라 GT 토큰 시퀀스만 보고, 카메라 노이즈와 무관한
"악보 자체의 어려움"만 격리해서 잰다.

지표:
  - accidental_rate: 임시표(#,b) 있는 음표 비율
  - chord_rate: 화음(2음+)에 속하는 음표 비율
  - chord_mean_size: 화음일 때 평균 음표 수
  - fast_dur_rate: dur-1/16 이하(1/16, 1/32) 비율 -- 빠른 리듬
  - register_span: 곡 내 최고음-최저음 반음 차이(레저선 부담 proxy)
  - tokens_per_measure: 마디당 토�큰 수(밀도)
"""
import glob
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTE_RE = re.compile(r'^([A-G])(#|b)?(-?\d+)$')

PITCH_CLASS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def note_to_semitone(pitch_str):
    m = NOTE_RE.match(pitch_str)
    if not m:
        return None
    letter, acc, octave = m.group(1), m.group(2), int(m.group(3))
    val = PITCH_CLASS[letter] + (1 if acc == '#' else -1 if acc == 'b' else 0)
    return val + octave * 12


def analyze(tokens):
    tokens = [t for t in tokens if t not in ('<SOS>', '<EOS>', '<PAD>')]
    n_notes = 0
    n_acc = 0
    n_chord_notes = 0
    chord_sizes = []
    n_fast = 0
    n_dur = 0
    semitones = []
    n_bar = 0

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith('note-'):
            n_notes += 1
            pitch_str = t[len('note-'):]
            if '#' in pitch_str or 'b' in pitch_str:
                n_acc += 1
            semi = note_to_semitone(pitch_str)
            if semi is not None:
                semitones.append(semi)
            # note- 바로 다음은 항상 dur- (화음이어도 대표 duration 1개만 표기됨,
            # chord- 토큰엔 duration이 따로 없음) -- 여기서 직접 카운트해야 함
            # (elif dur- 분기는 i가 j로 건너뛰면서 이 토큰을 다시 안 지나감).
            if i + 1 < len(tokens) and tokens[i + 1].startswith('dur-'):
                n_dur += 1
                val = tokens[i + 1][len('dur-'):]
                if val in ('1/16', '1/32', '3/16', '3/32'):
                    n_fast += 1
            size = 1
            j = i + 2
            while j < len(tokens) and tokens[j].startswith('chord-'):
                n_notes += 1
                cp = tokens[j][len('chord-'):]
                if '#' in cp or 'b' in cp:
                    n_acc += 1
                csemi = note_to_semitone(cp)
                if csemi is not None:
                    semitones.append(csemi)
                size += 1
                j += 1
            if size >= 2:
                n_chord_notes += size
                chord_sizes.append(size)
            i = j
        elif t.startswith('dur-'):
            n_dur += 1
            val = t[len('dur-'):]
            if val in ('1/16', '1/32', '3/16', '3/32'):
                n_fast += 1
            i += 1
        elif t.startswith('barline'):
            n_bar += 1
            i += 1
        else:
            i += 1

    n_bar = max(n_bar, 1)
    return {
        'accidental_rate': n_acc / max(n_notes, 1),
        'chord_rate': n_chord_notes / max(n_notes, 1),
        'chord_mean_size': st.mean(chord_sizes) if chord_sizes else 0.0,
        'fast_dur_rate': n_fast / max(n_dur, 1),
        'register_span': (max(semitones) - min(semitones)) if semitones else 0,
        'tokens_per_measure': len(tokens) / n_bar,
    }


def summarize(name, records):
    if not records:
        print(f'{name}: 데이터 없음')
        return
    keys = [k for k in records[0].keys() if k != '_name']
    print(f'\n=== {name} (n={len(records)}) ===')
    for k in keys:
        vals = [r[k] for r in records]
        print(f'  {k:20s} 평균={st.mean(vals):6.3f}  중앙값={st.median(vals):6.3f}  표준편차={st.pstdev(vals):6.3f}')


def main():
    gt_dir = HERE / 'data' / 'local_pools' / 'exactpicture_test_full'

    classical_records = []
    newage_records = []
    for p in sorted(gt_dir.glob('*.json')):
        name = p.stem
        try:
            tokens = json.loads(p.read_text(encoding='utf-8'))['tokens']
        except Exception:
            continue
        rec = analyze(tokens)
        rec['_name'] = name
        if name.startswith('newage'):
            newage_records.append(rec)
        else:
            classical_records.append(rec)

    synth_dir = HERE.parent / 'round3train' / 'data' / 'r7_l4_major_synth' if False else None
    # pod 경로 대응: 로컬에는 없을 수 있으므로 존재 확인 후 스킵
    synth_records = []
    synth_glob_local = HERE / 'data' / 'r7_l4_major_synth'
    search_dirs = [synth_glob_local, Path('/workspace/data/r7_l4_major_synth')]
    for d in search_dirs:
        if d.exists():
            for p in sorted(d.glob('*.json')):
                try:
                    tokens = json.loads(p.read_text(encoding='utf-8'))['tokens']
                except Exception:
                    continue
                synth_records.append(analyze(tokens))
            break

    summarize('클래식 실사(exactpicture_test_full, 100곡)', classical_records)
    summarize('뉴에이지(exactpicture_test_full, 20곡)', newage_records)
    summarize('r7b 합성(r7_l4_major_synth)', synth_records)

    def score(r):
        return r['accidental_rate'] + r['chord_rate'] + r['fast_dur_rate']

    print('\n=== 클래식 100곡 난이도 순위(쉬운 순) ===')
    classical_sorted = sorted(classical_records, key=score)
    for rank, r in enumerate(classical_sorted, 1):
        print(f'  {rank:3d}. {r["_name"]:25s} score={score(r):.3f}')

    print('\n=== 뉴에이지 20곡 난이도 순위(쉬운 순) ===')
    newage_sorted = sorted(newage_records, key=score)
    for rank, r in enumerate(newage_sorted, 1):
        print(f'  {rank:3d}. {r["_name"]:25s} score={score(r):.3f}')


if __name__ == '__main__':
    main()
