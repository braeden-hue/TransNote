"""GT/예측 토큰 시퀀스를 마디별(treble/bass 별도)로 나눠 총 길이가 박자표와 맞는지
검산. 안 맞는 마디를 전부 표시 -- 토큰화/트림 과정의 버그와 모델이 만들어낸 리듬적으로
불가능한(마디 길이가 안 맞는) 출력을 구분해서 잡아낸다.
"""
import sys
from fractions import Fraction

DUR_RE = None


def dur_token_to_ql(tok: str) -> float:
    """'1/8' 같은 duration 토큰(온음표 대비 분수) -> quarterLength(4분음표=1.0)."""
    n, d = tok.split('/')
    return float(Fraction(int(n), int(d)) * 4)


def split_into_systems(tokens):
    """토큰 스트림을 [(treble_measures, bass_measures, time_num, time_den)] 리스트로 분해.
    구조: clef-G key-X time-N/D [treble] staff-bass [clef-F 선택] [bass] barline ...
    """
    idx = 0
    time_num, time_den = 4, 4
    while idx < len(tokens):
        t = tokens[idx]
        if t.startswith('time-'):
            n, d = t[len('time-'):].split('/')
            time_num, time_den = int(n), int(d)
        if t.startswith('clef-') or t.startswith('key-') or t.startswith('time-'):
            idx += 1
            continue
        break
    body = tokens[idx:]

    # 베이스가 온마디 쉼표면 mscz_to_label.py가 staff-bass 자체를 생략한다(빈 마디를
    # 명시적으로 표시하지 않음) -- 그래서 "barline"이 오면 staff-bass를 못 봤어도 이번
    # 마디는 끝난 것으로 처리해야 한다(그렇지 않으면 다음 마디의 treble 내용이 앞 마디에
    # 이어붙어 마디 경계가 전부 밀림, 2026-07-28 확인된 파싱 버그).
    treble_measures, bass_measures = [], []
    cur_t, cur_b = [], []
    saw_staff_bass = False
    for t in body:
        if t == 'staff-bass':
            treble_measures.append(cur_t)
            cur_t = []
            saw_staff_bass = True
            continue
        if t.startswith('barline'):
            if not saw_staff_bass:
                treble_measures.append(cur_t)
                cur_t = []
                bass_measures.append([])
            else:
                bass_measures.append(cur_b)
                cur_b = []
            saw_staff_bass = False
            continue
        if t.startswith('clef-'):
            continue  # 중간 클레프 변경은 길이 계산에 영향 없음
        if saw_staff_bass:
            cur_b.append(t)
        else:
            cur_t.append(t)
    if cur_t:
        treble_measures.append(cur_t)
    if cur_b:
        bass_measures.append(cur_b)
    return treble_measures, bass_measures, time_num, time_den


def measure_ql(measure_tokens) -> float:
    """한 마디(치 또는 베이스) 토큰에서 실제 소리나는 총 길이 계산.
    note- 토큰은 relabel()이 note-{pitch}+dur-{dur}로 분리해놨으므로 다음 dur- 토큰이
    길이를 결정. rest- 토큰은 mscz_to_label.py가 애초에 자체 완결형(rest-{fraction},
    분리 안 함)으로 내보내므로 토큰 자체에서 길이를 바로 파싱해야 함(별도 dur- 토큰 없음).
    chord-*는 같은 시점이라 길이에 안 더함(화음의 추가음)."""
    total = 0.0
    i = 0
    while i < len(measure_tokens):
        t = measure_tokens[i]
        if t.startswith('rest-'):
            total += dur_token_to_ql(t[len('rest-'):])
            i += 1
            continue
        if t.startswith('note-'):
            if i + 1 < len(measure_tokens) and measure_tokens[i + 1].startswith('dur-'):
                total += dur_token_to_ql(measure_tokens[i + 1][len('dur-'):])
                i += 2
                continue
        i += 1
    return total


def check(name, tokens, label):
    tokens = [t for t in tokens if t not in ('<SOS>', '<EOS>', '<PAD>')]
    treble, bass, time_num, time_den = split_into_systems(tokens)
    # quarterLength 기준 마디당 기대 길이 -- 4분음표=1.0이므로 분모가 8이면 박자 하나가
    # 0.5, 4면 1.0이 되어(예: 3/8 -> 1.5, 3/4 -> 3.0) 단순히 분자와만 비교하면 8분음표
    # 계열 박자표에서 전부 오탐(false positive)이 뜬다(2026-07-28 3/8 데이터 준비 중 발견).
    expected_ql = time_num * 4.0 / time_den
    problems = []
    for mi, m in enumerate(treble):
        ql = measure_ql(m)
        if abs(ql - expected_ql) > 1e-6:
            problems.append(f"treble 마디{mi+1}: {ql}박(quarterLength 기준) (기대 {expected_ql}) -- {' '.join(m)}")
    for mi, m in enumerate(bass):
        ql = measure_ql(m)
        if abs(ql - expected_ql) > 1e-6:
            problems.append(f"bass 마디{mi+1}: {ql}박(quarterLength 기준) (기대 {expected_ql}) -- {' '.join(m)}")

    print(f"[{name}] {label}: treble {len(treble)}마디, bass {len(bass)}마디, "
          f"박자표 {time_num}/{time_den} -- {'문제 없음' if not problems else f'{len(problems)}건 불일치'}")
    for p in problems:
        print(f"    ! {p}")


if __name__ == '__main__':
    import json
    import glob
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'test')
    import re
    NOTE_DUR_RE = re.compile(r"^note-(.+)-(\d+/\d+)$")

    def split_note_token(tok):
        m = NOTE_DUR_RE.match(tok)
        if not m:
            return [tok]
        return [f"note-{m.group(1)}", f"dur-{m.group(2)}"]

    def relabel(tokens):
        out = []
        for t in tokens:
            out.extend(split_note_token(t))
        return out

    for folder in sorted(d for d in glob.glob(os.path.join(TEST_DIR, 'chop*')) if os.path.isdir(d)):
        name = os.path.basename(folder)
        json_paths = glob.glob(os.path.join(folder, '*_new.json'))
        if not json_paths:
            continue
        with open(json_paths[0], encoding='utf-8') as f:
            raw = json.load(f)['tokens']
        gt = relabel(raw)
        check(name, gt, 'GT(mscz 추출)')

    # eval_mscz_clean.py 최근 실행 콘솔 출력에서 그대로 옮긴 PRED 토큰(모델이 이미
    # note-{pitch}/dur-{dur} 분리 형태로 출력하므로 relabel() 불필요) -- chop64_3_21
    # "마디 수 오류" 등 예측쪽 리듬 붕괴를 직접 검산하기 위함.
    PRED_TOKENS = {
        'chop39_3_9': "clef-G key-F time-3/4 dynamic-ff tuplet-3-start note-C5 dur-1/8 note-Db5 dur-1/8 note-C5 dur-1/8 tuplet-3-end note-B4 dur-1/8 note-C5 dur-1/8 note-C5 dur-1/8 note-C#5 dur-1/8 note-D5 dur-1/8 note-D5 dur-1/8 note-C#5 dur-1/8 note-D5 dur-1/8 note-C#5 dur-1/8 staff-bass clef-F note-C3 dur-1/4 chord-G3 chord-B3 rest-1/4 barline-final",
        'chop64_2_33': "clef-G key-E time-3/4 note-G#5 dur-1/8 note-A#5 dur-1/8 note-F#5 dur-1/8 note-A4 dur-1/8 note-F#5 dur-1/8 note-F#5 dur-1/8 note-E5 dur-1/8 staff-bass clef-F note-Bb2 dur-1/4 note-G#3 dur-1/4 chord-B3 note-C#3 dur-1/4 note-G#3 dur-1/4 chord-B3 barline note-C#5 dur-1/8 note-F#5 dur-1/8 note-D#5 dur-1/8 note-A#3 dur-1/8 note-A#3 dur-1/8 chord-A#3 note-F#3 dur-1/4 chord-B3 note-A2 dur-1/4 note-E3 dur-1/4 chord-A#3 barline-final",
        'chop64_3_16': "clef-G key-Ab time-3/4 note-Ab4 dur-1/8 note-Bb4 dur-1/8 note-Db5 dur-1/8 note-Eb5 dur-1/8 staff-bass clef-F note-Eb3 dur-1/4 note-Bb3 dur-1/4 chord-Bb3 barline note-Ab4 dur-1/8 note-Ab4 dur-1/8 staff-bass note-Eb3 dur-1/4 chord-Ab3 chord-Ab3 note-Ab2 dur-1/4 note-Eb3 dur-1/4 chord-Ab3 chord-A3 barline note-D5 dur-1/8 note-Eb4 dur-1/8 note-Db5 dur-1/8 note-F5 dur-3/16 note-D5 dur-1/8 note-D5 dur-1/8 staff-bass clef-F note-Eb3 dur-1/4 chord-Ab3 chord-A3 note-Ab2 dur-1/4 chord-Ab3 chord-Ab3 chord-Ab3 note-Ab2 dur-1/4 note-Eb3 dur-1/4 chord-Ab3 chord-A3 barline-final",
        'chop64_3_21': "clef-G key-Ab time-3/4 note-G5 dur-1/8 note-E5 dur-1/8 note-D5 dur-1/8 staff-bass clef-F note-Bb2 dur-1/4 note-Eb3 dur-1/4 chord-G3 chord-Bb3 barline note-C5 dur-1/8 note-Db5 dur-1/8 note-G5 dur-1/8 note-E5 dur-1/8 staff-bass note-Eb3 dur-1/4 chord-E3 chord-Bb3 note-Eb2 dur-1/4 note-Eb3 dur-1/4 chord-G3 chord-Bb3 note-Bb2 dur-1/4 note-Eb3 dur-1/4 chord-G3 chord-Bb3 barline note-Db5 dur-1/8 note-Db5 dur-1/8 note-Db5 dur-1/8 note-Eb5 dur-1/8 note-C5 dur-1/8 note-Eb5 dur-1/8 note-Db5 dur-1/8 note-Db5 dur-1/8 note-Db5 dur-1/8 staff-bass clef-F note-Ab2 dur-1/8 staff-bass clef-F note-Bb2 dur-1/4 chord-Ab3 chord-A3 fermata staff-bass clef-F note-Bb2 dur-1/4 note-Eb3 dur-1/4 chord-A3 chord-Bb3 chord-B3 note-E2 dur-1/4 note-Eb3 dur-1/4 chord-A3 chord-Bb3 chord-B3 note-E2 dur-1/4 note-Eb3 dur-1/4 note-Eb3 dur-1/4 note-Eb3 dur-1/4 note-Eb3 dur-1/4 chord-G3 chord-Bb3 chord-A3 chord-Bb3 chord-A3 barline-final note-Db5 dur-1/8 chord-A3 barline-final",
    }
    print()
    for name, pred_str in PRED_TOKENS.items():
        check(name, pred_str.split(), 'PRED(모델출력)')
