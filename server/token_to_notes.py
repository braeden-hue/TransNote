"""
server/token_to_notes.py — DeepScore 토큰 리스트를 webpage/js/app.js가 기대하는
{notes: [...]} / {staves: [...]}  JSON 스키마로 변환한다.

문법 요약(train/mscz_to_tokens.py 기준)은 그 파일 docstring 참고. (Flutter 시절
lib/services/omr_token_parser.dart의 Python 포트였으나 Flutter 트랙 자체는 폐기됨.)

pitch 표기: 웹 쪽(webpage/js/samples.js)의 NOTE_NAMES/BLACK_LABEL이 샤프(#)
표기만 이해하므로(예: 'C#4' -> '1'), 모델이 내는 플랫(Db/Eb/Gb/Ab/Bb) 표기는 여기서
동일 음의 샤프 표기로 정규화한다.

붙임줄(tie) 병합 로직(tie_active/pending_carry_duration)은 2026-08-10 추가.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, TypedDict

_FLAT_TO_SHARP = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
}


def _normalize_pitch(pitch: str) -> str:
    """'Db4' -> 'C#4' 처럼 옥타브 숫자를 보존한 채 이름부만 정규화."""
    if not pitch:
        return pitch
    name, octave = pitch[:-1], pitch[-1]
    return _FLAT_TO_SHARP.get(name, name) + octave


class ScoreNote(TypedDict, total=False):
    pitch: str
    duration: float
    beat: int
    chordNotes: List[str]
    isRest: bool
    dynamicMark: Optional[str]
    repeatMark: Optional[str]
    tuplet: bool  # 3연음(셋잇단음표) 그룹의 일원이면 True — 항상 3개가 연속으로 옴


def _fraction_to_quarters(frac: str) -> float:
    """'1/16' -> 0.25 (1.0 = 4분음표, samples.js note.duration과 동일 단위)."""
    parts = frac.split('/')
    if len(parts) != 2:
        return 1.0
    try:
        n, d = float(parts[0]), float(parts[1])
    except ValueError:
        return 1.0
    if d == 0:
        return 1.0
    return n / d * 4.0


def tokens_to_score(tokens: List[str]) -> Dict:
    """토큰 리스트 -> {'treble': [...], 'bass': [...], 'timeSignature': [n, d], 'clefs': [...]}.

    'treble'/'bass'는 실제 음자리표와 무관하게 "위 보표"/"아래 보표"를 가리키는 위치
    라벨(기존 스키마 그대로 유지 — 프론트가 이 두 키로 접근하는 곳이 많아서 이름 자체는
    안 바꿈). 실제 음자리표는 별도 'clefs': [위 보표 clef, 아래 보표 clef] 필드로 반환한다
    ('treble' | 'bass'). 예전엔 이 값을 안 보고 무조건 위=treble/아래=bass로 가정했는데,
    편곡에 따라 아래쪽 보표도 치클렙(G)으로 표기되는 실제 사례가 있어(2026-08-03,
    mscz_to_tokens.py 쪽엔 이미 반영돼 있었음) 토큰이 실제로 알려주는 clef-G/clef-F를
    읽어서 채운다.
    """
    treble: List[ScoreNote] = []
    bass: List[ScoreNote] = []
    time_sig = [4, 4]
    clefs = ['treble', 'bass']  # [위 보표, 아래 보표] — clef- 토큰을 못 만나면 기존 기본값 유지

    on_bass = False
    # 마디 시작(barline)부터 지금까지 실제로 지난 길이(4분음표=1.0 단위) 누적값.
    # 예전엔 정수 "beat 슬롯" 카운터(treble_beat/bass_beat, 1~4)를 썼는데, 한 음표가
    # 몇 분음표든 상관없이 최소 1슬롯을 소비하는 것으로 반올림해서(아래 옛 advance
    # 계산 참고) 8분음표보다 빠른 음이 여러 개 나오면 실제 마디 길이보다 훨씬 빨리
    # 마디가 끝난 것처럼 잘못 셌다(예: 2/4 박자에서 16분음표 8개=진짜 한 마디인데
    # beat이 4개마다 한 번씩 1로 리셋돼 마디가 2개가 아니라 4개로 보임, 2026-08-17
    # 발견). 실제 duration을 그대로 누적하고, 표시용 beat는 그 누적값에서 정수부만
    # 뽑아 time_sig[0]로 나눈 나머지로 계산한다.
    treble_pos = 0.0
    bass_pos = 0.0
    treble_measure_start = 0  # 이번 마디 들어서 treble에 처음 push된 인덱스(마디 경계 보정용)
    bass_measure_start = 0
    start_repeat_pending: Optional[str] = None

    pending_pitch: Optional[str] = None
    pending_duration = 1.0
    pending_carry_duration = 0.0  # 붙임줄(tie)로 앞 음표에서 넘어온 지속시간 — 병합 시 최종 duration에 더해짐
    pending_chord: List[str] = []
    pending_dynamic: Optional[str] = None
    has_pending = False
    tie_active = False  # 방금 'tie' 토큰을 봤음 — 바로 다음 음표가 같은 피치면 합침
    # 'tuplet-3-start'~'tuplet-3-end' 구간 — mscz_to_tokens.py의 관례상 이 구간의 음표는
    # 항상 3개, 각각 인쇄 표기는 'dur-1/8'이지만 실제로는 8분음표 2개(=4분음표 1개) 분량을
    # 셋이 나눠 갖는다(3:2 셋잇단음표). 그래서 실제 duration은 인쇄값의 2/3로 보정해야
    # beat 진행/마디 폭 계산이 맞고, note에 tuplet 플래그를 남겨야 프론트(notation.js)가
    # VexFlow Tuplet(3연음 괄호 표기)으로 묶어 그릴 수 있다.
    tuplet_active = False

    def push(pitch: str, duration: float, chord: Optional[List[str]] = None,
             is_rest: bool = False, dynamic_mark: Optional[str] = None) -> None:
        nonlocal start_repeat_pending, treble_pos, bass_pos
        target = bass if on_bass else treble
        pos = bass_pos if on_bass else treble_pos
        if tuplet_active:
            duration = duration * 2 / 3
        # 1e-9 여유: 0.25를 네 번 더하면 부동소수점 오차로 0.9999999...가 되는 경우가
        # 있어, 그 상태로 int()를 취하면 마디 경계에서 beat이 한 박 밀린다.
        beat = int(pos + 1e-9) % max(1, time_sig[0]) + 1
        note: ScoreNote = {
            'pitch': _normalize_pitch(pitch),
            'duration': duration,
            'beat': beat,
        }
        if tuplet_active:
            note['tuplet'] = True
        if chord:
            note['chordNotes'] = [_normalize_pitch(c) for c in chord]
        if is_rest:
            note['isRest'] = True
        if dynamic_mark:
            note['dynamicMark'] = dynamic_mark
        if start_repeat_pending:
            note['repeatMark'] = start_repeat_pending
        target.append(note)
        start_repeat_pending = None
        if on_bass:
            bass_pos += duration
        else:
            treble_pos += duration

    def finalize_pending() -> None:
        nonlocal pending_pitch, pending_duration, pending_carry_duration, pending_chord, pending_dynamic, has_pending, tie_active
        if not has_pending:
            return
        push(pending_pitch or '', pending_duration + pending_carry_duration, chord=list(pending_chord),
             dynamic_mark=pending_dynamic)
        pending_pitch = None
        pending_duration = 1.0
        pending_carry_duration = 0.0
        pending_chord = []
        pending_dynamic = None
        has_pending = False
        tie_active = False

    def mark_end_repeat() -> None:
        target = bass if on_bass else treble
        if not target:
            return
        target[-1]['repeatMark'] = 'end-repeat'

    def snap_measure_to_time_sig() -> None:
        """barline에서 마디를 넘기기 직전, 치/베이스 각각 이번 마디에 쌓인 실제 길이(pos)가
        확정된 박자표 기대 총합과 다르면 그 마디 안의 모든 음표 duration을 (기대값/실제합)
        비율로 비례 축소·확대해서 정확히 맞춘다(모델 재디코딩 없이, 이미 나온 토큰들의 길이만
        사후 정렬 -- 2026-08-24, 두 손 타이밍이 마디를 넘어가며 계속 어긋나던 문제 대응).
        한 마디의 오차가 다음 마디로 전파되지 않게 막는 게 목적.

        처음엔 "마지막 음표 하나만 조정"하는 방식으로 시작했는데, 실사 10곡(newage21~30)
        검증에서 초과분이 마지막 음표 자체 길이보다 큰 경우(예: 4/4박 마디에 4.5박어치가
        나온 경우, 마지막 16분음표 하나로는 0.5박을 다 못 줄임) 절반 가까이(4/10곡)
        완전히 안 맞고 남는 걸 확인 -- 비례 축소는 초과분 크기와 무관하게 항상 정확히
        맞춰지므로(단, 그 마디의 모든 음표 길이가 조금씩 달라짐) 이 방식으로 교체.

        빈 마디(그 성부에 음표가 하나도 없음)는 보정할 대상이 없어 건너뛴다."""
        expected = time_sig[0] * 4.0 / max(1, time_sig[1])
        for notes, pos, start in (
            (treble, treble_pos, treble_measure_start),
            (bass, bass_pos, bass_measure_start),
        ):
            measure_notes = notes[start:]
            if not measure_notes or pos <= 0 or abs(pos - expected) < 1e-6:
                continue
            scale = expected / pos
            for note in measure_notes:
                note['duration'] *= scale

    for tok in tokens:
        if tok.startswith('time-'):
            parts = tok[5:].split('/')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                time_sig = [int(parts[0]), int(parts[1])]
        elif tok.startswith('clef-'):
            clefs[1 if on_bass else 0] = 'bass' if tok[5:] == 'F' else 'treble'
        elif tok.startswith('key-'):
            pass  # 무시 — 조표는 커스텀 표기에 안 씀
        elif tok == 'staff-bass':
            finalize_pending()
            on_bass = True
        elif tok.startswith('note-'):
            new_pitch = tok[5:]
            # 붙임줄(tie) 병합: 방금 'tie' 토큰을 봤고(tie_active) 아직 push 안 된 이전
            # 음표(has_pending)가 있고 피치가 같으면, 새 음표를 별도로 밀어내지 않고
            # 이전 음표의 지속시간에 이어 붙여서 최종적으로 "한 음"으로 재생되게 한다
            # (커스텀 악보 규칙). has_pending이 이미 False라는 건 그 사이 staff-bass/
            # barline/rest가 끼어서 finalize된 것 — 즉 마디를 넘거나 보표를 바꾸는 tie는
            # 자연스럽게 이 분기를 타지 않고 기존처럼 두 음표로 남는다(스코프 밖, 안전한 폴백).
            if (tie_active and has_pending
                    and _normalize_pitch(pending_pitch or '') == _normalize_pitch(new_pitch)):
                pending_carry_duration += pending_duration
                pending_duration = 1.0
                pending_chord = []
                tie_active = False
            else:
                finalize_pending()
                pending_pitch = new_pitch
                pending_duration = 1.0
                pending_carry_duration = 0.0
                has_pending = True
        elif tok.startswith('dur-'):
            if has_pending:
                pending_duration = _fraction_to_quarters(tok[4:])
        elif tok.startswith('chord-'):
            if has_pending:
                pending_chord.append(tok[6:])
        elif tok.startswith('rest-'):
            finalize_pending()
            push('', _fraction_to_quarters(tok[5:]), is_rest=True)
        elif tok.startswith('dynamic-'):
            if has_pending:
                pending_dynamic = tok[8:]
        elif tok == 'barline-start-repeat':
            finalize_pending()
            start_repeat_pending = 'start-repeat'
        elif tok in ('barline', 'barline-double', 'barline-final', 'barline-end-repeat'):
            finalize_pending()
            if tok == 'barline-end-repeat':
                mark_end_repeat()
            snap_measure_to_time_sig()
            treble_pos = 0.0
            bass_pos = 0.0
            treble_measure_start = len(treble)
            bass_measure_start = len(bass)
            on_bass = False
        elif tok == 'tie':
            tie_active = True
        elif tok == 'tuplet-3-start':
            finalize_pending()
            tuplet_active = True
        elif tok == 'tuplet-3-end':
            finalize_pending()  # 마지막 3연음 음표를 tuplet_active=True 상태로 먼저 flush
            tuplet_active = False
        # artic-*/fermata/ornament-*/slur-*/trill-*/ottava-*/hairpin-*: 무시.

    finalize_pending()

    return {'treble': treble, 'bass': bass, 'timeSignature': time_sig, 'clefs': clefs}
