"""
token_to_notes.py — DeepScore 토큰 리스트를 online_webpage/js/app.js가 기대하는
{notes: [...]} / {staves: [...]}  JSON 스키마로 변환한다.

lib/services/omr_token_parser.dart(Flutter)와 동일한 문법 파싱 로직의 Python 포트.
문법 요약(round3train/mscz_to_tokens.py 기준)은 그 파일 docstring 참고.

pitch 표기: 웹 쪽(online_webpage/js/samples.js)의 NOTE_NAMES/BLACK_LABEL이 샤프(#)
표기만 이해하므로(예: 'C#4' -> '1'), 모델이 내는 플랫(Db/Eb/Gb/Ab/Bb) 표기는 여기서
동일 음의 샤프 표기로 정규화한다.
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
    """토큰 리스트 -> {'treble': [...], 'bass': [...], 'timeSignature': [n, d]}."""
    treble: List[ScoreNote] = []
    bass: List[ScoreNote] = []
    time_sig = [4, 4]

    on_bass = False
    treble_beat = 1
    bass_beat = 1
    start_repeat_pending: Optional[str] = None

    pending_pitch: Optional[str] = None
    pending_duration = 1.0
    pending_chord: List[str] = []
    pending_dynamic: Optional[str] = None
    has_pending = False

    def push(pitch: str, duration: float, chord: Optional[List[str]] = None,
             is_rest: bool = False, dynamic_mark: Optional[str] = None) -> None:
        nonlocal start_repeat_pending, treble_beat, bass_beat
        target = bass if on_bass else treble
        beat = bass_beat if on_bass else treble_beat
        note: ScoreNote = {
            'pitch': _normalize_pitch(pitch),
            'duration': duration,
            'beat': beat,
        }
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
        advance = min(4, max(1, round(duration)))
        if on_bass:
            bass_beat = (bass_beat - 1 + advance) % 4 + 1
        else:
            treble_beat = (treble_beat - 1 + advance) % 4 + 1

    def finalize_pending() -> None:
        nonlocal pending_pitch, pending_duration, pending_chord, pending_dynamic, has_pending
        if not has_pending:
            return
        push(pending_pitch or '', pending_duration, chord=list(pending_chord),
             dynamic_mark=pending_dynamic)
        pending_pitch = None
        pending_duration = 1.0
        pending_chord = []
        pending_dynamic = None
        has_pending = False

    def mark_end_repeat() -> None:
        target = bass if on_bass else treble
        if not target:
            return
        target[-1]['repeatMark'] = 'end-repeat'

    for tok in tokens:
        if tok.startswith('time-'):
            parts = tok[5:].split('/')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                time_sig = [int(parts[0]), int(parts[1])]
        elif tok.startswith('clef-') or tok.startswith('key-'):
            pass  # 무시 — treble/bass 배열 소속 자체가 clef를 암시 (Dart 쪽과 동일 관례)
        elif tok == 'staff-bass':
            finalize_pending()
            on_bass = True
        elif tok.startswith('note-'):
            finalize_pending()
            pending_pitch = tok[5:]
            pending_duration = 1.0
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
            treble_beat = 1
            bass_beat = 1
            on_bass = False
        # artic-*/fermata/ornament-*/tie/slur-*/trill-*/tuplet-*/ottava-*/hairpin-*: 무시.

    finalize_pending()

    return {'treble': treble, 'bass': bass, 'timeSignature': time_sig}
