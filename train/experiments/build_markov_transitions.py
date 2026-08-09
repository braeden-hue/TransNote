"""
round3train/build_markov_transitions.py -- PDMX 실제 곡(mxl 파일들)에서 진짜 멜로디
음정 전이(다이어토닉 스텝) 통계를 뽑아 generate_scores.py가 가중치로 쓸 확률 테이블
(markov_transitions.json)로 저장한다.

state: 직전 음 -> 다음 음의 "다이어토닉 스텝"(반음 아님, 음이름이 몇 칸 이동했는지 --
예: C->E는 +2, C->G는 +4, C->D는 +1). music21 Pitch.diatonicNoteNum 차이라 조성과
무관하게(transposition-invariant) 재사용 가능 -- 어느 조표든 그대로 적용됨.

화음은 chosen[0](최저음)만 대표 피치로 사용 -- generate_scores.py가 화음의 "다음 음
기준점"으로 최저음(정렬 후 chosen[0])을 쓰는 것과 동일한 규칙(일관성 유지).
쉼표는 선율 흐름을 끊는다고 보고 그 앞뒤 음 사이는 전이로 세지 않는다.

MuseScore CLI 불필요(music21이 .mxl을 직접 파싱) -- 다른 MuseScore 작업과 동시 실행 가능.

사용법:
    python3 build_markov_transitions.py --mxl_dir mxl_extracted --out markov_transitions.json
"""
import argparse
import glob
import json
from collections import Counter

import music21 as m21


def representative_pitches(part):
    """part의 노트/화음/쉼표 시퀀스를 대표 피치(단음은 그 음, 화음은 최저음, 쉼표는 None)
    순서로 반환."""
    pitches = []
    for el in part.recurse().notesAndRests:
        if el.isRest:
            pitches.append(None)
            continue
        if el.isChord:
            if not el.pitches:
                pitches.append(None)  # 화성기호(ChordSymbol) 등 실음 없는 요소
                continue
            p = min(el.pitches, key=lambda x: x.midi)
        elif hasattr(el, 'pitch'):
            p = el.pitch
        else:
            pitches.append(None)  # Unpitched(타악기 표기 등) -- 쉼표처럼 흐름을 끊음
            continue
        pitches.append(p)
    return pitches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mxl_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_interval', type=int, default=14,
                     help='이 값(다이어토닉 스텝) 넘는 도약은 +-이 값으로 clamp (기본 14=2옥타브)')
    args = ap.parse_args()

    files = sorted(glob.glob(f'{args.mxl_dir}/**/*.mxl', recursive=True))
    print(f'{len(files)}개 mxl 파일 발견')

    interval_counts = Counter()
    n_parsed = 0
    n_failed = 0
    n_intervals = 0

    for i, fp in enumerate(files):
        # 9845개 커뮤니티 업로드 파일이라 파싱 실패뿐 아니라 예상 못 한 요소 타입(타악기
        # 표기 등)으로 처리 도중 죽는 경우도 있음 -- 파일 하나가 전체 여러 시간짜리 실행을
        # 끝장내지 않도록 파일 단위로 통째로 방어.
        try:
            score = m21.converter.parse(fp)
            n_parsed += 1
            for part in score.parts:
                pitches = representative_pitches(part)
                prev = None
                for p in pitches:
                    if p is None:
                        prev = None
                        continue
                    if prev is not None:
                        interval = p.diatonicNoteNum - prev.diatonicNoteNum
                        interval = max(-args.max_interval, min(args.max_interval, interval))
                        interval_counts[interval] += 1
                        n_intervals += 1
                    prev = p
        except Exception:
            n_failed += 1
            continue
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(files)} 처리 중... (성공 {n_parsed}, 실패 {n_failed}, 전이 {n_intervals})', flush=True)

    print(f'파싱 성공 {n_parsed}개, 실패 {n_failed}개, 전이 샘플 {n_intervals}개')
    if n_intervals == 0:
        print('전이 샘플이 0개 -- 저장하지 않고 종료')
        return

    total = sum(interval_counts.values())
    table = {str(k): v / total for k, v in sorted(interval_counts.items())}

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({
            'description': 'PDMX 실제곡에서 집계한 다이어토닉 음정 전이확률(직전 음 -> 다음 음, 음이름 스텝 단위)',
            'n_source_pieces': n_parsed,
            'n_transitions': n_intervals,
            'max_interval': args.max_interval,
            'table': table,
        }, f, ensure_ascii=False, indent=1)

    print(f'저장: {args.out}')
    top = sorted(interval_counts.items(), key=lambda x: -x[1])[:12]
    for interval, cnt in top:
        print(f'  interval={interval:+d}: {cnt}회 ({cnt/total*100:.1f}%)')


if __name__ == '__main__':
    main()
