"""exactPicture(실사 86곡) 중 무작위 50곡을 골라, mscz 정답 라벨을 그대로 music21 악보로
복원해 MuseScore로 "깨끗하게" 렌더링한 이미지 테스트셋을 만든다. eval_mscz_clean.py와 동일한
방법론(정답 토큰 -> 악보 복원 -> 렌더링, 실사 촬영 노이즈 배제하고 콘텐츠 자체의 인식률만 봄)을
exactPicture 86곡 전체(구 ml/data/test/chop*의 확장판)에 적용하도록 재작성.
출력은 eval_r3_synth_test.py/error_breakdown.py/error_location_r4.py가 그대로 읽을 수 있게
{stem}.png + {stem}.json(tokens 키) 평평한 구조로 저장.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_scores as gs
from compare_recognition import tokens_to_score

_HERE = Path(__file__).resolve().parent
SRC_DIR = _HERE / 'data' / 'local_pools' / 'exactPicture'
OUT_DIR = _HERE / 'data' / 'local_pools' / 'exactpicture_test50'
MUSESCORE = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
MAX_MEASURES = 4

_NOTE_DUR_RE = None
_OUT_OF_SCOPE_PREFIXES = ('artic-', 'ornament-', 'slur-')


def split_note_token(tok):
    import re
    global _NOTE_DUR_RE
    if _NOTE_DUR_RE is None:
        _NOTE_DUR_RE = re.compile(r"^note-(.+)-(\d+/\d+)$")
    m = _NOTE_DUR_RE.match(tok)
    if not m:
        return [tok]
    return [f"note-{m.group(1)}", f"dur-{m.group(2)}"]


def relabel(tokens):
    out = []
    for t in tokens:
        out.extend(split_note_token(t))
    return out


def trim_to_measures(tokens, max_measures):
    tokens = [t for t in tokens if t not in ('<SOS>', '<EOS>', '<PAD>')]
    out = []
    n_bar = 0
    for t in tokens:
        out.append(t)
        if t.startswith('barline'):
            n_bar += 1
            if n_bar >= max_measures:
                break
    if not out or not out[-1].startswith('barline'):
        out.append('barline-final')
    elif out[-1] != 'barline-final':
        out[-1] = 'barline-final'
    return out


def strip_out_of_scope(tokens):
    return [t for t in tokens if not t.startswith(_OUT_OF_SCOPE_PREFIXES)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--src_dir', default=str(SRC_DIR))
    ap.add_argument('--out_dir', default=str(OUT_DIR))
    ap.add_argument('--musescore', default=MUSESCORE)
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    song_dirs = sorted(d for d in src_dir.iterdir()
                        if d.is_dir() and not d.name.startswith('.') and list(d.glob('*.json')))
    print(f"exactPicture 전체 {len(song_dirs)}곡 중 {args.n}곡 무작위 선별(seed={args.seed})")
    random.seed(args.seed)
    picked = random.sample(song_dirs, min(args.n, len(song_dirs)))

    ok = 0
    for song_dir in picked:
        name = song_dir.name
        json_paths = sorted(song_dir.glob('*.json'))
        if not json_paths:
            print(f"[{name}] json 라벨 없음 -- 스킵")
            continue
        with open(json_paths[0], encoding='utf-8') as f:
            raw_tokens = json.load(f)['tokens']

        gt_full = relabel(raw_tokens)
        gt_trimmed = trim_to_measures(gt_full, MAX_MEASURES)
        gt_clean = strip_out_of_scope([t for t in gt_trimmed if t not in ('<SOS>', '<EOS>')])

        try:
            score = tokens_to_score(gt_clean)
        except Exception as exc:
            print(f"[{name}] 악보 복원 실패: {exc}")
            continue

        xml_path = out_dir / f"{name}.musicxml"
        png_path = out_dir / f"{name}.png"
        lbl_path = out_dir / f"{name}.json"
        score.write("musicxml", fp=str(xml_path))
        is_grand = any(t == 'staff-bass' for t in gt_clean)
        if not gs.render_png(args.musescore, xml_path, png_path, wide_page=True, grand=is_grand):
            print(f"[{name}] 렌더링 실패")
            continue

        lbl_path.write_text(
            json.dumps({"id": name, "tokens": ['<SOS>'] + gt_clean + ['<EOS>']}, ensure_ascii=False),
            encoding='utf-8'
        )
        ok += 1

    print(f"\n완료: {ok}/{len(picked)}곡 -> {out_dir.resolve()}")


if __name__ == '__main__':
    main()
