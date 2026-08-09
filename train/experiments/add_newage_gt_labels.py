"""newage04~15의 정답 라벨을 4마디로 트리밍해 exactpicture_test_full/에 추가.
exactPicture/newageXX/*.json(mscz_to_tokens.py 원본, 전체 마디)을 소스로 쓰고,
PNG 재렌더링은 하지 않음(newage는 실사 사진으로만 검증하므로 PNG 불필요) --
prepare_exactpicture_test.py와 동일한 relabel/trim 로직만 재사용.
"""
import glob
import json
import os

HERE = os.path.dirname(__file__)
SRC_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')
OUT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
MAX_MEASURES = 4
_OUT_OF_SCOPE_PREFIXES = ('artic-', 'ornament-', 'slur-')


def split_note_token(tok):
    import re
    m = re.match(r"^note-(.+)-(\d+/\d+)$", tok)
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
    targets = [f'newage{i:02d}' for i in range(4, 21)]
    ok = 0
    for name in targets:
        song_dir = os.path.join(SRC_DIR, name)
        jsons = glob.glob(os.path.join(song_dir, '*.json'))
        out_path = os.path.join(OUT_DIR, name + '.json')
        if not jsons:
            print(f"[{name}] 원본 json 없음 -- 스킵")
            continue
        if os.path.exists(out_path):
            print(f"[{name}] 이미 존재 -- 스킵")
            continue
        with open(jsons[0], encoding='utf-8') as f:
            raw_tokens = json.load(f)['tokens']
        gt_full = relabel(raw_tokens)
        gt_trimmed = trim_to_measures(gt_full, MAX_MEASURES)
        gt_clean = strip_out_of_scope([t for t in gt_trimmed if t not in ('<SOS>', '<EOS>')])
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({"id": name, "tokens": ['<SOS>'] + gt_clean + ['<EOS>']}, f, ensure_ascii=False)
        print(f"[{name}] {len(gt_clean)}토큰 -> {out_path}")
        ok += 1
    print(f"\n완료: {ok}/{len(targets)}곡")


if __name__ == '__main__':
    main()
