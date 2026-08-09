"""신규 검증용 6곡(sonatine_22_30/23_38/23_42/32_38/36_60/81_92, 2026-08-03 촬영)의
4마디 트리밍 GT를 exactpicture_test_full/에 추가. _add_new_classical_gt.py와 동일 로직.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_exactpicture_test import relabel, trim_to_measures, strip_out_of_scope, MAX_MEASURES

SRC_DIR = HERE / 'data' / 'local_pools' / 'exactPicture'
OUT_DIR = HERE / 'data' / 'local_pools' / 'exactpicture_test_full'

NEW_SONGS = [
    'sonatine_22_30', 'sonatine_23_38', 'sonatine_23_42',
    'sonatine_32_38', 'sonatine_36_60', 'sonatine_81_92',
]


def main():
    ok = 0
    for name in NEW_SONGS:
        out_path = OUT_DIR / f"{name}.json"
        if out_path.exists():
            print(f"[{name}] 이미 존재 -- 스킵")
            continue
        song_dir = SRC_DIR / name
        json_paths = sorted(song_dir.glob('*.json'))
        if not json_paths:
            print(f"[{name}] 원본 json 없음 -- 스킵")
            continue
        with open(json_paths[0], encoding='utf-8') as f:
            raw_tokens = json.load(f)['tokens']
        gt_full = relabel(raw_tokens)
        gt_trimmed = trim_to_measures(gt_full, MAX_MEASURES)
        gt_clean = strip_out_of_scope([t for t in gt_trimmed if t not in ('<SOS>', '<EOS>')])
        out_path.write_text(
            json.dumps({"id": name, "tokens": ['<SOS>'] + gt_clean + ['<EOS>']}, ensure_ascii=False),
            encoding='utf-8')
        print(f"[{name}] {len(gt_clean)}토큰 -> {out_path}")
        ok += 1
    print(f"\n완료: {ok}/{len(NEW_SONGS)}곡")


if __name__ == '__main__':
    main()
