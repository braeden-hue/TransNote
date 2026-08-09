"""신규 클래식(비-newage) 14곡의 4마디 트리밍 GT를 exactpicture_test_full/에 추가.
prepare_exactpicture_test.py와 동일한 relabel/trim 로직 재사용 -- 기존 86곡은 이미
exactpicture_test_full에 있으므로 건드리지 않는다(이미 존재하면 스킵).
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
    'sonata_84_4', 'sonata_88_104', 'sonatineHa_11_57', 'sonatineHa_12_21',
    'sonatineHa_28_49', 'sonatineHa_29_68', 'sonatineHa_30_92', 'sonatineHa_32_13',
    'sonatineHa_32_26', 'sonatineHa_33_32', 'sonatineHa_33_34', 'sonatineHa_34_48',
    'sonatineHa_9_19', 'sonatineHa_9_23',
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
