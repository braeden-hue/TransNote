"""exactPicture 산하 특정 곡 하나만 골라 clean 렌더링 테스트셋에 추가.
prepare_exactpicture_test.py의 단일곡 버전(신규 추가곡을 빠르게 검증할 때 50곡 전체를 다시
돌릴 필요 없게 하기 위함)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_scores as gs
from compare_recognition import tokens_to_score
from prepare_exactpicture_test import relabel, trim_to_measures, strip_out_of_scope, MAX_MEASURES, MUSESCORE

_HERE = Path(__file__).resolve().parent
SRC_DIR = _HERE / 'data' / 'local_pools' / 'exactPicture'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('song', help='exactPicture 하위 폴더명 (예: newage01)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--musescore', default=MUSESCORE)
    args = ap.parse_args()

    song_dir = SRC_DIR / args.song
    json_paths = sorted(song_dir.glob('*.json'))
    if not json_paths:
        raise SystemExit(f"{song_dir}에 json 라벨 없음")
    with open(json_paths[0], encoding='utf-8') as f:
        raw_tokens = json.load(f)['tokens']

    gt_full = relabel(raw_tokens)
    gt_trimmed = trim_to_measures(gt_full, MAX_MEASURES)
    gt_clean = strip_out_of_scope([t for t in gt_trimmed if t not in ('<SOS>', '<EOS>')])

    score = tokens_to_score(gt_clean)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{args.song}.musicxml"
    png_path = out_dir / f"{args.song}.png"
    lbl_path = out_dir / f"{args.song}.json"
    score.write("musicxml", fp=str(xml_path))
    is_grand = any(t == 'staff-bass' for t in gt_clean)
    ok = gs.render_png(args.musescore, xml_path, png_path, wide_page=True, grand=is_grand)
    if not ok:
        raise SystemExit(f"{args.song} 렌더링 실패")

    lbl_path.write_text(
        json.dumps({"id": args.song, "tokens": ['<SOS>'] + gt_clean + ['<EOS>']}, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f"완료: {png_path}")


if __name__ == '__main__':
    main()
