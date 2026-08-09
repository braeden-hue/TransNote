"""newage 20곡의 이미 트리밍된 GT(exactpicture_test_full/newageXX.json)를 실사 촬영 없이
"깨끗하게" MuseScore로 렌더링한 테스트셋을 만든다. 목적: 화음 보강 학습(r9) 이후 체크포인트가
실제 콘텐츠 자체(카메라 노이즈 없이)를 얼마나 잘 인식하는지, 실사 도메인시프트와 분리해서 본다.
출력은 eval_r3_synth_test.py가 그대로 읽을 수 있게 {stem}.png + {stem}.json 평평한 구조.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_scores as gs
from compare_recognition import tokens_to_score

GT_DIR = HERE / 'data' / 'local_pools' / 'exactpicture_test_full'
OUT_DIR = HERE / 'data' / 'local_pools' / 'newage_clean_test'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--musescore', default=None)
    args = ap.parse_args()
    musescore = gs.find_musescore(args.musescore)
    if not musescore:
        print("ERROR: MuseScore를 찾을 수 없음 -- --musescore로 경로 지정 필요", file=sys.stderr)
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    songs = sorted(p.stem for p in GT_DIR.glob('newage*.json'))
    ok = 0
    for name in songs:
        gt_path = GT_DIR / f"{name}.json"
        with open(gt_path, encoding='utf-8') as f:
            gt_tokens = json.load(f)['tokens']
        gt_clean = [t for t in gt_tokens if t not in ('<SOS>', '<EOS>')]

        try:
            score = tokens_to_score(gt_clean)
        except Exception as exc:
            print(f"[{name}] 악보 복원 실패: {exc}")
            continue

        xml_path = OUT_DIR / f"{name}.musicxml"
        png_path = OUT_DIR / f"{name}.png"
        lbl_path = OUT_DIR / f"{name}.json"
        score.write("musicxml", fp=str(xml_path))
        is_grand = any(t == 'staff-bass' for t in gt_clean)
        if not gs.render_png(musescore, xml_path, png_path, wide_page=True, grand=is_grand):
            print(f"[{name}] 렌더링 실패")
            continue

        lbl_path.write_text(
            json.dumps({"id": name, "tokens": ['<SOS>'] + gt_clean + ['<EOS>']}, ensure_ascii=False),
            encoding='utf-8')
        ok += 1
        print(f"[{name}] OK")

    print(f"\n완료: {ok}/{len(songs)}곡 -> {OUT_DIR}")


if __name__ == '__main__':
    main()
