"""realImage/exactPicture/ 전체(장르 다양, 실사 촬영) held-out 셋으로 r15 FP32 정확도를
측정하는 재사용 가능한 평가 스크립트. 양자화 전/후 비교용 기준점 -- 매번 이 스크립트로
동일 조건 재측정하면 됨.

각 하위 폴더(예: newage21/, chop18_183/)는 <폴더명>.json(GT 토큰) + 실사 사진(.jpg)을
담고 있다. 사진이 여러 장이면 ASCII 파일명(KakaoTalk_*.jpg)을 우선 사용(cv2.imread가
Windows에서 비-ASCII 경로를 못 여는 경우가 있어 회피).

사용법:
    python train/eval_exactpicture.py [--limit N] [--out report.json]
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from dataset import load_tokenizer
from inference import run_image, analyze_sample
from model import OmrSeq2Seq, infer_arch_from_state_dict

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "train" / "checkpoints" / "r15_cropfix_coordconv" / "seq2seq_best.pt"
TOKENIZER = ROOT / "train" / "tokenizer258.json"
DATA_DIR = ROOT / "realImage" / "exactPicture"


def pick_image(folder: Path, stem: str) -> Path | None:
    jpgs = sorted(folder.glob("KakaoTalk_*.jpg")) or sorted(folder.glob("*.jpg"))
    if not jpgs:
        return None
    img = jpgs[0]
    if not img.name.isascii():
        tmp = Path(tempfile.gettempdir()) / f"{stem}_eval_tmp.jpg"
        shutil.copy(img, tmp)
        return tmp
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="테스트용 상한(디버그)")
    ap.add_argument("--out", default=str(ROOT / "train" / "eval_exactpicture_report.json"))
    args = ap.parse_args()

    tok2id, id2tok = load_tokenizer(str(TOKENIZER))
    ckpt = torch.load(str(CKPT), map_location='cpu', weights_only=False)
    arch = infer_arch_from_state_dict(ckpt['model'])
    print(f"arch={arch}", flush=True)
    model = OmrSeq2Seq(vocab_size=len(tok2id), **arch)
    model.load_state_dict(ckpt['model'])
    model.eval()
    device = torch.device('cpu')

    folders = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    if args.limit:
        folders = folders[:args.limit]

    results = []
    t0 = time.time()
    for idx, folder in enumerate(folders, 1):
        stem = folder.name
        gt_path = folder / f"{stem}.json"
        img_path = pick_image(folder, stem)
        if img_path is None or not gt_path.exists():
            print(f"[{idx}/{len(folders)}] {stem}: 파일 없음, 건너뜀", flush=True)
            continue
        try:
            r = analyze_sample(str(img_path), str(gt_path), model, tok2id, id2tok, device)
            results.append({
                'id': stem, 'is_grand': r['is_grand'],
                'overall_ter': r['overall_ter'], 'note_err': r['note_err'],
            })
            print(f"[{idx}/{len(folders)}] {stem}: note_acc={100*(1-r['note_err']):.1f}%  "
                  f"({time.time()-t0:.0f}s 경과)", flush=True)
        except Exception as e:
            print(f"[{idx}/{len(folders)}] {stem}: [ERROR] {e}", flush=True)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    if results:
        avg_note_acc = 100 * (1 - sum(r['note_err'] for r in results) / len(results))
        avg_ter_acc = 100 * (1 - sum(r['overall_ter'] for r in results) / len(results))
        print(f"\n=== {len(results)}개 샘플 ===")
        print(f"음표 레벨 평균 Acc: {avg_note_acc:.1f}%")
        print(f"전체 TER 평균 Acc : {avg_ter_acc:.1f}%")
        print(f"총 소요 시간: {time.time()-t0:.0f}s ({(time.time()-t0)/len(results):.1f}s/장)")


if __name__ == '__main__':
    main()
