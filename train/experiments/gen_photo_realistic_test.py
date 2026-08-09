"""designKit/scoped_test/ 재생성 -- MuseScore 풀페이지 렌더(제목+작곡가+넓은 여백)를
그대로 쓰는 대신, 오선 부분만 타이트하게 크롭해 "실제로 오선 위주로 촬영한 사진"처럼
만든다. compare_recognition.py와 같은 확정 스코프(옥타브/헤어핀 제외)로 생성하고,
detect_staffs()로 찾은 오선 경계 기준 작은 여백만 남기고 잘라낸 뒤 모델로 재검증까지 한다.
"""
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_scores as gs
from compare_recognition import _apply_current_scope
from dataset import _staff_y_bounds, detect_staffs, load_preprocessed, load_tokenizer
from inference import run_image
from model import OmrSeq2Seq
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS

COUNT = 20
SEED = 777011
OUT_DIR = Path(__file__).resolve().parent.parent / 'designKit' / 'scoped_test'
MUSESCORE = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
CKPT = Path(__file__).resolve().parent.parent / 'secrets' / 'checkpoints' / 'seq2seq_p2s5n5_best.pt'
TOKENIZER = Path(__file__).resolve().parent / 'tokenizer258.json'

X_PAD = 30


def crop_to_staff(gray: np.ndarray) -> np.ndarray | None:
    """풀페이지 렌더를 오선(들) 주변만 타이트하게 크롭. 실패 시 None.
    고정 픽셀 여백 대신 _staff_y_bounds()(dataset.py, extract_system_canvas와 동일 로직)로
    첫/마지막 오선의 실제 콘텐츠 범위(옥타브 브래킷·높은/낮은 음표 등)를 반영한다 --
    고정 여백(예: 45px)은 오선에서 멀리 그려지는 음표머리가 잘려나가는 문제가 있었음
    (2026-07-27 실측: photo15 등에서 상단 음표 절단 확인)."""
    staffs = detect_staffs(gray)
    if not staffs:
        return None
    y0, _ = _staff_y_bounds(gray, staffs[0])
    _, y1 = _staff_y_bounds(gray, staffs[-1])
    H, W = gray.shape
    y1 = min(H, y1 + 1)
    strip = gray[y0:y1, :]

    binary = strip < 250
    cols = np.where(binary.any(axis=0))[0]
    if len(cols) == 0:
        x0, x1 = 0, W
    else:
        x0 = max(0, int(cols[0]) - X_PAD)
        x1 = min(W, int(cols[-1]) + X_PAD)
    return gray[y0:y1, x0:x1]


def main():
    random.seed(SEED)
    _apply_current_scope(ottava_prob=0, hairpin_prob=0)

    work = OUT_DIR / '_work'
    work.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    tok2id, id2tok = load_tokenizer(str(TOKENIZER))
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(str(CKPT), map_location='cpu', weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {CKPT.name}")

    results = []
    for i in range(1, COUNT + 1):
        score, gt_tokens, _breaks = gs.build_score_r3(i, force_c_major=False)
        xml_path = work / f"s{i}.musicxml"
        full_png = work / f"s{i}_full.png"
        score.write("musicxml", fp=str(xml_path))
        if not gs.render_png(MUSESCORE, xml_path, full_png):
            print(f"  [WARN] {i}번 렌더링 실패, 건너뜀")
            continue

        gray = load_preprocessed(str(full_png))
        cropped = crop_to_staff(gray)
        if cropped is None or cropped.shape[0] < 30:
            print(f"  [WARN] {i}번 오선 검출 실패, 건너뜀")
            continue

        out_png = OUT_DIR / f"photo{i}.png"
        ok, buf = cv2.imencode('.png', cropped)
        with open(out_png, 'wb') as f:
            f.write(buf.tobytes())

        pred_tokens = run_image(str(out_png), seq2seq, tok2id, id2tok, device, beam_width=1)
        gt_clean = [t for t in gt_tokens if t not in ('<SOS>', '<EOS>')]
        gt_ids = [tok2id[t] for t in gt_clean if t in tok2id]
        pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]
        pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
        ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        acc = max(0.0, 1 - ter) * 100
        exact = (pred_tokens == gt_clean)
        results.append(acc)
        print(f"[{i}/{COUNT}] photo{i}.png ({cropped.shape[1]}x{cropped.shape[0]}) "
              f"exact={exact} Acc={acc:.1f}%")

    shutil.rmtree(work, ignore_errors=True)
    if results:
        print(f"\n=== 평균 Acc: {sum(results)/len(results):.1f}% (n={len(results)}) ===")


if __name__ == '__main__':
    main()
