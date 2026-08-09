"""ml/data/test/chop*/의 실사 사진 대신, 정답 라벨(GT 토큰)을 그대로 music21 악보로
복원해 MuseScore로 "깨끗하게" 렌더링한 이미지로 검증. 목적: 지난 실사 테스트(4.0%)가
사진 자체의 문제(조명/각도/종이 질감)인지, 아니면 실제 쇼팽 곡 내용 자체를 모델이
못 배운 것인지 분리 진단. compare_recognition.py의 tokens_to_score()(토큰 -> music21
Score 역변환, 이미 검증된 함수)를 재사용 -- artic 등 스코프 밖 토큰은 자동으로 무시하고
렌더링하므로, GT 비교에서도 동일하게 제외해 렌더된 이미지와 일관되게 맞춘다.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
import generate_scores as gs
from compare_recognition import tokens_to_score
from dataset import load_tokenizer
from inference import run_image
from model import OmrSeq2Seq
from train import fix_chord_tokens, fix_span_tokens, measure_segmented_ter, _BARLINE_TOKEN_STRS

TEST_DIR = Path(__file__).resolve().parent.parent / 'ml' / 'data' / 'test'
WORK_DIR = Path(__file__).resolve().parent.parent / 'designKit' / 'mscz_clean_test'
MUSESCORE = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
CKPT = Path(__file__).resolve().parent.parent / 'secrets' / 'checkpoints' / 'seq2seq_p2s5n5_best.pt'
TOKENIZER = Path(__file__).resolve().parent / 'tokenizer258.json'
MAX_MEASURES = 4

_NOTE_DUR_RE = re.compile(r"^note-(.+)-(\d+/\d+)$")
_OUT_OF_SCOPE_PREFIXES = ('artic-', 'ornament-', 'slur-')


def split_note_token(tok):
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
    out = []
    n_bar = 0
    for t in tokens:
        out.append(t)
        if t.startswith('barline'):
            n_bar += 1
            if n_bar >= max_measures:
                break
    if not out[-1].startswith('barline'):
        out.append('barline-final')
    elif out[-1] != 'barline-final':
        out[-1] = 'barline-final'
    return out


def strip_out_of_scope(tokens):
    return [t for t in tokens if not t.startswith(_OUT_OF_SCOPE_PREFIXES)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(CKPT), help='검증할 체크포인트 경로 덮어씀')
    ap.add_argument('--musescore', default=MUSESCORE)
    ap.add_argument('--work_dir', default=str(WORK_DIR))
    ap.add_argument('--tokenizer', default=str(TOKENIZER))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    tok2id, id2tok = load_tokenizer(args.tokenizer)
    barline_ids = {tok2id[t] for t in _BARLINE_TOKEN_STRS if t in tok2id}
    seq2seq = OmrSeq2Seq(vocab_size=len(tok2id)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    seq2seq.load_state_dict(ckpt['model'])
    seq2seq.eval()
    print(f"모델 로드: {args.ckpt}\n")

    folders = sorted(d for d in glob.glob(str(TEST_DIR / 'chop*')) if os.path.isdir(d))
    results = []
    for folder in folders:
        name = os.path.basename(folder)
        # mscz_to_label.py로 .mscz에서 새로 뽑은 라벨(*_new.json)을 우선 사용 -- 기존
        # *.json은 마디 경계도 안 맞고(6마디 vs 실제 mscz 4마디) tie 처리도 달라서
        # (stripTies 미적용으로 추정) 부정확한 비교였음(2026-07-28 사용자 확인).
        json_paths = glob.glob(os.path.join(folder, '*_new.json'))
        if not json_paths:
            json_paths = glob.glob(os.path.join(folder, '*.json'))
        if not json_paths:
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

        xml_path = work_dir / f"{name}.musicxml"
        png_path = work_dir / f"{name}_clean.png"
        score.write("musicxml", fp=str(xml_path))
        if not gs.render_png(args.musescore, xml_path, png_path):
            print(f"[{name}] 렌더링 실패")
            continue

        gt_ids = [tok2id[t] for t in gt_clean if t in tok2id]
        pred_tokens = run_image(str(png_path), seq2seq, tok2id, id2tok, device, beam_width=1)
        pred_ids = [tok2id[t] for t in pred_tokens if t in tok2id]
        pred_ids = fix_span_tokens(fix_chord_tokens(pred_ids, id2tok), id2tok)
        ter = measure_segmented_ter(pred_ids, gt_ids, barline_ids)
        acc = max(0.0, 1 - ter) * 100
        results.append(acc)
        print(f"[{name}] Acc={acc:.1f}%")
        print(f"  GT  : {' '.join(gt_clean)}")
        print(f"  PRED: {' '.join(pred_tokens)}\n")

        # 예측 토큰도 GT와 똑같이 악보로 복원해서 렌더링 -- 콘솔 텍스트만으로는
        # 비교가 어렵다는 피드백(2026-07-28)에 따라 GT PNG 옆에 나란히 저장.
        try:
            pred_score = tokens_to_score(pred_tokens)
            pred_xml = work_dir / f"{name}_pred.musicxml"
            pred_png = work_dir / f"{name}_pred.png"
            pred_score.write("musicxml", fp=str(pred_xml))
            if not gs.render_png(args.musescore, pred_xml, pred_png):
                print(f"  [WARN] {name} 예측 렌더링 실패")
        except Exception as exc:
            print(f"  [WARN] {name} 예측 -> 악보 복원 실패: {exc}")

    if results:
        print(f"=== 전체 평균 Acc: {sum(results)/len(results):.1f}% (n={len(results)}장) ===")


if __name__ == '__main__':
    main()
