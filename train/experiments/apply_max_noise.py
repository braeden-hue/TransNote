"""
round3train/apply_max_noise.py — 테스트셋 PNG에 지정 레벨(기본 L4=최대) 노이즈를 결정론적으로
한 번 구워서 저장한다.

학습 때 쓰는 on-the-fly 확률적 augment_image/geometric_augment를 그대로 재사용하되, 파일명
기반 고정 시드로 매 실행마다 같은 결과가 나오게 한다(평가 재현성 확보 -- dataset.py의 val
_frozen_rng와 동일한 목적). 페이지 전체에 적용한다 -- run_image()가 raw page 이미지를 받아
자체적으로 staff 검출부터 다시 하므로, 캔버스 단위가 아니라 실제 촬영 사진과 동일하게 페이지
레벨에서 노이즈를 줘야 staff 검출 강건성까지 같이 테스트된다.

사용법:
    python3 apply_max_noise.py --data_dir <원본 PNG 디렉토리> --out_dir <노이즈 적용본 저장 디렉토리> [--level 4]
    (--out_dir 미지정 시 --data_dir에 덮어씀. json 라벨 파일은 그대로 복사)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_preprocessed, geometric_augment, augment_image, NOISE_LEVELS, _frozen_rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default=None, help='미지정 시 --data_dir에 덮어씀')
    ap.add_argument('--level', type=int, default=4, choices=[1, 2, 3, 4])
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lvl = NOISE_LEVELS[args.level]

    fnames = sorted(f for f in os.listdir(args.data_dir) if f.lower().endswith('.png'))
    print(f"{len(fnames)}장에 level={args.level} 노이즈 적용 중... -> {out_dir}")
    for fname in fnames:
        img_path = os.path.join(args.data_dir, fname)
        gray0 = load_preprocessed(img_path)
        seed = abs(hash(fname)) % (2 ** 31)
        with _frozen_rng(seed):
            gray, _ = geometric_augment(gray0,
                                        max_angle_deg=lvl['angle_page'],
                                        persp_margin_frac=lvl['persp_page'],
                                        p_rotate=lvl['p_rotate'], p_persp=lvl['p_persp'])
            gray = augment_image(gray, level=args.level)
        ok, buf = cv2.imencode('.png', gray)
        if not ok:
            print(f"  [WARN] {fname} 인코딩 실패, 건너뜀")
            continue
        with open(out_dir / fname, 'wb') as f:
            f.write(buf.tobytes())

        stem = os.path.splitext(fname)[0]
        src_json = Path(args.data_dir) / f"{stem}.json"
        if src_json.is_file() and out_dir != Path(args.data_dir):
            shutil.copyfile(src_json, out_dir / f"{stem}.json")

    print("완료")


if __name__ == '__main__':
    main()
