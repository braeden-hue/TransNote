"""Round7용: exactPicture의 클래식(비-newage) 곡 실사 사진을 평탄한 학습용 디렉토리로
준비. OMRDataset은 이미지와 같은 stem의 .json을 요구하는데, exactPicture는 곡 폴더 하나에
json 1개 + 사진 여러 장 구조라서, 사진마다 그 곡의 GT json을 같은 stem으로 복사해준다.
newage*는 검증 전용으로 남겨야 하므로 여기서 완전히 제외한다.
"""
import argparse
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/local_pools/exactPicture')
    ap.add_argument('--gt_dir', default='data/local_pools/exactpicture_test_full',
                     help='4마디로 이미 트리밍된 정답(실제 사진 내용과 일치) -- 곡 폴더 안의 '
                          'raw mscz_to_tokens 출력(전체 마디)이 아니라 이쪽을 써야 함')
    ap.add_argument('--out', default='data/local_pools/classical_realphotos')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_photos = 0
    n_songs = 0
    for song_dir in sorted(os.listdir(args.src)):
        if song_dir.startswith('.') or song_dir.startswith('newage'):
            continue
        dp = os.path.join(args.src, song_dir)
        if not os.path.isdir(dp):
            continue
        gt_path = os.path.join(args.gt_dir, song_dir + '.json')
        if not os.path.isfile(gt_path):
            continue
        photos = glob.glob(os.path.join(dp, '*.jpg')) + glob.glob(os.path.join(dp, '*.jpeg'))
        if not photos:
            continue
        n_songs += 1
        for photo in photos:
            stem = os.path.splitext(os.path.basename(photo))[0]
            dst_photo = os.path.join(args.out, stem + os.path.splitext(photo)[1])
            dst_json = os.path.join(args.out, stem + '.json')
            shutil.copy2(photo, dst_photo)
            shutil.copy2(gt_path, dst_json)
            n_photos += 1
    print(f'{n_songs}곡, {n_photos}장 -> {args.out}')


if __name__ == '__main__':
    main()
