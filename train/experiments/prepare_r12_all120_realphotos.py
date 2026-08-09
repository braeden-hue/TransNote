"""120곡(exactPicture 전체: 클래식100+뉴에이지20, 학습분+기존 held-out분 전부) 실사
학습 풀 준비(2026-08-03). r8_2_diversity(60곡/904장) 확장판 -- "실사 곡 수/사진 자체를
늘리기"만이 지금까지 유일하게 검증된 성공 패턴([[project_r8_2_diversity_is_best_checkpoint]])
이라 그 축을 더 확장한다.

scan_staff_mismatch.py로 찾은 오선검출 불일치 사진(staff_mismatch_exclude.json, 33장 --
단일오선 GT인데 대보표로 오검출되는 등 이미지/정답 불일치)은 제외 -- 학습에 노이즈만
줄 뿐 신호가 아님.

주의: 이후로는 held-out 60곡이라는 개념이 사라짐(전부 학습에 포함) -- 검증은 별도
신규 곡 + 120곡 자체 실측(적합도 확인용, 일반화 검증 아님)으로 대체.
"""
import glob
import json
import os
import shutil

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')
GT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
OUT = os.path.join(HERE, 'data', 'local_pools', 'r12_all120_realphotos')
EXCLUDE_JSON = os.path.join(HERE, 'staff_mismatch_exclude.json')


def main():
    with open(EXCLUDE_JSON, encoding='utf-8') as f:
        exclude_list = json.load(f)
    exclude_set = {(e['song'], e['photo']) for e in exclude_list}
    print(f"제외 대상: {len(exclude_set)}장")

    os.makedirs(OUT, exist_ok=True)
    songs = sorted(f[:-5] for f in os.listdir(GT_DIR) if f.endswith('.json'))
    n_songs = 0
    n_photos = 0
    n_excluded = 0
    for name in songs:
        song_dir = os.path.join(SRC, name)
        gt_path = os.path.join(GT_DIR, name + '.json')
        if not os.path.isdir(song_dir) or not os.path.isfile(gt_path):
            continue
        photos = glob.glob(os.path.join(song_dir, '*.jpg')) + glob.glob(os.path.join(song_dir, '*.jpeg'))
        if not photos:
            continue
        n_songs += 1
        for photo in photos:
            pname = os.path.basename(photo)
            if (name, pname) in exclude_set:
                n_excluded += 1
                continue
            stem = os.path.splitext(pname)[0]
            dst_photo = os.path.join(OUT, stem + os.path.splitext(photo)[1])
            dst_json = os.path.join(OUT, stem + '.json')
            shutil.copy2(photo, dst_photo)
            shutil.copy2(gt_path, dst_json)
            n_photos += 1
    print(f'{n_songs}곡, {n_photos}장 -> {OUT} (제외 {n_excluded}장)')


if __name__ == '__main__':
    main()
