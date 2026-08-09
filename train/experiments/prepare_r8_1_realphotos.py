"""Round8 2단계 커리큘럼의 1단계(r8_1)용 실사 풀 준비.
클래식 난이도 하위 30곡 + 뉴에이지 난이도 상위(쉬운) 10곡의 실사 사진을 골라
GT(exactpicture_test_full의 트리밍된 json)와 함께 평평한 디렉토리로 모은다.

뉴에이지를 학습에 포함시키는 건 이번이 처음(2026-08-02, 사용자 결정) -- 나머지
10곡(03,04,05,06,07,09,11,14,19,20)은 계속 검증 전용으로 완전히 제외한다.
"""
import glob
import os
import shutil

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')
GT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
OUT = os.path.join(HERE, 'data', 'local_pools', 'r8_1_realphotos')

CLASSICAL_EASY = """winter_35_33 sonatine_21p_1 winter_35_25 winter_35_21 winter_35_37
winter_35_17 summer1_10p_21 winter_40_25 sonatineHa_81_36 sonatine_13p_35
summer1_11p_25 sonata_108_109 sonatineHa_32_26 sonata_33_1 sonatineHa_78_37
sonatine_16p_54 sonatineHa_8_1 sonata_149_80 summer1_18p_9 sonatine_14_1
sonatineHa_49_4 sonatineHa_81_40 sonatineHa_8_13 winter_34_1 sonata_149_68
winter_34_5 winter_34_9 summer_16p_37 sonata_99_110 sonatineHa_32_13""".split()

NEWAGE_EASY = "newage08 newage10 newage18 newage17 newage15 newage16 newage12 newage02 newage01 newage13".split()

assert len(CLASSICAL_EASY) == 30, len(CLASSICAL_EASY)
assert len(NEWAGE_EASY) == 10, len(NEWAGE_EASY)


def main():
    os.makedirs(OUT, exist_ok=True)
    n_songs = 0
    n_photos = 0
    for name in CLASSICAL_EASY + NEWAGE_EASY:
        song_dir = os.path.join(SRC, name)
        gt_path = os.path.join(GT_DIR, name + '.json')
        if not os.path.isfile(gt_path):
            print(f"[{name}] GT 없음 -- 스킵")
            continue
        photos = glob.glob(os.path.join(song_dir, '*.jpg')) + glob.glob(os.path.join(song_dir, '*.jpeg'))
        if not photos:
            print(f"[{name}] 사진 없음 -- 스킵")
            continue
        n_songs += 1
        for photo in photos:
            stem = os.path.splitext(os.path.basename(photo))[0]
            dst_photo = os.path.join(OUT, stem + os.path.splitext(photo)[1])
            dst_json = os.path.join(OUT, stem + '.json')
            shutil.copy2(photo, dst_photo)
            shutil.copy2(gt_path, dst_json)
            n_photos += 1
    print(f'{n_songs}곡, {n_photos}장 -> {OUT}')


if __name__ == '__main__':
    main()
