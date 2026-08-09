"""Round8 실사 직접 학습 실험(2026-08-02)용 실사 풀 준비.
클래식 난이도 하위 50곡(기존 30 + 추가 20) + 뉴에이지 쉬운 10곡 -- r8_1_realphotos(40곡)
확장판. 나머지 뉴에이지 10곡(03,04,05,06,07,09,11,14,19,20)은 계속 검증 전용 유지.
"""
import glob
import os
import shutil

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, 'data', 'local_pools', 'exactPicture')
GT_DIR = os.path.join(HERE, 'data', 'local_pools', 'exactpicture_test_full')
OUT = os.path.join(HERE, 'data', 'local_pools', 'r8_2_realphotos')

CLASSICAL_EASY_30 = """winter_35_33 sonatine_21p_1 winter_35_25 winter_35_21 winter_35_37
winter_35_17 summer1_10p_21 winter_40_25 sonatineHa_81_36 sonatine_13p_35
summer1_11p_25 sonata_108_109 sonatineHa_32_26 sonata_33_1 sonatineHa_78_37
sonatine_16p_54 sonatineHa_8_1 sonata_149_80 summer1_18p_9 sonatine_14_1
sonatineHa_49_4 sonatineHa_81_40 sonatineHa_8_13 winter_34_1 sonata_149_68
winter_34_5 winter_34_9 summer_16p_37 sonata_99_110 sonatineHa_32_13""".split()

CLASSICAL_NEXT_20 = """chop39_3_9 spring_3p_31 sonatine_14p_9 spring_1p_5 summer_15p_25
sonatineHa_29_68 sonata_150_86 sonata_149_61 spring_2p_21 sonatineHa_27_36
sonata_84_4 fall_23_9 sonatine_41p_47 summer_14p_9 fall_24_32 sonatineHa_9_27
summer_16p_29 sonatine_14p_5 sonatineHa_28_49 sonata_100_135""".split()

CLASSICAL_EASY_50 = CLASSICAL_EASY_30 + CLASSICAL_NEXT_20

NEWAGE_EASY = "newage08 newage10 newage18 newage17 newage15 newage16 newage12 newage02 newage01 newage13".split()

assert len(CLASSICAL_EASY_50) == 50, len(CLASSICAL_EASY_50)
assert len(NEWAGE_EASY) == 10, len(NEWAGE_EASY)


def main():
    os.makedirs(OUT, exist_ok=True)
    n_songs = 0
    n_photos = 0
    for name in CLASSICAL_EASY_50 + NEWAGE_EASY:
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
