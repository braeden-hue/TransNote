"""
round3train/generate_scores.py  –  Round 3 대보표(Grand Staff) 데이터 생성

Round 2 기호 전체 포함 + 대보표:
  - treble(clef-G) + bass(clef-F) 두 파트를 한 시스템으로 생성
  - 토큰 구조: <SOS> clef-G key time [treble_measure] staff-bass [bass_measure] barline ...
  - staff-bass: treble 마디 끝, bass 마디 시작 사이에 삽입
  - MEASURES_MAX=4: 단일 시스템(treble 1행 + bass 1행) 보장

사용법:
    python round3train/generate_scores.py ^
        --count 4000 ^
        --output round3train/Round3 ^
        --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"

    python round3train/generate_scores.py ^
        --count 300 --start-idx 4001 ^
        --output round3train/Round3_test ^
        --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Optional

from music21 import (
    articulations, bar, chord as m21chord, clef, dynamics,
    expressions, key, layout, metadata, meter, note as m21note, spanner, stream, tie as tie_mod
)
from music21.note import Note, Rest
from music21.pitch import Pitch, Accidental
from music21.stream import Measure, Part, Score
from music21.chord import Chord

_MUSESCORE_CANDIDATES = [
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    "/usr/bin/musescore4", "/usr/bin/musescore3", "/usr/bin/musescore",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
]

def find_musescore(override=None):
    if override and Path(override).exists():
        return str(Path(override))
    for c in _MUSESCORE_CANDIDATES:
        if Path(c).exists():
            return c
    for cmd in ["musescore4", "musescore3", "musescore", "mscore4", "mscore3"]:
        found = shutil.which(cmd)
        if found:
            return found
    return None

_WIDE_PAGE_STYLE       = str(Path(__file__).resolve().parent / 'wide_page.mss')
_WIDE_PAGE_GRAND_STYLE = str(Path(__file__).resolve().parent / 'wide_page_grand.mss')

_MSS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<museScore version="4.00">
  <Style>
    <pageWidth>{page_width}</pageWidth>
    <pageHeight>{page_height}</pageHeight>
    <pagePrintableWidth>{printable_width}</pagePrintableWidth>
    <pageEvenLeftMargin>1</pageEvenLeftMargin>
    <pageOddLeftMargin>1</pageOddLeftMargin>
    <pageEvenTopMargin>1</pageEvenTopMargin>
    <pageEvenBottomMargin>1</pageEvenBottomMargin>
    <pageOddTopMargin>1</pageOddTopMargin>
    <pageOddBottomMargin>1</pageOddBottomMargin>
  </Style>
</museScore>
"""


def _make_dynamic_wide_page_style(n_measures: int, grand: bool) -> Path:
    """마디 수에 비례해 페이지 폭을 늘린 임시 .mss 스타일 파일 생성.

    기존 wide_page*.mss는 pagePrintableWidth가 11in 고정이라 MAX_MEASURES=4 기준으로
    맞춰져 있었음. 2026-07-31 MAX_MEASURES를 6으로 늘리고 화음/반주 패턴 등으로 내용이
    조밀해지면서, 고정 11in으로는 6마디가 한 시스템에 다 안 들어가고 MuseScore가 조용히
    2번째 시스템으로 줄바꿈해버리는 사고가 실측으로 확인됨(사용자 제보, num8500010.png:
    5마디는 1번째 줄, 마지막 1마디만 2번째 줄로 밀려남 -- system_breaks 라벨과 어긋남).
    마디당 약 2.5in로 스케일(최소 11in 유지, 기존 동작과 호환)."""
    printable_width = max(11.0, 2.5 * max(n_measures, 1))
    page_width = printable_width + 1.0
    page_height = 8 if grand else 4
    content = _MSS_TEMPLATE.format(page_width=page_width, page_height=page_height,
                                    printable_width=printable_width)
    tmp_style = Path(tempfile.gettempdir()) / f"_wide_page_dyn_{os.getpid()}_{n_measures}_{'g' if grand else 's'}.mss"
    tmp_style.write_text(content, encoding='utf-8')
    return tmp_style


def _looks_two_system(png_path: Path) -> bool:
    """렌더 결과가 한 시스템에 안 들어가고 2번째 시스템으로 줄바꿈됐는지 감지.
    행(row) 단위로 잉크(비-백색 픽셀) 유무를 스캔해서, 내용이 있는 두 구간이 상당한
    공백 구간(한 시스템 높이 이상)을 사이에 두고 분리돼 있으면 2시스템으로 판정
    (2026-07-31, num8500010.png 실측 사고 대응). PIL 없으면 이 검사는 건너뜀."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return False
    try:
        with Image.open(png_path) as im:
            arr = np.array(im.convert('L'))
        row_has_ink = (arr < 250).any(axis=1)
        ink_rows = np.where(row_has_ink)[0]
        if len(ink_rows) == 0:
            return False
        # 잉크가 있는 행들 사이에서, 공백이 이 이미지 높이의 15% 이상 연속되면
        # 별개 시스템 두 개로 판단(한 시스템 내부의 자연스러운 줄 간격보다 훨씬 큼).
        gap_threshold = arr.shape[0] * 0.15
        gaps = np.diff(ink_rows)
        return bool((gaps > gap_threshold).any())
    except Exception:
        return False


def render_png(musescore, xml_path, png_path, wide_page=False, grand=False, n_measures=None):
    """wide_page=True: MuseScore가 기본(A4 등) 페이지 폭 기준으로 내용이 많으면 자동
    2줄로 줄바꿈해버리는 것을 막기 위해 넓은 스타일을 강제 적용. music21 Score에
    layout.PageLayout을 직접 넣어봤지만 MuseScore4가 import 시 이를 무시하고 자체
    스타일로 재계산해서(2026-07-28 실측 확인) `-S` CLI 옵션으로 우회함.
    단일 오선/대보표 모두 항상 True로 호출됨(2026-07-30) -- 실제 카메라 캡처가 항상
    시스템 1개만 담으므로 학습 이미지도 항상 한 시스템이어야 함. 예전엔 대보표는
    "system_breaks로 줄바꿈을 직접 관리하니 불필요"라고 판단해 False로 뒀었는데,
    --density-break 없이도 내용이 조밀하면 MuseScore가 조용히 2번째 시스템으로
    줄바꿈해버리면서 system_breaks는 여전히 []로 남아 라벨-이미지가 어긋나는 사례를
    실측으로 확인함(4마디 중 4번째 마디가 둘째 줄로 밀려난 케이스).

    grand=True: wide_page_grand.mss(pageHeight=8in) 사용 -- wide_page.mss(pageHeight=4in)는
    단일 오선 1줄 기준 높이라 대보표(치+베이스 두 줄)에 쓰면 세로 공간이 부족해서 내용이
    통째로 2페이지로 밀려나고(우리가 캡처하는 1페이지는 빈 이미지가 됨) render_png()는
    "-1.png가 존재하니 성공"으로 잘못 판정하는 훨씬 심각한 사고가 남(2026-07-30 실측
    확인, 완전히 흰 이미지가 라벨과 함께 저장될 뻔함)."""
    tmp = png_path.with_name(png_path.stem + "_ms_tmp.png")
    cmd = [musescore]
    dyn_style = None
    if wide_page:
        if n_measures is not None:
            dyn_style = _make_dynamic_wide_page_style(n_measures, grand)
            style = str(dyn_style)
        else:
            style = _WIDE_PAGE_GRAND_STYLE if grand else _WIDE_PAGE_STYLE
        cmd += ["-S", style]
    cmd += ["-o", str(tmp), "-r", "150", str(xml_path)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except Exception:
        return False
    finally:
        if dyn_style is not None:
            try:
                dyn_style.unlink()
            except Exception:
                pass
    page1 = tmp.with_name(tmp.stem + "-1.png")
    src = page1 if page1.exists() else (tmp if tmp.exists() else None)
    if src is None:
        return False
    # 내용이 페이지 높이를 넘어가 2페이지로 밀려나면 우리가 캡처하는 1페이지가 완전히
    # 빈 이미지가 되는 사고가 실제로 있었음(2026-07-30, wide_page_grand.mss 도입 계기) --
    # wide_page 스타일로 방지했지만, 혹시 모를 극단적 케이스를 대비해 저장 직전 한 번 더
    # 확인한다. PIL이 없는 환경이면 이 방어는 건너뛰고 기존 동작대로 진행(경고만 출력).
    if _looks_blank(src):
        print(f"  [WARN] {png_path.stem}: 렌더 결과가 빈 이미지(내용이 페이지를 넘어갔을 가능성) -- 스킵")
        try:
            src.unlink()
        except Exception:
            pass
        return False
    # 2026-07-31: 페이지 높이는 안 넘었지만 폭이 부족해 2번째 "시스템"으로 조용히
    # 줄바꿈되는 사고(빈 페이지는 아니라 위 _looks_blank로는 안 걸림) 실측 확인
    # (num8500010.png: 6마디 중 마지막 1마디가 둘째 줄로 밀려남) -- 별도 검사로 방어.
    if _looks_two_system(src):
        print(f"  [WARN] {png_path.stem}: 렌더 결과가 2개 시스템으로 줄바꿈됨(페이지 폭 부족 추정) -- 스킵")
        try:
            src.unlink()
        except Exception:
            pass
        return False
    # replace() 사용(rename() 아님) -- Windows에서 Path.rename()은 목적지 파일이 이미
    # 있으면 조용히 덮어쓰지 않고 FileExistsError(WinError 183)를 던짐(POSIX rename()과
    # 다른 동작). 같은 인덱스로 재생성/재실행할 때(예: 이전 실행이 중간에 멈췄다가 재개)
    # 실제로 이 에러를 만남(2026-07-30). replace()는 두 플랫폼 다 덮어씀.
    src.replace(png_path)
    return True


def _looks_blank(png_path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        with Image.open(png_path) as im:
            extrema = im.convert('L').getextrema()
        return extrema[0] >= 250  # 거의 순백(빈 페이지)이면 실패로 취급
    except Exception:
        return False


def render_batch_png(musescore, jobs, wide_page=False, grand=False, chunk_size=100, n_measures=None):
    """jobs: (stem, xml_path, png_path) 튜플 리스트. render_png()를 이미지 개수만큼
    반복 호출하는 대신, MuseScore의 -j(배치 잡) 모드로 청크 하나당 프로세스 1개만 띄워서
    여러 장을 한 번에 처리한다.

    2026-07-30 도입 배경: 이미지 1장마다 MuseScore(Qt 앱 전체)를 새로 띄우는 오버헤드가
    실제 렌더링 연산보다 훨씬 컸음(실측: 개별 호출 15.2초/장 vs 배치 8.0초/장, user+sys
    CPU 시간은 배치 쪽이 real 시간에 훨씬 가까워짐 -- 대기 시간이 진짜로 줄어듦, 약 1.9배).

    chunk_size: 한 번의 -j 호출에 묶을 최대 장수. 너무 크게 묶으면 그 청크 안에서 뭔가
    잘못됐을 때(타임아웃 등) 청크 전체를 다시 해야 하는 위험이 커져서 적당히 나눔.
    render_png()와 동일한 안전장치(빈 페이지 감지 -> 스킵, Windows 덮어쓰기 안전한
    replace() 사용) 그대로 적용.

    n_measures: 지정하면(예: 이 배치를 생성한 --max-measures 값) 그 마디 수 기준으로
    페이지 폭을 넓힌 임시 스타일을 배치 전체에 균일 적용(2026-07-31 -- 고정 11in 폭이던
    wide_page*.mss로는 MAX_MEASURES=6+화음/반주 패턴 조밀화로 2번째 시스템 줄바꿈 사고가
    실측 확인됨, num8500010.png). 배치 안에서 곡마다 실제 마디 수가 달라도 한 스타일을
    같이 쓰므로, 짧은 곡은 여백이 좀 남을 뿐 문제는 없음.

    반환: 성공한 stem 집합(set)."""
    style = None
    dyn_style = None
    if wide_page:
        if n_measures is not None:
            dyn_style = _make_dynamic_wide_page_style(n_measures, grand)
            style = str(dyn_style)
        else:
            style = _WIDE_PAGE_GRAND_STYLE if grand else _WIDE_PAGE_STYLE
    ok = set()
    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start:start + chunk_size]
        tmp_map = {}
        job_entries = []
        for stem, xml_path, png_path in chunk:
            tmp = png_path.with_name(png_path.stem + "_ms_tmp.png")
            job_entries.append({"in": str(xml_path), "out": str(tmp)})
            tmp_map[stem] = (tmp, png_path)

        # 2026-07-30 수정: start는 이 샤드 내부의 로컬 청크 인덱스라서(항상 0부터 시작),
        # 여러 샤드를 병렬로 돌리면 전부 같은 출력 폴더에 "_batch_job_0.json"을 동시에
        # 써서 서로 덮어씀(N=6에서 5/6 샤드가 0장으로 실패한 진짜 원인 -- 타임아웃 문제가
        # 아니었음). PID까지 넣어 샤드 간 파일명이 절대 겹치지 않게 함.
        job_path = chunk[0][2].parent / f"_batch_job_{os.getpid()}_{start}.json"
        with open(job_path, 'w', encoding='utf-8') as f:
            json.dump(job_entries, f)

        cmd = [musescore]
        if style:
            cmd += ["-S", style]
        # 2026-07-31: -r 없이 -j만 쓰면 MuseScore가 기본 해상도(1200dpi)로 렌더해
        # render_png()의 150dpi 대비 가로/세로 각 8배(=픽셀 수 64배) 큰 이미지가 나옴 --
        # step1_pool 6019장 전체가 평균 raw 516MB/장이 돼 학습 시작 시 OMRDataset의
        # ThreadPoolExecutor 전처리에서 즉시 cgroup 메모리 한도(50GB)를 넘겨 OOM-kill
        # (exit 137) 되던 진짜 원인이었음(--workers 튜닝은 무관했음, 실측으로 확인).
        cmd += ["-r", "150", "-j", str(job_path)]
        try:
            # 2026-07-30 수정: 4초/장은 실측 배치 단일스레드 속도(6.2~10.6초/장)보다도
            # 느려서 타임아웃이 청크 전체를 죽이는 사고가 발생함(N=6 병렬 테스트에서
            # 6개 샤드 중 3개가 0/200으로 전멸). 병렬 경합까지 감안해 20초/장으로 상향.
            subprocess.run(cmd, capture_output=True, text=True,
                            timeout=max(120, 20 * len(chunk)))
        except Exception:
            pass  # 개별 성공/실패는 아래에서 결과 파일 존재 여부로 판정
        finally:
            try:
                job_path.unlink()
            except Exception:
                pass

        for stem, (tmp, png_path) in tmp_map.items():
            page1 = tmp.with_name(tmp.stem + "-1.png")
            src = page1 if page1.exists() else (tmp if tmp.exists() else None)
            if src is None:
                continue
            if _looks_blank(src):
                try:
                    src.unlink()
                except Exception:
                    pass
                continue
            if _looks_two_system(src):
                try:
                    src.unlink()
                except Exception:
                    pass
                continue
            src.replace(png_path)
            ok.add(stem)
    if dyn_style is not None:
        try:
            dyn_style.unlink()
        except Exception:
            pass
    return ok


# ─── 설정 ─────────────────────────────────────────────────────────────────────

KEY_SIGS   = [(0,'C'),(1,'G'),(-1,'F'),(2,'D'),(-2,'Bb'),(3,'A'),(-3,'Eb'),
              (4,'E'),(-4,'Ab'),(5,'B'),(-5,'Db'),(6,'F#'),(-6,'Gb')]
KS_WEIGHTS = [0.20, 0.14, 0.14, 0.10, 0.10, 0.07, 0.07, 0.05, 0.05, 0.03, 0.03, 0.01, 0.01]

TIME_SIGS  = [(4,4),(3,4),(2,4),(6,8)]
TS_WEIGHTS = [0.35, 0.30, 0.20, 0.15]  # 2026-08-01: 3/4 비중 절충 상향(exactPicture 실사
                                        # 66% vs 기존 25% 격차, 특정 레퍼토리 과적합 방지 위해 소폭만)

# 지정 시 매 N마디마다 강제 줄바꿈(SystemLayout isNew=True) 삽입 -- MuseScore의 자동
# 줄바꿈은 마디 수를 균등하게 안 나누기 때문에(예: 8마디->6+2), dataset.py의 시스템별
# 토큰 분할 로직(barline 기준 균등 분배 가정)과 어긋난다. 이 값을 쓰면 항상 정확히
# N마디씩 나뉘어서 균등분할 가정이 실제로 성립한다.
MEASURES_PER_SYSTEM = None

# --auto-measures-per-system 사용 시: --min/--max-measures를 무작위 범위(예: 1~4)로 줄 때도
# 샘플마다 실제 뽑힌 n_measures에 맞춰 시스템당 마디 수를 자동으로 정한다(3마디 이하는
# 한 시스템에 그대로, 4마디부터는 절반씩 2시스템) -- 홀수 마디가 섞여도 균등분할이 항상
# 성립하도록 보장. round1_curriculum 4h~4i 단계에서 검증된 안전 범위(단일 시스템 최대
# 2~3마디, 4마디부터 줄바꿈)를 그대로 반영.
AUTO_MEASURES_PER_SYSTEM = False


def _effective_measures_per_system(n_measures: int):
    if AUTO_MEASURES_PER_SYSTEM:
        return n_measures if n_measures <= 3 else 2
    return MEASURES_PER_SYSTEM

DURATIONS  = [
    # 실제 사람이 쓰는 일반적인 악보에 가깝게: 점음표(3/8, 3/16)는 확실히 줄이고,
    # 8분/16분음표는 상대적으로 적게만 줄여서 빠진 확률을 4분/2분/온음표 쪽으로 옮김.
    # 점음표(3/16 < 3/8 < 3/4)는 음가가 길수록(점8분→점4분→점2분) 확률도 오름차순으로
    # 커지되, 항상 대응하는 비점음표(1/8, 1/4, 1/2)보다는 확실히 낮게(2026-07-31 사용자
    # 지시). 2026-07-31 후속 지시: 음표가 마디당 너무 많이 분포한다는 피드백 -- 4분음표
    # 비중을 0.45->0.55로 더 올리고, 8분음표를 0.25->0.15로 낮춰서(그만큼 짧은 음의
    # 절대량이 줄어 마디당 평균 음표 수가 감소) 전체적으로 덜 촘촘하게 조정.
    (4.0,'1/1',0.04),(3.0,'3/4',0.02),(2.0,'1/2',0.16),(1.5,'3/8',0.015),
    (1.0,'1/4',0.55),(0.5,'1/8',0.15),(0.25,'1/16',0.08),(0.75,'3/16',0.005),
]

SHORT_NOTE_BIAS = 0.0   # 0=끔(DURATIONS 가중치 그대로). >0이면 1/8 이하 짧은 음의 가중치를
                        # (1+bias)배로 올려 마디당 음표 개수(=음표 간 간격이 좁은 밀집 마디)
                        # 비율을 늘림. 마디마다 이 값이 독립 추첨되는 게 아니라 전역 확률
                        # 자체를 옮기는 것이라, 마디별 음표 수 편차(밀집/희소 마디가 한
                        # 시스템에 섞이는 것 = 실제 악보처럼 마디 폭이 들쭉날쭉해지는 것)는
                        # 기존처럼 마디마다 독립 추첨되는 난수에서 자연히 생김.
LONG_NOTE_BIAS = 0.0    # 0=끔. >0이면 1/4 이상 긴 음(1/4, 1/2, 1/1)의 가중치를 (1+bias)배로
                        # 올려 마디당 음표 개수를 줄임 -- SHORT_NOTE_BIAS의 반대 방향
                        # (2026-07-31 사용자 요청: 오선 검출 강화용 밀집 묶음 데이터와
                        # 별개로, 전체 음표 수를 줄인 "희소" 데이터 축을 분리 생성).

EIGHTH_RUN_PROB = 0.0    # 0=끔. >0이면 remaining>=2박일 때 이 확률로 8분음표 4개(정확히
                          # 2박)를 강제 배치 -- 매번 다시 굴려서 한 마디 안에 여럿 나올 수
                          # 있음. 실제 악보에 흔한 "8분음표 4개 묶음(beam)" 형태와 밀집
                          # 16분음표 구간의 오선 검출 학습 강화용(2026-07-31 사용자 요청).
SIXTEENTH_RUN_PROB = 0.0  # 0=끔. >0이면 remaining>=1박일 때 이 확률로 16분음표 4개(정확히
                          # 1박)를 강제 배치 -- EIGHTH_RUN_PROB와 동일 취지.
EIGHTH_RUN_PROB_2_4    = None  # None=EIGHTH_RUN_PROB와 동일. 지정되면 2/4 박자 파트에서만
                                # 이 값을 대신 씀(2026-07-31 사용자 요청 -- 2/4는 16분음표
                                # 묶음이 8분음표 묶음보다 더 자주 나오게, 나머지 박자는
                                # 8분음표 묶음 비율만 기본값보다 살짝 올림).
SIXTEENTH_RUN_PROB_2_4 = None  # None=SIXTEENTH_RUN_PROB와 동일. 2/4 박자 전용 오버라이드.

COURTESY_ACCIDENTAL_PROB = 0.0  # 0=끔. >0이면 음표(또는 화음의 각 구성음)마다 이 확률로
                                 # "굳이" 임시표를 시각적으로 강제 표시 -- 자연음이면
                                 # 내추럴 기호를, 이미 임시표가 있는 음이면 강제 표시 유지.
                                 # 실제 악보에서 조표/앞 마디의 임시표로 이미 자명한 음에도
                                 # 확인용으로 임시표를 붙이는 경우가 있음(2026-07-31 사용자
                                 # 요청). pitch.nameWithOctave(=note-{pitch} 토큰 문자열)는
                                 # 안 바뀌므로 라벨/vocab에는 영향 없는 순수 시각적 증강.

EIGHTH_BIAS = 0.0       # 0=끔(dur-1/8 기본 가중치 15% 그대로). >0이면 8분음표(1/8)만 (1+bias)배로
                        # 올림. SHORT_NOTE_BIAS(1/8+1/16 동시 가중)와 달리 1/8만 따로 조절할 때 사용
                        # -- 실사 촬영 곡 GT 실측(2026-08-04)에서 1/8이 전체 duration의 52%(합성
                        # 기본값은 15%)로 가장 저평가된 구간이라 SIXTEENTH_BIAS와 분리해 개별 조정.
SIXTEENTH_BIAS = 0.0    # 0=끔(dur-1/16 기본 가중치 8% 그대로). >0이면 16분음표(1/16)만 (1+bias)배로
                        # 올림. EIGHTH_BIAS와 동일 취지, 실사 실측 18.8% 목표.
DOTTED8_BIAS = 0.0      # 0=끔(dur-3/16 기본 가중치 0.5% 그대로). >0이면 점8분음표(3/16) 가중치를
                        # (1+bias)배로 올려 "점8분+16분음표" 리듬 셀의 노출 빈도를 높임. 기본
                        # 가중치가 워낙 낮아(1/16의 8%, 3/8의 1.5%에 비해서도 낮은 0.5%) 이
                        # 리듬 자체가 학습 데이터에 거의 등장하지 않던 것을 보정하기 위함
                        # (2026-07-28 error_breakdown 점검에서 확인).

RARE_LONG_BIAS = 0.0    # 0=끔. >0이면 온음표(1/1, 기본 4%)와 점2분음표(3/4, 기본 2%)만
                        # (1+bias)배로 올림. 기존 LONG_NOTE_BIAS는 1/4·3/8·1/2까지 전부 같이
                        # 올려서(이미 55%인 4분음표까지 부풀림) 이 두 희귀 duration만 콕 집어
                        # 노출을 늘리기엔 부적합해서 별도로 분리(2026-08-02, 실사 89곡+신규
                        # newage 라벨 확인 결과 dur-3/4 실사 존재율 대비 합성 노출 부족 확인).

CROSS_REGISTER_PROB = 0.0  # 0=끔(항상 treble=TREBLE_PITCHES, bass=BASS_PITCHES). >0이면
                        # 이 확률로 대보표 한 쌍의 음역을 비정상적으로 바꿔서(swap: treble이
                        # 베이스 음역/bass가 치음역, both_high: 둘 다 치음역, both_low: 둘 다
                        # 베이스 음역) 덧줄이 많은 "교차 음역" 케이스를 생성. 2026-07-28
                        # error_breakdown 점검에서 이런 덧줄 많은 케이스일 때 모델이 실제
                        # 음높이 대신 그 음자리표의 "전형적 음역"으로 회귀하는 편향을 확인해서
                        # 추가함 (클렙 자체는 항상 정상 표기 -- clef-G가 위, clef-F가 아래인
                        # 건 안 바뀌고, 그 안에 실제로 찍히는 피치 풀만 바뀜).

CLEF_CHANGE_PROB = 0.0  # 0=끔. >0이면 이 확률로 한 오선 안에서 마디 중간에 반대쪽
                        # 클렙(치↔베이스)으로 전환했다가 이후 마디에서 다시 되돌아오는
                        # 표준 표기를 생성 (예: 치보표가 극저음 구간에서 마디 중간에
                        # 낮은음자리로 잠시 전환 후 복귀). --single-staff와 대보표 둘 다
                        # 지원(2026-07-30 대보표 추가) -- 대보표는 치/베이스 "둘 중 하나"에만
                        # 적용(양쪽 동시 전환은 안 함). 새 vocab 토큰 불필요 --
                        # 기존 clef-G/clef-F 토큰이 시퀀스 중간에 재등장하는 것으로 표현
                        # (docs/music-notation-rule-designer.md "마디 중간 클렙 전환 표기"
                        # 참고). 대보표(build_score_r3)에는 적용 안 함 -- staff-bass 인터리빙
                        # 구조·dataset.py의 대보표 분할 로직과 상호작용을 아직 검증하지 않았고,
                        # 확정된 정의도 "한 보표 내부"이므로 단일 오선 경로로 범위를 한정함.

SAME_CLEF_PROB = 0.0  # 0=끔. >0이면 이 확률로 대보표 시스템 전체가 같은 clef(둘 다
                        # clef-G 또는 둘 다 clef-F)로 나옴 -- CLEF_CHANGE_PROB의 "마디 중간
                        # 일시 전환"과 달리 시스템 처음부터 끝까지 같은 clef가 유지됨.
                        # 2026-08-10 추가: r15가 "위=치/아래=베이스" 데이터로만 학습돼서
                        # 실제로 같은-clef 대보표를 만나면 clef 기호를 무시하고 위치만으로
                        # 오인식하는 걸 실측으로 확인함(train/docs/PLAN_r18_grand_staff_scope.md).
                        # 트리거되면 두 파트 다 같은 clef_obj/clef_tok을 쓰되, 시각적으로
                        # 구분되게 피치 풀은 CROSS_REGISTER_PROB용 TREBLE_LOW_PITCHES/
                        # BASS_HIGH_PITCHES를 재사용해 위/아래를 음역으로 분리함. 마디 중간
                        # 클렙 전환(CLEF_CHANGE_PROB)과는 상호작용 미검증이라 동시 적용 안 함
                        # (같은-clef가 뽑히면 CLEF_CHANGE_PROB 쪽이 자동으로 꺼짐).

TIE_PROB = 0.0  # 0=끔. >0이면 마디마다 이 확률로 "이 마디 안에서 붙임줄을 하나 시작한다"를
                # 결정하고, 그 마디를 생성하는 도중 처음 마주치는 이어붙이기 가능한 음표/
                # 2음 화음에서 실제로 시작함(다음에 생성되는 음표/화음을 같은 피치로 강제해
                # 이어붙임). 시작 지점이 마디 마지막 음표면 자연히 다음 마디로 넘어가는
                # 붙임줄이 되고, 마디 중간 음표면 같은 마디 안에서 끝나는 붙임줄이 됨
                # (2026-07-30, 사용자 요청으로 두 경우 다 지원 -- 예전엔 마디 끝에서만 시작
                # 가능했음). vocab에 없던 개념이라 새 토큰 `tie` 1개를 추가함(258->259,
                # tokenizer258.json). 토큰은 항상 "이 음표가 바로 다음 음표로 이어진다"는
                # 뜻으로 이어지는 음표 쪽(끝)에는 붙지 않음(체인 길이 항상 2 -- 3개 이상
                # 이어지는 tie는 미지원). 피치 아닌 마지막 마디의 마지막 음표에서는 시작
                # 안 함(이어붙일 다음 자리가 아예 없음). _build_part()가 single-staff/
                # grand-staff 양쪽에서 공유되므로 두 경로 모두에 적용됨. 마디 중간 클렙
                # 전환(clef_events)이 있는 파트에서는 상호작용을 검증하지 않아 자동으로 꺼짐.
CHORD_TIE_SUBPROB = 0.15  # 마디가 붙임줄을 시작하기로 한 상태에서 마주친 기회가 2음
                          # 화음이면 이 확률로만 실제로 화음-화음 붙임줄을 시작(그 외엔
                          # 소비하지 않고 다음 기회 -- 음표든 화음이든 -- 를 기다림). 3음
                          # 화음은 애초에 이어붙이기 후보에서 제외. 단음-단음 붙임줄보다
                          # 현저히 드물게 하려는 의도(2026-07-30 사용자 요청).

DYNAMICS_LIST = ['pp','p','mp','mf','f','ff','fp']
ARTICS        = ['staccato','accent','tenuto','marcato']
ORNAMENTS     = ['trill','mordent','turn']

REST_PROB     = 0.08  # 기존 0.15 -- 무작위 생성 특유의 과도한 쉼표 밀도 완화
HIDE_TIMESIG_PROB = 0.0  # 2026-08-04: 실사 촬영은 곡 중간부터 찍혀 박자표 기호가 안 보이는
                          # 경우가 흔한데(라벨엔 여전히 정답 time-* 토큰 존재), 합성 데이터는
                          # 지금까지 전부 1번째 마디에 박자표를 렌더링해 항상 보이는 상태로만
                          # 학습돼왔다 -- 박자표發 초반 이탈 캐스케이드(diag_new6_analysis
                          # 첫 이탈 토큰 36.8%가 time)의 유력한 원인. >0이면 이 확률로 해당
                          # 점수의 박자표 기호를 렌더링에서만 숨긴다(라벨은 정상 유지 --
                          # 마디 안 음표 박자를 세어 알아내야 하는 상황을 재현).
CHORD_PROB    = 0.08  # 기존 0.15 -- 화음 밀집도 완화
CHORD_MIN_NOTES   = 2
CHORD_MAX_NOTES   = 3
CHORD_TWO_NOTE_PROB = 0.5  # 화음 노트 수를 CHORD_MIN_NOTES(2개)로 고정할 확률 -- 0.5=기존
                           # random.randint(2,3)과 동일한 균등분포. 이보다 올리면 2음 화음
                           # 비중이 늘고, 나머지 확률로는 CHORD_MIN_NOTES+1~CHORD_MAX_NOTES에서
                           # 균등 선택(현재 범위 2~3이면 곧 3개 고정). CHORD_SIZE_WEIGHTS가
                           # 설정되면 이 값 대신 그쪽이 우선 적용됨.
CHORD_SIZE_WEIGHTS = None  # None=끔(CHORD_TWO_NOTE_PROB로 폴백). dict로 주어지면(예:
                           # {2:0.6, 3:0.3, 4:0.1}) 화음 노트 개수(2/3/4개)를 이 가중치로
                           # 직접 선택 -- CHORD_MIN_NOTES~CHORD_MAX_NOTES 범위로 클램프
                           # (2026-07-31 사용자 요청: 2/3/4개 화음을 내림차순 확률로).
CHORD_INTERVAL_WEIGHTS = None  # None=끔(기존처럼 루트 기준 화음 후보를 균등 샘플). dict로
                               # 주어지면(예: {2:0.6, 3:0.3, 4:0.1}) 루트와 후보음 사이 음정
                               # 도수(2도/3도/4도, 옥타브 넘어가면 mod 7로 단순화)에 따라
                               # 가중 샘플. 도수가 딕셔너리에 없으면 기본 EPS 가중치.
_CHORD_INTERVAL_WEIGHT_EPS = 0.02
CHORD_MAX_INTERVAL = 12   # 화음 내 최저-최고음 간격 상한(반음 수) -- 기본 1옥타브(8도)
DYNAMIC_PROB  = 0.35
HAIRPIN_PROB  = 0.20
ARTIC_PROB    = 0.10  # 기존 0.18 -- 줄인 만큼 OTTAVA_PROB로 이전
ORNAMENT_PROB = 0.03  # 기존 0.05 -- 줄인 만큼 OTTAVA_PROB로 이전
FERMATA_PROB  = 0.04
SLUR_PROB     = 0.12
TUPLET_PROB   = 0.06  # 2026-07-31: Round1~3 커리큘럼은 --tuplet-prob 0.35로 덮어써 썼으나,
                      # Round4부터는 셋잇단음표 노출 확률을 절반(0.175)으로 낮추기로 함(사용자
                      # 지시) -- 이 모듈 기본값도 그 비율(기존 0.12의 절반)로 맞춰 둠.
TUPLET_LEDGER_PROB = 0.0  # 0=끔(항상 정상 음역 안에서만, 기존 동작). >0이면 3연음 전체가
                          # 이 확률로 오선 밖(최대 두 줄, _extend_pool_by_semitones 참고)까지
                          # 확장된 음역에서 뽑힘 -- 나머지 확률은 기존처럼 정상 음역 고정
                          # (2026-07-31 사용자 요청: 0.7은 정상 음역, 0.3은 오선 밖 확장).
TUPLET_REST_PROB = 0.0    # 0=끔. >0이면 3연음 3개 슬롯 중 하나(무작위)가 이 확률로 음표
                          # 대신 쉼표(rest-1/8 토큰, 실제 길이는 1/3박)가 됨(2026-07-31
                          # 사용자 요청).
OTTAVA_PROB   = 0.18  # 기존 0.08 -- ARTIC_PROB/ORNAMENT_PROB 감소분(0.10)만큼 증가.
                      # 오선 밖에 그려지는 기호라 인식 시 크롭 잘림 위험 있음 -- dataset.py의
                      # MARGIN_UNITS_CAP(콘텐츠 기준 마진 확장)으로 대응 완료.
REPEAT_PROB   = 0.12

MIN_MEASURES  = 2
MAX_MEASURES  = 6

# 커리큘럼 난이도 프로파일 — easy: 음표/쉼표 위주(그랜드 스태프 정렬 먼저 학습),
# medium: 기호 절반 밀도, full: 위 기본값(기존 동작 그대로, --difficulty 미지정 시 기본).
DIFFICULTY_PROFILES = {
    'full': {},
    'medium': {
        'CHORD_PROB': 0.08, 'DYNAMIC_PROB': 0.20, 'HAIRPIN_PROB': 0.10,
        'ARTIC_PROB': 0.10, 'ORNAMENT_PROB': 0.03, 'FERMATA_PROB': 0.02,
        'SLUR_PROB': 0.06, 'TUPLET_PROB': 0.06, 'OTTAVA_PROB': 0.04,
        'REPEAT_PROB': 0.12,
    },
    'easy': {
        'CHORD_PROB': 0.0, 'DYNAMIC_PROB': 0.0, 'HAIRPIN_PROB': 0.0,
        'ARTIC_PROB': 0.0, 'ORNAMENT_PROB': 0.0, 'FERMATA_PROB': 0.0,
        'SLUR_PROB': 0.0, 'TUPLET_PROB': 0.0, 'OTTAVA_PROB': 0.0,
        'REPEAT_PROB': 0.10,
    },
}


def _pitch_pool(lo, hi):
    lo_m = Pitch(lo).midi; hi_m = Pitch(hi).midi
    pcs  = ["C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"]
    pool = []
    for oct in range(1, 7):   # 1부터 시작(현재는 모든 호출이 C2 이상만 써서 옥타브 1은
                               # 실제로 안 걸리지만, lo/hi 필터가 알아서 걸러주므로 무해함)
        for pc in pcs:
            try:
                p = Pitch(f"{pc}{oct}")
                if lo_m <= p.midi <= hi_m:
                    pool.append(p.nameWithOctave)
            except Exception:
                pass
    return pool


def _extend_pool_by_semitones(pool: list, semitones: int) -> list:
    """pool의 최저~최고 MIDI 범위를 위아래로 semitones만큼 확장한 새 피치 목록
    (vocab 최저음 C2(midi 36)~최고음 B6(midi 95)로 클램프). 셋잇단음표가 오선 밖으로
    "최대 두 줄까지만" 확장될 때 씀(2026-07-31 사용자 요청) -- 완전5도(7반음) 정도를
    "덧줄 2개"의 근사치로 사용."""
    if not pool:
        return pool
    midis = [Pitch(p).midi for p in pool]
    lo_m = max(36, min(midis) - semitones)
    hi_m = min(95, max(midis) + semitones)
    pcs = ["C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"]
    out = []
    for oct in range(1, 7):
        for pc in pcs:
            try:
                p = Pitch(f"{pc}{oct}")
                if lo_m <= p.midi <= hi_m:
                    out.append(p.nameWithOctave)
            except Exception:
                pass
    return out


TREBLE_PITCHES = _pitch_pool("C4", "B5")
BASS_PITCHES   = _pitch_pool("C2", "B3")

# CROSS_REGISTER_PROB 전용(2026-07-30) -- swap/both_high/both_low 모드가 치보표를
# BASS_PITCHES(바닥 C2)로, 베이스보표를 TREBLE_PITCHES(천장 B5)로 그대로 바꿔치기하면
# 사용자가 range.mscz로 지정한 상한/하한(높은음자리 C3~B6, 낮은음자리 E1~B4)을 벗어남 --
# "덧줄 많은 극단 케이스"는 원하지만 지나치게 낮은/높은 음역까지는 원하지 않아서, 그
# 지점(C3/B4)에서 잘라낸 별도 풀을 씀. 정상 풀(TREBLE_PITCHES/BASS_PITCHES) 자체는 이미
# 범위 안(C4~B5 ⊂ C3~B6, C2~B3 ⊂ E1~B4)이라 그대로 둠.
TREBLE_LOW_PITCHES  = _pitch_pool("C3", "B3")   # 치보표가 낮은 음역을 보일 때(swap/both_low)
BASS_HIGH_PITCHES   = _pitch_pool("C4", "B4")   # 베이스보표가 높은 음역을 보일 때(swap/both_high)

# PREFERRED_REGISTER_PROB 전용(2026-07-30, range.mscz 참고) -- 사용자가 지정한 "덧줄이
# 많이 필요한" 선호 구간 두 곳(각 클렙 own 오선에서 멀리 떨어진 낮은쪽/높은쪽) 합집합.
# 치: D3~A3(오선 아래 덧줄) ∪ A5~A6(오선 위 덧줄). 베이스: C2~B2(오선 아래 덧줄) ∪
# F4~B4(오선 위 덧줄) -- 사용자가 원래 지정한 건 G1~C2였지만 G1은 tokenizer258.json
# vocab 범위 밖(최저 음이 C2, mscz_to_tokens.py의 PC_MIN도 C2)이라 <UNK>로 깨질 위험이
# 있어 vocab 최저인 C2부터 시작하는 C2~E2로 대체(사용자 확인 완료). 이후(2026-07-30,
# markov 단계 도입 시점) 베이스 저음역이 range.mscz 원래 의도(E1~B4)보다 여전히 좁다는
# 피드백으로 상한을 5도 더 올림(E2 -> B2, C2 floor는 vocab 제약이라 그대로) -- 이 변경은
# 코드 시점부터 적용되므로 이미 생성된 6reg1/tie1/7den1엔 영향 없고, markov 단계부터의
# 신규 생성에 반영됨.
# _pick_pitch()가 PREFERRED_REGISTER_PROB 확률로 후보를 이 집합과의 교집합으로 좁힌다
# (현재 pool과 교집합이 비면 원래 pool 유지 -- cross-register로 이미 극단 풀을 쓰는
# 경우에도 안전하게 동작).
TREBLE_PREFERRED_PITCHES = set(_pitch_pool("D3", "A3")) | set(_pitch_pool("A5", "A6"))
BASS_PREFERRED_PITCHES   = set(_pitch_pool("C2", "B2")) | set(_pitch_pool("F4", "B4"))


def _naturals_only(pool):
    """샵(#)/플랫(-) 임시표가 붙은 음이름 제외 (자연음만)."""
    return [p for p in pool if '#' not in p and '-' not in p]


TREBLE_PITCHES_NATURAL       = _naturals_only(TREBLE_PITCHES)
BASS_PITCHES_NATURAL         = _naturals_only(BASS_PITCHES)
TREBLE_LOW_PITCHES_NATURAL   = _naturals_only(TREBLE_LOW_PITCHES)
BASS_HIGH_PITCHES_NATURAL    = _naturals_only(BASS_HIGH_PITCHES)
TREBLE_PREFERRED_NATURAL     = set(_naturals_only(TREBLE_PREFERRED_PITCHES))
BASS_PREFERRED_NATURAL       = set(_naturals_only(BASS_PREFERRED_PITCHES))


def _np(name: str) -> str:
    return name.replace('-', 'b')


DIATONIC_BIAS = 0.75   # 조표 음계에 속한 음을 우선 고를 확률 (나머지는 임시표 포함 자유 선택)
_DIATONIC_PC_CACHE: dict = {}


def _diatonic_pitch_classes(ks_sharps: int) -> set:
    """조표(ks_sharps, -# ~ +#)의 장조 음계에 속한 (pitchClass, 철자이름) 쌍의 집합.
    pitchClass만 맞추면 D장조인데 'Db'(원래는 C#)처럼 음은 맞아도 조표 관례와 다른
    철자가 나올 수 있어서, 철자(스텝+임시표, 옥타브 제외)까지 함께 맞춘다.
    같은 결과를 매 마디/매 음표마다 다시 계산하지 않도록 캐싱."""
    if ks_sharps not in _DIATONIC_PC_CACHE:
        scale = key.KeySignature(ks_sharps).getScale('major').pitches
        _DIATONIC_PC_CACHE[ks_sharps] = {(p.pitchClass, p.step + (p.accidental.modifier if p.accidental else ''))
                                          for p in scale}
    return _DIATONIC_PC_CACHE[ks_sharps]


MAX_SYSTEM_WEIGHT = 6.0   # 시스템(줄) 하나에 담을 수 있는 밀도 상한 (기존 8.0 -- 무작위
                          # 생성 특유의 과밀 방지, 줄바꿈을 더 자주 일으켜 시스템당 덜 빽빽하게)
                          # (음표/쉼표 이벤트=1, 화음에 딸린 추가 음=0.5)
DENSITY_BREAK = False     # True면 마디 개수 대신 내용 밀도 기준으로 줄바꿈 결정
                          # (실제 조판자처럼 화음/장식음이 많으면 한 줄에 마디를 적게 담음)


def _measure_weight(m_tok: list) -> float:
    """마디 하나의 시각적 밀도 추정치. 실제 렌더링 폭과 강하게 상관되는 이벤트
    개수를 기준으로 하되(이번 세션에서 확인된 사실 -- 8분음표 8개=4분음표 4개의
    거의 2배 폭), 화음은 덧붙는 음마다 약간의 추가 폭만 잡는다(세로로 쌓이므로
    새 이벤트만큼은 아님)."""
    w = 0.0
    for t in m_tok:
        if t.startswith('note-') or t.startswith('rest-'):
            w += 1.0
        elif t.startswith('chord-'):
            w += 0.5
    return w


def _decide_system_breaks(t_toks: list, b_toks: list, max_weight: float) -> list:
    """치/베이스 각 마디 밀도(둘 중 더 복잡한 쪽 기준)를 누적하다가 상한을 넘기
    직전에 새 시스템을 시작한다. 반환값은 새 시스템이 시작되는 마디 인덱스 목록
    (0번째 마디는 항상 시작이므로 포함 안 함) -- 마디 개수가 아니라 실제 내용
    밀도 기준이라 각 시스템의 마디 수가 다를 수 있다(홀수 마디도 안전하게 처리)."""
    n = len(t_toks)
    if n == 0:
        return []
    weights = [max(_measure_weight(t_toks[i]), _measure_weight(b_toks[i])) for i in range(n)]
    breaks = []
    cum = weights[0]
    for i in range(1, n):
        if cum + weights[i] > max_weight:
            breaks.append(i)
            cum = weights[i]
        else:
            cum += weights[i]
    return breaks


MELODIC_BIAS     = 0.0   # 이전 음에서 MELODIC_MAX_STEP 반음 이내로 다음 음을 고를 확률
                          # (기본 0=끔, 기존 동작 그대로 -- 매 음을 이전 음과 무관하게 독립
                          # 추첨). 실제 쇼팽 곡 GT에서 보이는 반음계 진행(예: C5-Db5-B4-C5)
                          # 같은 매끄러운 선율선은 이 방식이 없으면 절대 안 나옴.
MELODIC_MAX_STEP = 4      # MELODIC_BIAS 적용 시 "가까운 음" 취급 반음 상한(4=장3도 이내)

MARKOV_BIAS  = 0.0   # 이전 음 -> 다음 음의 다이어토닉 음정을, PDMX 실제곡 통계 확률표로
                      # 가중 추첨할 확률(기본 0=끔). MELODIC_BIAS/DIATONIC_BIAS는 "가까운
                      # 음 우선"/"조표 음계 우선" 정도의 손으로 정한 규칙이지만, 이건
                      # build_markov_transitions.py가 실제 작곡된 곡(PDMX 9845곡)에서 집계한
                      # "직전 음 다음에 실제로 어떤 음정이 오는지" 분포를 그대로 씀 --
                      # 2026-07-30 사용자 피드백("음의 규칙성이 없다") 대응.
MARKOV_TABLE = None   # {다이어토닉 스텝(int): 확률} -- --markov-table로 로드
MARKOV_MAX_INTERVAL = 14  # 로드한 테이블의 max_interval로 덮어씀(build_markov_transitions.py와 동일 값이어야 함)

PREFERRED_REGISTER_PROB = 0.0   # 0=끔(기본). >0이면 이 확률로 후보를 preferred_pool
                                 # (호출자가 넘긴 TREBLE_PREFERRED_PITCHES/
                                 # BASS_PREFERRED_PITCHES)과의 교집합으로 좁힘 -- "덧줄이
                                 # 많이 필요한 구간" 노출을 높이기 위함(2026-07-30,
                                 # range.mscz 참고).

CHORD_PROGRESSION_BIAS = 0.0   # 0=끔(기본). >0이면 마디마다 정해지는 암묵적 화성(장음계
                                # 디그리 1~7 기준 다이어토닉 3화음, _next_degree()가 흔한
                                # 화성 진행 패턴으로 다음 디그리를 고름)의 구성음(근음/3도/
                                # 5도) 쪽으로 이 확률만큼 음표 후보를 좁힘 -- "무작위가
                                # 아니라 코드를 따라가는 멜로디"를 만들기 위함(2026-07-31,
                                # 사용자 요청). 화음(chord- 토큰) 자체와는 별개 -- 이건 각
                                # 음표(단음이든 화음의 근음이든)가 그 순간의 암묵적 화성과
                                # 얼마나 어울리는 음이름을 쓰는지에 대한 것.
_CHORD_DEGREE_TRANSITIONS = {
    # 대중적인 종지/진행 패턴(T-S-D-T 기능화성 감각)을 아주 단순화한 디그리(1~7) 전이표.
    # 정밀한 음악이론 모델이 아니라 "완전 무작위보다는 그럴듯한 진행" 정도의 근사치.
    1: {1: 0.10, 4: 0.25, 5: 0.25, 6: 0.25, 2: 0.15},
    2: {5: 0.50, 4: 0.20, 1: 0.10, 7: 0.20},
    3: {6: 0.40, 4: 0.30, 1: 0.30},
    4: {5: 0.35, 1: 0.30, 2: 0.15, 6: 0.20},
    5: {1: 0.45, 6: 0.25, 4: 0.15, 5: 0.15},
    6: {4: 0.30, 2: 0.25, 5: 0.25, 1: 0.20},
    7: {1: 0.60, 5: 0.20, 3: 0.20},
}


def _next_degree(cur_degree: int) -> int:
    trans = _CHORD_DEGREE_TRANSITIONS.get(cur_degree, {1: 1.0})
    degrees = list(trans.keys())
    weights = list(trans.values())
    return random.choices(degrees, weights=weights)[0]


def _chord_tones_for_degree(ks_sharps: int, degree: int, pool: list) -> list:
    """조표(ks_sharps)의 장음계에서 degree(1~7)를 근음으로 하는 다이어토닉 3화음의
    피치클래스 3개(근음/3도/5도)와 일치하는 pool 내 음이름들을 반환."""
    scale = key.KeySignature(ks_sharps).getScale('major').pitches
    root_pc  = scale[(degree - 1) % 7].pitchClass
    third_pc = scale[(degree + 1) % 7].pitchClass
    fifth_pc = scale[(degree + 3) % 7].pitchClass
    chord_pcs = {root_pc, third_pc, fifth_pc}
    return [p for p in pool if Pitch(p).pitchClass in chord_pcs]

# 2026-07-30 사용자 지정 "대원칙" -- 항상(옵션 아님) 적용:
# 두 음표(직전 음 기준) 간 계이름 간격은 8도(옥타브) 이하. 반음 수 기준 12로 매핑
# ("8도"는 옥타브 자체를 가리키는 명칭이라 반음 수가 고정됨 -- 모호함 없음).
MAX_MELODIC_INTERVAL = 12
# 8분음표 이하가 연속되는(2개·4개 묶음 등 촘촘한 구간) 경우엔 더 좁게 7도 이하.
# "7도"는 단7도(10반음)/장7도(11반음)로 갈릴 수 있어 완전히 명확하진 않지만, "옥타브
# 바로 아래까지"로 해석해 11로 잡음(사용자 확인 필요 시 조정).
MAX_DENSE_MELODIC_INTERVAL = 11


def _weighted_choice(candidates: list, prev_pitch: Optional[str]):
    """MARKOV_BIAS 확률로 candidates를 PDMX 실제곡 음정 전이 확률표로 가중 추첨하고,
    그 외(확률 미적중/prev_pitch 없음/테이블 없음)엔 기존처럼 균등 무작위(random.choice).
    가중치는 (직전 음 -> 후보 음)의 다이어토닉 스텝 차이(Pitch.diatonicNoteNum 차, 조성
    무관)를 테이블에서 찾아 씀 -- 테이블에 없는(범위 밖) 간격은 그 확률표의 최소 확률값을
    바닥으로 줘서 완전히 배제하지는 않는다(0 가중치가 섞이면 후보가 전부 0이 되는 극단
    상황을 방지)."""
    if prev_pitch is None or not MARKOV_TABLE or random.random() >= MARKOV_BIAS:
        return random.choice(candidates)
    prev_step = Pitch(prev_pitch).diatonicNoteNum
    floor = min(MARKOV_TABLE.values())
    weights = []
    for p in candidates:
        interval = Pitch(p).diatonicNoteNum - prev_step
        interval = max(-MARKOV_MAX_INTERVAL, min(MARKOV_MAX_INTERVAL, interval))
        weights.append(MARKOV_TABLE.get(interval, floor))
    return random.choices(candidates, weights=weights, k=1)[0]


def _pick_pitch(pool: list, diatonic_spellings: set, prev_pitch: Optional[str] = None,
                 preferred_pool: Optional[set] = None, max_interval: Optional[int] = None,
                 chord_pool: Optional[set] = None):
    """DIATONIC_BIAS 확률로 조표 음계 안의 음(철자까지 일치)만 우선 고르고, 나머지는
    자유롭게 고른다. 조표가 다양해질 때 "같은 음 토큰이 조표에 따라 임시표 유무가
    달라지는" 낯선 조합이 너무 자주 나오지 않도록 완화 -- 임시표 자체를 없애는 게
    아니라 빈도만 현실적으로 낮춤.
    MELODIC_BIAS > 0이고 prev_pitch가 있으면, 그 확률만큼 이전 음 근처(반음계 진행/
    순차 진행)로 후보를 좁혀서 실제 선율처럼 매끄럽게 이어지게 한다(나머지는 기존처럼
    도약 포함 자유 선택 -- 완전히 매끈하기만 한 것도 비현실적이라 확률로만 섞음).
    preferred_pool이 주어지고 PREFERRED_REGISTER_PROB > 0이면, 그 확률만큼 먼저 pool을
    preferred_pool과의 교집합으로 좁힌 뒤(교집합이 비면 원래 pool 유지) 위 두 단계를
    그 위에서 적용한다.
    chord_pool이 주어지고 CHORD_PROGRESSION_BIAS > 0이면, 그 확률만큼 pool을 현재 마디의
    암묵적 화성(진행표 기반, _next_degree/_chord_tones_for_degree 참고) 구성음과의
    교집합으로 좁힌다(교집합이 비면 원래 pool 유지) -- preferred_pool 다음, MELODIC_BIAS
    이전에 적용(2026-07-31, 사용자 요청: "실제 악보는 멜로디가 코드를 따라간다").
    max_interval이 주어지고 prev_pitch가 있으면, 이후 모든 단계보다 먼저(가장 우선순위
    높은 하드 제약으로) prev_pitch로부터 max_interval 반음 이내로 후보를 강제로 좁힌다
    (교집합이 비면 -- 극단적 pool 상황 -- 원래 pool 유지, 크래시 방지). 확률이 아니라
    항상 적용되는 하드 캡이라는 점이 preferred_pool/MELODIC_BIAS와 다름.
    최종 선택(균등 무작위였던 지점) 두 곳 다 MARKOV_BIAS가 켜져 있으면 _weighted_choice로
    대체 -- PDMX 실전 통계로 "그럴듯한 다음 음"을 우선하되 완전히 배제하진 않음."""
    base_pool = pool
    if max_interval is not None and prev_pitch is not None:
        prev_midi = Pitch(prev_pitch).midi
        capped = [p for p in base_pool if abs(Pitch(p).midi - prev_midi) <= max_interval]
        if capped:
            base_pool = capped

    if preferred_pool and PREFERRED_REGISTER_PROB > 0 and random.random() < PREFERRED_REGISTER_PROB:
        # 2026-07-31 버그 수정: 기존엔 base_pool(TREBLE/BASS_PITCHES, 예: C4~B5)과
        # preferred_pool(예: 치 D3~A3 U A5~A6)의 "교집합"을 취했는데, preferred_pool은
        # 애초에 base_pool 범위 *밖*(덧줄 많은 극단 음역)을 겨냥한 것이라 교집합이
        # 거의 항상 비어(D3~A3 쪽) 원래 pool로 폴백하거나 겹치는 일부(A5~B5)만 남아,
        # 실제로는 극단 음역이 거의 안 뽑혔음(실측: Step1 데이터 옥타브6=0.0%,
        # --preferred-register-prob 0.7로 올려도 5장 테스트에서 옥타브6/1 이하 0건).
        # preferred_pool을 그대로 후보로 써야 진짜 그 음역이 나온다.
        narrowed = list(preferred_pool)
        if narrowed:
            base_pool = narrowed

    if chord_pool and CHORD_PROGRESSION_BIAS > 0 and random.random() < CHORD_PROGRESSION_BIAS:
        narrowed = [p for p in base_pool if p in chord_pool]
        if narrowed:
            base_pool = narrowed

    near_pool = base_pool
    if prev_pitch is not None and MELODIC_BIAS > 0 and random.random() < MELODIC_BIAS:
        prev_midi = Pitch(prev_pitch).midi
        candidates = [p for p in base_pool if 0 < abs(Pitch(p).midi - prev_midi) <= MELODIC_MAX_STEP]
        if candidates:
            near_pool = candidates

    if diatonic_spellings and random.random() < DIATONIC_BIAS:
        candidates = [p for p in near_pool
                     if (Pitch(p).pitchClass, Pitch(p).step + (Pitch(p).accidental.modifier if Pitch(p).accidental else ''))
                        in diatonic_spellings]
        if candidates:
            return _weighted_choice(candidates, prev_pitch)
    return _weighted_choice(near_pool, prev_pitch)


_STEP_ORDER = 'CDEFGAB'


def _interval_degree(root: str, cand: str) -> int:
    """루트음과 후보음의 음이름(옥타브·임시표 무시) 사이 도수. 2도=2, 3도=3, 4도=4, ...
    옥타브를 넘으면(9도 등) mod 7로 단순 도수(옥타브 내 등가 도수)로 접는다."""
    r_step = Pitch(root).step
    c_step = Pitch(cand).step
    diff = (_STEP_ORDER.index(c_step) - _STEP_ORDER.index(r_step)) % 7
    return diff + 1


def _choose_chord_size() -> int:
    """화음 노트 개수(CHORD_MIN_NOTES~CHORD_MAX_NOTES) 결정. CHORD_SIZE_WEIGHTS가
    설정되면 그 가중치로 직접 선택(범위 밖 키는 무시), 아니면 CHORD_TWO_NOTE_PROB
    기반 기존 로직으로 폴백."""
    if CHORD_SIZE_WEIGHTS:
        sizes = [n for n in CHORD_SIZE_WEIGHTS if CHORD_MIN_NOTES <= n <= CHORD_MAX_NOTES]
        weights = [CHORD_SIZE_WEIGHTS[n] for n in sizes]
        if sizes and sum(weights) > 0:
            return random.choices(sizes, weights=weights)[0]
    if CHORD_MIN_NOTES < CHORD_MAX_NOTES and random.random() < CHORD_TWO_NOTE_PROB:
        return CHORD_MIN_NOTES
    return random.randint(min(CHORD_MIN_NOTES + 1, CHORD_MAX_NOTES), CHORD_MAX_NOTES)


def _weighted_sample_by_interval(root: str, candidates: list, k: int) -> list:
    """CHORD_INTERVAL_WEIGHTS에 따라 루트 기준 음정 도수로 가중 샘플(비복원).
    같은 MIDI(동일 음, 다른 철자 -- 예: D#2/Eb2)가 화음 안에 중복으로 뽑히지 않도록
    한 번 뽑힌 피치와 같은 MIDI를 가진 나머지 후보도 함께 제거(2026-07-31, 실제 렌더링
    에서 D#2+Eb2가 한 화음에 동시에 나오는 버그 발견 후 수정)."""
    pool = list(candidates)
    chosen = []
    for _ in range(k):
        if not pool:
            break
        weights = [CHORD_INTERVAL_WEIGHTS.get(_interval_degree(root, p), _CHORD_INTERVAL_WEIGHT_EPS)
                   for p in pool]
        if sum(weights) <= 0:
            weights = [1.0] * len(pool)
        pick = random.choices(pool, weights=weights)[0]
        chosen.append(pick)
        pick_midi = Pitch(pick).midi
        pool = [p for p in pool if Pitch(p).midi != pick_midi]
    return chosen


def _sample_unique_midi(candidates: list, k: int) -> list:
    """candidates에서 서로 다른 MIDI를 갖는 k개를 비복원 무작위 샘플(같은 음의 다른
    철자가 함께 뽑히는 것 방지, 2026-07-31)."""
    pool = list(candidates)
    random.shuffle(pool)
    chosen = []
    chosen_midis = set()
    for p in pool:
        if len(chosen) >= k:
            break
        m = Pitch(p).midi
        if m in chosen_midis:
            continue
        chosen.append(p)
        chosen_midis.add(m)
    return chosen


def _choose_dur(max_ql: float):
    possible = [(ql, tok, w) for ql, tok, w in DURATIONS if ql <= max_ql + 1e-9]
    if not possible:
        return 0.25, '1/16'
    if SHORT_NOTE_BIAS > 0:
        weights = [w * (1 + SHORT_NOTE_BIAS) if ql <= 0.5 + 1e-9 else w for ql, tok, w in possible]
    else:
        weights = [w for *_, w in possible]
    if LONG_NOTE_BIAS > 0:
        weights = [wt * (1 + LONG_NOTE_BIAS) if ql >= 1.0 - 1e-9 else wt
                   for (ql, _, _), wt in zip(possible, weights)]
    if EIGHTH_BIAS > 0:
        weights = [wt * (1 + EIGHTH_BIAS) if tok == '1/8' else wt
                   for (_, tok, _), wt in zip(possible, weights)]
    if SIXTEENTH_BIAS > 0:
        weights = [wt * (1 + SIXTEENTH_BIAS) if tok == '1/16' else wt
                   for (_, tok, _), wt in zip(possible, weights)]
    if DOTTED8_BIAS > 0:
        weights = [wt * (1 + DOTTED8_BIAS) if tok == '3/16' else wt
                   for (_, tok, _), wt in zip(possible, weights)]
    if RARE_LONG_BIAS > 0:
        weights = [wt * (1 + RARE_LONG_BIAS) if tok in ('1/1', '3/4') else wt
                   for (_, tok, _), wt in zip(possible, weights)]
    ql, tok, _ = random.choices(possible, weights=weights)[0]
    return round(min(ql, max_ql), 6), tok


def _maybe_add_courtesy_accidental(el):
    """COURTESY_ACCIDENTAL_PROB 확률로 el(Note 또는 Chord)의 각 구성음에 "굳이 붙이는"
    임시표를 시각적으로만 추가(자연음이면 내추럴 기호, 임시표가 이미 있으면 강제 표시).
    pitch.nameWithOctave는 그대로라 note-{pitch} 토큰에는 영향 없음."""
    if COURTESY_ACCIDENTAL_PROB <= 0:
        return
    pitches = el.pitches if hasattr(el, 'pitches') else [el.pitch]
    for p in pitches:
        if random.random() >= COURTESY_ACCIDENTAL_PROB:
            continue
        if p.accidental is None:
            p.accidental = Accidental('natural')
        p.accidental.displayStatus = True


def _add_artic(el, name: str):
    amap = {
        'staccato': articulations.Staccato(),
        'accent':   articulations.Accent(),
        'tenuto':   articulations.Tenuto(),
        'marcato':  articulations.StrongAccent(),
    }
    obj = amap.get(name)
    if obj is None:
        return
    targets = list(el.notes) if isinstance(el, Chord) else [el]
    for t in targets:
        t.articulations = [obj]


def _add_ornament(el, name: str):
    omap = {
        'trill':   expressions.Trill(),
        'mordent': expressions.Mordent(),
        'turn':    expressions.Turn(),
    }
    obj = omap.get(name)
    if obj is None:
        return
    try:
        el.expressions.append(obj)
    except Exception:
        el.expressions = [obj]


def _build_part(pitch_pool, clef_obj, clef_tok, ks_sharps, ks_name,
                ts_num, ts_den, n_measures, use_ottava=False, clef_events=None,
                preferred_pool=None, hide_timesig=False) -> tuple:
    """
    단일 파트(treble 또는 bass) 생성. 시스템(줄) 나누기는 여기서 결정하지 않는다
    -- 치/베이스 양쪽 내용을 다 봐야 정확한 밀도를 알 수 있어서, 줄바꿈 삽입은
    호출자(build_score_r3)가 양쪽 마디 객체를 받은 뒤 처리한다.

    clef_events (선택, CLEF_CHANGE_PROB 전용): 마디 중간 클렙 전환 이벤트 리스트.
      [{'measure_idx': int, 'offset_ql': float, 'clef_obj': Clef, 'clef_tok': str,
        'pitch_pool': list}, ...] -- (measure_idx, offset_ql) 오름차순으로 정렬돼 있어야
      함(호출자 책임). offset_ql은 해당 마디 시작 기준 오프셋. 각 이벤트 시점 이후로
      생성되는 음표는 이벤트의 pitch_pool/clef_tok을 따르고, 그 오프셋에 실제 music21
      Clef 객체가 마디에 삽입되어 렌더링에도 반영된다.

    TIE_PROB(전역) > 0이면: 마디 끝 음표를 다음 마디 첫 음표와 같은 피치로 강제해
    붙임줄(tie)로 잇는다. clef_events가 있는 파트는 상호작용 미검증이라 자동으로 끔.
    Returns (Part, list_of_measure_token_lists, list_of_barline_toks, list_of_measure_objs)
    """
    measure_ql = ts_num * (4.0 / ts_den)
    diatonic_pcs = _diatonic_pitch_classes(ks_sharps)

    # 2026-07-31 사용자 요청: 2/4 박자에 한해 16분음표4개 묶음 확률을 8분음표4개 묶음보다
    # 높게, 나머지 박자는 8분음표4개 비율을 기본값보다 조금 더 높게 -- 박자별로 갈리므로
    # 파트 시작 시점(이 파트의 ts_num/ts_den 고정)에 한 번만 유효값을 정한다.
    if ts_num == 2 and ts_den == 4:
        eighth_run_prob_eff = (EIGHTH_RUN_PROB_2_4 if EIGHTH_RUN_PROB_2_4 is not None
                                else EIGHTH_RUN_PROB)
        sixteenth_run_prob_eff = (SIXTEENTH_RUN_PROB_2_4 if SIXTEENTH_RUN_PROB_2_4 is not None
                                   else SIXTEENTH_RUN_PROB)
    else:
        eighth_run_prob_eff = EIGHTH_RUN_PROB
        sixteenth_run_prob_eff = SIXTEENTH_RUN_PROB

    part = Part()
    part.insert(0, clef_obj)
    part.insert(0, key.KeySignature(ks_sharps))
    ts_obj = meter.TimeSignature(f'{ts_num}/{ts_den}')
    if hide_timesig:
        ts_obj.style.hideObjectOnPrint = True
    part.insert(0, ts_obj)

    clef_events = sorted(clef_events or [], key=lambda e: (e['measure_idx'], e['offset_ql']))
    event_i = 0
    cur_pitch_pool = pitch_pool
    tie_enabled = TIE_PROB > 0 and not clef_events
    pending_tie = None   # None 또는 {'pitches': [str, ...]} (길이 1=단음, 2=화음)

    # 스팬 사전 결정
    use_hairpin = random.random() < HAIRPIN_PROB
    hp_type     = random.choice(['cresc', 'dim'])
    hp_start_m  = random.randint(0, max(0, n_measures - 2)) if use_hairpin else -1
    hp_end_m    = min(hp_start_m + random.randint(1, 2), n_measures - 1) if use_hairpin else -1

    use_ott    = use_ottava and random.random() < OTTAVA_PROB
    ott_type   = random.choice(['8va', '8vb'])
    ott_start_m = random.randint(0, max(0, n_measures - 2)) if use_ott else -1
    ott_end_m  = min(ott_start_m + 1, n_measures - 1) if use_ott else -1

    hp_notes  = []
    ott_notes = []

    measure_tok_lists = []
    barline_toks      = []
    measure_objs      = []
    prev_pitch        = None   # MELODIC_BIAS용 -- 마디 경계 넘어서도 선율선이 이어지게 유지
    prev_ql           = None   # 직전 음가(quarterLength) -- 촘촘한 연속 그룹(8분음표 이하가
                                # 연속) 판정용, 마디 경계 넘어서도 유지
    prev_degree       = 1      # CHORD_PROGRESSION_BIAS용 -- 마디마다 _next_degree()로 갱신,
                                # 첫 마디는 으뜸화음(1도)에서 시작

    for m_idx in range(n_measures):
        m     = Measure(number=m_idx + 1)
        m_tok = []
        measure_objs.append(m)

        if CHORD_PROGRESSION_BIAS > 0:
            prev_degree = _next_degree(prev_degree)
            measure_chord_pool = set(_chord_tones_for_degree(ks_sharps, prev_degree, cur_pitch_pool))
        else:
            measure_chord_pool = None

        if random.random() < DYNAMIC_PROB:
            dyn = random.choice(DYNAMICS_LIST)
            m.insert(0, dynamics.Dynamic(dyn))
            m_tok.append(f'dynamic-{dyn}')

        if use_hairpin and m_idx == hp_start_m:
            m_tok.append(f'hairpin-{hp_type}-start')

        if use_ott and m_idx == ott_start_m:
            m_tok.append(f'ottava-{ott_type}-start')

        use_slur  = random.random() < SLUR_PROB
        slur_ns   = []
        slur_open = False

        use_tuplet  = random.random() < TUPLET_PROB
        tuplet_done = False

        remaining = measure_ql
        # 이 마디가 붙임줄을 하나 "시작"할지 마디 시작 시점에 미리 결정 -- 실제 시작 지점은
        # 이 마디를 생성하는 도중 처음 마주치는 이어붙이기 후보(단음 또는 2음 화음)에서
        # 소비된다(그게 마디 마지막 음표면 다음 마디로 넘어가는 tie, 중간 음표면 이 마디
        # 안에서 끝나는 tie -- 2026-07-30 사용자 요청으로 둘 다 지원).
        measure_tie_pending = tie_enabled and random.random() < TIE_PROB

        while remaining > 1e-9:
            # 마디 중간 클렙 전환 이벤트 -- 현재 오프셋을 지난 이벤트가 있으면 여기서 처리
            # (다음 while-loop 반복까지 지연될 수 있음: 3연음 도중엔 끊지 않음). 목표 마디를
            # 온음표 등 한 이벤트로 통째로 건너뛴 경우(catch-up)엔 이 마디의 현재 오프셋에
            # 삽입 -- 이벤트가 영영 소실되지 않도록 함(measure_idx는 <=로 완화).
            offset_in_measure = measure_ql - remaining
            while (event_i < len(clef_events)
                   and clef_events[event_i]['measure_idx'] <= m_idx
                   and (clef_events[event_i]['measure_idx'] < m_idx
                        or clef_events[event_i]['offset_ql'] <= offset_in_measure + 1e-9)):
                ev = clef_events[event_i]
                insert_off = ev['offset_ql'] if ev['measure_idx'] == m_idx else offset_in_measure
                m.insert(insert_off, ev['clef_obj'])
                m_tok.append(ev['clef_tok'])
                cur_pitch_pool = ev['pitch_pool']
                event_i += 1

            # pending_tie(이전에 시작된 붙임줄의 이어붙이기 의무)가 있으면 바로 다음
            # 생성 요소를 무조건 그 연속으로 강제(같은 마디 안이든 다음 마디로 넘어가든
            # 동일 로직 -- "바로 다음"이 보장되도록 여기서 즉시 소비).
            forced_tie = None
            if tie_enabled and pending_tie is not None:
                forced_tie = pending_tie
                pending_tie = None

            if forced_tie is None and use_tuplet and not tuplet_done and remaining >= 1.0 - 1e-9:
                from music21 import duration as dur_mod
                m_tok.append('tuplet-3-start')
                # 이 3연음 전체가 오선 밖(최대 두 줄)까지 확장될지를 한 번만 결정(음마다
                # 따로 굴리면 세 음이 서로 다른 음역으로 튀어 부자연스러움, 2026-07-31
                # 사용자 요청 -- 기본 70%는 기존처럼 정상 음역 고정, 30%는 확장 허용).
                tuplet_pool = (_extend_pool_by_semitones(cur_pitch_pool, 7)
                               if random.random() < TUPLET_LEDGER_PROB else cur_pitch_pool)
                # 세 슬롯 중 하나(무작위)가 쉼표가 될 확률(2026-07-31 사용자 요청).
                rest_slot = random.randint(0, 2) if random.random() < TUPLET_REST_PROB else -1
                for slot in range(3):
                    if slot == rest_slot:
                        r_obj = Rest()
                        r_obj.duration.quarterLength = 1.0 / 3.0
                        r_obj.duration.appendTuplet(dur_mod.Tuplet(3, 2))
                        m.append(r_obj)
                        m_tok.append("rest-1/8")
                        prev_ql = 1.0 / 3.0
                        continue
                    # 3연음도 8분음표 길이 3개가 촘촘히 연속되는 그룹이라 좁은 간격 상한
                    # (7도) 적용 -- 사용자가 명시한 "2개/4개 묶음"의 자연스러운 확장.
                    # preferred_pool=None(2026-07-31, 사용자 요청) -- 셋잇단음표는
                    # PREFERRED_REGISTER_PROB의 먼 덧줄 음역은 타지 않고, tuplet_pool
                    # (정상 음역 또는 위에서 결정한 제한적 확장 음역) 안에서만 고름.
                    p = _pick_pitch(tuplet_pool, diatonic_pcs, prev_pitch, None,
                                     max_interval=MAX_DENSE_MELODIC_INTERVAL,
                                     chord_pool=measure_chord_pool)
                    n_obj = Note(p)
                    n_obj.duration.quarterLength = 1.0 / 3.0
                    n_obj.duration.appendTuplet(dur_mod.Tuplet(3, 2))
                    _maybe_add_courtesy_accidental(n_obj)
                    m.append(n_obj)
                    m_tok.append(f"note-{_np(n_obj.pitch.nameWithOctave)}")
                    m_tok.append("dur-1/8")
                    prev_pitch = p
                    prev_ql = 1.0 / 3.0
                    if hp_start_m <= m_idx <= hp_end_m:
                        hp_notes.append(n_obj)
                    if use_ott and ott_start_m <= m_idx <= ott_end_m:
                        ott_notes.append(n_obj)
                m_tok.append('tuplet-3-end')
                remaining -= 1.0
                tuplet_done = True
                continue

            if (forced_tie is None and remaining >= 2.0 - 1e-9
                    and eighth_run_prob_eff > 0 and random.random() < eighth_run_prob_eff):
                # 8분음표 4개로 정확히 두 박(ql=2.0) 채움 -- 매번 다시 확률을 굴리므로
                # 한 마디 안에서 여러 번 걸릴 수 있음(오선 검출 강화, 2026-07-31 사용자
                # 요청). preferred_pool=None(3연음과 동일 이유).
                for _ in range(4):
                    p = _pick_pitch(cur_pitch_pool, diatonic_pcs, prev_pitch, None,
                                     max_interval=MAX_DENSE_MELODIC_INTERVAL,
                                     chord_pool=measure_chord_pool)
                    n_obj = Note(p, quarterLength=0.5)
                    _maybe_add_courtesy_accidental(n_obj)
                    m.append(n_obj)
                    m_tok.append(f"note-{_np(n_obj.pitch.nameWithOctave)}")
                    m_tok.append("dur-1/8")
                    prev_pitch = p
                    prev_ql = 0.5
                    if hp_start_m <= m_idx <= hp_end_m:
                        hp_notes.append(n_obj)
                    if use_ott and ott_start_m <= m_idx <= ott_end_m:
                        ott_notes.append(n_obj)
                remaining -= 2.0
                continue

            if (forced_tie is None and remaining >= 1.0 - 1e-9
                    and sixteenth_run_prob_eff > 0 and random.random() < sixteenth_run_prob_eff):
                # 16분음표 4개로 정확히 한 박(ql=1.0) 채움 -- 매 박 경계마다 다시 확률을
                # 굴리므로 한 마디 안에 이런 묶음이 여럿 나올 수 있음(밀집 16분음표 구간의
                # 오선 검출 강화 목적, 2026-07-31 사용자 요청). preferred_pool=None(위와
                # 동일 이유).
                for _ in range(4):
                    p = _pick_pitch(cur_pitch_pool, diatonic_pcs, prev_pitch, None,
                                     max_interval=MAX_DENSE_MELODIC_INTERVAL,
                                     chord_pool=measure_chord_pool)
                    n_obj = Note(p, quarterLength=0.25)
                    _maybe_add_courtesy_accidental(n_obj)
                    m.append(n_obj)
                    m_tok.append(f"note-{_np(n_obj.pitch.nameWithOctave)}")
                    m_tok.append("dur-1/16")
                    prev_pitch = p
                    prev_ql = 0.25
                    if hp_start_m <= m_idx <= hp_end_m:
                        hp_notes.append(n_obj)
                    if use_ott and ott_start_m <= m_idx <= ott_end_m:
                        ott_notes.append(n_obj)
                remaining -= 1.0
                continue

            ql, dtok = _choose_dur(remaining)
            # 대원칙(2026-07-30): 두 음표 간 간격은 8도(옥타브=12반음) 이하. 직전 음도
            # 이번 음도 8분음표 이하(촘촘히 연속되는 그룹)면 7도(11반음)까지 더 좁힘.
            _dense_now = prev_ql is not None and prev_ql <= 0.5 + 1e-9 and ql <= 0.5 + 1e-9
            _interval_cap = MAX_DENSE_MELODIC_INTERVAL if _dense_now else MAX_MELODIC_INTERVAL
            tie_start_pitches = None   # 이번에 생성한 요소가 붙임줄 시작 후보면 그 피치(들)

            if forced_tie is not None:
                chosen = forced_tie['pitches']
                if len(chosen) >= 2:
                    el = Chord(chosen, quarterLength=ql)
                    el_toks = [f"note-{_np(chosen[0])}", f"dur-{dtok}"]
                    el_toks += [f"chord-{_np(p)}" for p in chosen[1:]]
                else:
                    el = Note(chosen[0], quarterLength=ql)
                    el_toks = [f"note-{_np(chosen[0])}", f"dur-{dtok}"]
                el.tie = tie_mod.Tie('stop')
                prev_pitch = chosen[0]
                # 붙임줄 도착점은 새 붙임줄의 시작점이 될 수 없음(체인 길이 항상 2) --
                # tie_start_pitches를 None으로 둬서 아래 트리거 블록이 건너뛰게 함.
            else:
                r_val = random.random()
                if r_val < REST_PROB:
                    m.append(Rest(quarterLength=ql))
                    m_tok.append(f'rest-{dtok}')
                    remaining -= ql
                    prev_ql = None   # 쉼표는 촘촘한 연속을 끊음
                    continue
                elif r_val < REST_PROB + CHORD_PROB:
                    n_notes = _choose_chord_size()
                    root      = _pick_pitch(cur_pitch_pool, diatonic_pcs, prev_pitch, preferred_pool,
                                            max_interval=_interval_cap, chord_pool=measure_chord_pool)
                    root_midi = Pitch(root).midi
                    # extra 후보도 prev_pitch 기준 간격 상한을 같이 적용 -- 안 그러면 root는
                    # 캡 안이어도 extra가 root보다 더 아래로 내려가 정렬 후 chosen[0](=음표
                    # 토큰의 대표 피치, 다음 음의 prev_pitch로도 쓰임)가 캡을 넘을 수 있음
                    # (2026-07-30 실측으로 발견 -- root는 정상인데 chosen[0]만 위반하는 사례).
                    prev_midi = Pitch(prev_pitch).midi if prev_pitch is not None else None
                    # p != root뿐 아니라 MIDI도 root와 달라야 함 -- 문자열만 다르고
                    # 실제로는 같은 음(동일 MIDI, 다른 철자, 예: D#2/Eb2)인 후보가
                    # 화음에 root와 나란히 뽑히는 걸 방지(2026-07-31 발견/수정).
                    candidates = [p for p in cur_pitch_pool
                                 if Pitch(p).midi != root_midi and abs(Pitch(p).midi - root_midi) <= CHORD_MAX_INTERVAL
                                 and (prev_midi is None or abs(Pitch(p).midi - prev_midi) <= _interval_cap)]
                    if CHORD_INTERVAL_WEIGHTS:
                        extra = _weighted_sample_by_interval(root, candidates, n_notes - 1)
                    else:
                        extra = _sample_unique_midi(candidates, n_notes - 1)
                    chosen = sorted([root] + extra, key=lambda pp: Pitch(pp).midi)
                    el = Chord(chosen, quarterLength=ql)
                    el_toks = [f"note-{_np(chosen[0])}", f"dur-{dtok}"]
                    el_toks += [f"chord-{_np(p)}" for p in chosen[1:]]
                    prev_pitch = chosen[0]
                    if len(chosen) == 2:   # 3음 화음은 이어붙이기 후보에서 제외
                        tie_start_pitches = chosen
                else:
                    p  = _pick_pitch(cur_pitch_pool, diatonic_pcs, prev_pitch, preferred_pool,
                                     max_interval=_interval_cap, chord_pool=measure_chord_pool)
                    el = Note(p, quarterLength=ql)
                    el_toks = [f"note-{_np(el.pitch.nameWithOctave)}", f"dur-{dtok}"]
                    prev_pitch = p
                    tie_start_pitches = [p]
            prev_ql = ql

            if use_slur and not slur_open and remaining > ql + 1e-9:
                m_tok.append('slur-start')
                slur_open = True

            m_tok.extend(el_toks)
            _maybe_add_courtesy_accidental(el)
            m.append(el)

            if random.random() < ARTIC_PROB:
                artic = random.choice(ARTICS)
                _add_artic(el, artic)
                m_tok.append(f'artic-{artic}')

            if random.random() < ORNAMENT_PROB:
                orn = random.choice(ORNAMENTS)
                _add_ornament(el, orn)
                m_tok.append(f'ornament-{orn}')

            if random.random() < FERMATA_PROB and remaining <= ql + 1e-9:
                try:
                    el.expressions.append(expressions.Fermata())
                except Exception:
                    el.expressions = [expressions.Fermata()]
                m_tok.append('fermata')

            if hp_start_m <= m_idx <= hp_end_m:
                hp_notes.append(el)
            if use_ott and ott_start_m <= m_idx <= ott_end_m:
                ott_notes.append(el)
            if slur_open:
                slur_ns.append(el)

            remaining -= ql

            # 이 마디가 붙임줄을 시작하기로 했고(measure_tie_pending) 아직 소비 안 됐으면,
            # 방금 생성한 요소가 시작 후보(tie_start_pitches)일 때 여기서 실제로 시작한다.
            # 이게 마디 마지막 음표면 자연히 다음 마디로 넘어가는 tie가 되고, 아니면 이
            # 마디 안에서 끝나는 tie가 됨(둘 다 pending_tie를 통해 "바로 다음 생성 요소"에
            # 동일하게 강제되므로 별도 분기 불필요). 마지막 마디의 마지막 슬롯이면(이어붙일
            # 자리가 아예 없음) 시작 안 함. note+dur(+artic/ornament/fermata) 토큰 바로 뒤,
            # 트레일러(slur-end 등)보다 앞에 'tie'가 붙도록 여기서 바로 append.
            is_last_slot = (m_idx == n_measures - 1) and (remaining <= 1e-9)
            if (tie_enabled and measure_tie_pending and pending_tie is None
                    and tie_start_pitches is not None and not is_last_slot):
                take = True
                if len(tie_start_pitches) == 2:
                    take = random.random() < CHORD_TIE_SUBPROB
                if take:
                    el.tie = tie_mod.Tie('start')
                    m_tok.append('tie')
                    pending_tie = {'pitches': tie_start_pitches}
                    measure_tie_pending = False

        if slur_open and len(slur_ns) >= 2:
            m_tok.append('slur-end')
            try:
                sl = spanner.Slur()
                sl.addSpannedElements([slur_ns[0], slur_ns[-1]])
                part.insert(0, sl)
            except Exception:
                pass

        if use_hairpin and m_idx == hp_end_m:
            m_tok.append(f'hairpin-{hp_type}-end')

        if use_ott and m_idx == ott_end_m:
            m_tok.append(f'ottava-{ott_type}-end')

        if m_idx < n_measures - 1:
            barline_toks.append('barline')
        else:
            m.rightBarline = bar.Barline('final')
            barline_toks.append('barline-final')

        measure_tok_lists.append(m_tok)
        part.append(m)

    # 스패너 연결
    if len(hp_notes) >= 2:
        try:
            hp_obj = dynamics.Crescendo() if hp_type == 'cresc' else dynamics.Diminuendo()
            hp_obj.addSpannedElements([hp_notes[0], hp_notes[-1]])
            part.insert(0, hp_obj)
        except Exception:
            pass

    if len(ott_notes) >= 2:
        try:
            ott_obj = spanner.Ottava(type=ott_type)
            ott_obj.addSpannedElements([ott_notes[0], ott_notes[-1]])
            part.insert(0, ott_obj)
        except Exception:
            pass

    return part, measure_tok_lists, barline_toks, measure_objs


# ─────────────────────────────────────────────────────────────────────────────
#  왼손 반주(브로큰코드) / 오른손 멜로디 구조 (2026-07-31 사용자 요청)
#  exactPicture의 뉴에이지 스타일 실사곡(newage01~03) 참고 -- 베이스 성부가 무작위
#  음이 아니라 그 마디의 화음(근음/3도/5도)을 아르페지오로 순환하는 패턴이 흔함.
#  _build_part()보다 훨씬 단순한 전용 생성기 -- 실제 반주가 멜로디보다 규칙적인 것을
#  그대로 반영(다이나믹/셋잇단음표/붙임줄 등 표현 요소는 생략, 브로큰코드 패턴 자체에
#  집중).
# ─────────────────────────────────────────────────────────────────────────────

ACCOMPANIMENT_PROB = 0.0  # 0=끔(기존 동작). >0이면 대보표 곡에서 이 확률로 베이스 성부
                          # 전체가 무작위 생성 대신 브로큰코드 반주 패턴으로 생성됨.

_ACCOMPANIMENT_PATTERNS = [
    ['root', 'fifth', 'third', 'fifth'],
    ['root', 'third', 'fifth', 'third'],
    ['root', 'fifth', 'root', 'fifth'],
]
_QL_TO_DUR_TOK = {ql: tok for ql, tok, _ in DURATIONS}


def _chord_role_pitches(ks_sharps: int, degree: int, pool: list) -> dict:
    """조표(ks_sharps)의 장음계에서 degree(1~7)를 근음으로 하는 다이어토닉 3화음의
    근음/3도/5도 각각에 해당하는 pool 내 음이름 목록을 role별로 반환."""
    scale = key.KeySignature(ks_sharps).getScale('major').pitches
    role_pc = {
        'root':  scale[(degree - 1) % 7].pitchClass,
        'third': scale[(degree + 1) % 7].pitchClass,
        'fifth': scale[(degree + 3) % 7].pitchClass,
    }
    return {role: [p for p in pool if Pitch(p).pitchClass == pc] for role, pc in role_pc.items()}


def _build_accompaniment_part(pitch_pool, clef_obj, clef_tok, ks_sharps, ts_num, ts_den,
                               n_measures, hide_timesig=False) -> tuple:
    """브로큰코드(아르페지오) 반주 패턴으로 한 파트를 생성. 마디마다 화성 진행
    (_next_degree, CHORD_PROGRESSION_BIAS와 동일 로직)을 따라가며 근음-5도-3도 등
    순환 패턴을 8분음표 단위로 채우고, 마지막 한 박(remaining<=1.0)은 하나의 긴
    음표로 마무리(관찰된 실제 곡 패턴의 근사 -- 매번 정확히 4분음표로 끝나는 단순화가
    있음, 필요시 추후 다양화 가능)."""
    measure_ql = ts_num * (4.0 / ts_den)
    part = Part()
    part.insert(0, clef_obj)
    part.insert(0, key.KeySignature(ks_sharps))
    ts_obj = meter.TimeSignature(f'{ts_num}/{ts_den}')
    if hide_timesig:
        ts_obj.style.hideObjectOnPrint = True
    part.insert(0, ts_obj)

    measure_tok_lists, barline_toks, measure_objs = [], [], []
    prev_pitch = None
    prev_degree = 1
    pattern = random.choice(_ACCOMPANIMENT_PATTERNS)

    for m_idx in range(n_measures):
        m = Measure(number=m_idx + 1)
        m_tok = []
        measure_objs.append(m)

        prev_degree = _next_degree(prev_degree)
        roles = _chord_role_pitches(ks_sharps, prev_degree, pitch_pool)

        remaining = measure_ql
        slot = 0
        while remaining > 1e-9:
            ql = remaining if remaining <= 1.0 + 1e-9 else 0.5
            dtok = _QL_TO_DUR_TOK.get(round(ql, 6), '1/8')
            if random.random() < 0.05:
                m.append(Rest(quarterLength=ql))
                m_tok.append(f'rest-{dtok}')
                remaining -= ql
                slot += 1
                continue
            role = pattern[slot % len(pattern)]
            candidates = roles.get(role) or pitch_pool
            if prev_pitch is not None:
                prev_midi = Pitch(prev_pitch).midi
                near = [p for p in candidates if abs(Pitch(p).midi - prev_midi) <= MAX_MELODIC_INTERVAL]
                if near:
                    candidates = near
            p = random.choice(candidates)
            n_obj = Note(p, quarterLength=ql)
            _maybe_add_courtesy_accidental(n_obj)
            m.append(n_obj)
            m_tok.append(f"note-{_np(p)}")
            m_tok.append(f"dur-{dtok}")
            prev_pitch = p
            remaining -= ql
            slot += 1

        if m_idx < n_measures - 1:
            barline_toks.append('barline')
        else:
            m.rightBarline = bar.Barline('final')
            barline_toks.append('barline-final')

        measure_tok_lists.append(m_tok)
        part.append(m)

    return part, measure_tok_lists, barline_toks, measure_objs


def build_score_r3(score_id: int, force_c_major: bool = False, natural_only: bool = False) -> tuple:
    """
    대보표 생성.

    토큰 구조 (마디별):
      <SOS> clef-G key time
      [treble_m0_toks] staff-bass clef-F [bass_m0_toks] barline
      [treble_m1_toks] staff-bass [bass_m1_toks] barline
      ...
      [treble_mN_toks] staff-bass [bass_mN_toks] barline-final
      <EOS>

    오선 감지 순서: treble(0), bass(1), treble(2), bass(3) ...
    → inference.py가 짝수=treble, 홀수=bass로 처리

    force_c_major=True: 조표를 C장조(0 sharps)로 고정 (커리큘럼 중간 단계용 --
    오선 개수 축만 격리해서 학습시키고 싶을 때).
    natural_only=True: 임시표(#, b) 없는 자연음 피치만 사용 + 조표도 C장조로 강제
    (build_score_single_staff와 동일한 이유 -- 조표가 C장조가 아니면 자연음을 내려면
    임시표 취소용 natural 기호가 오히려 필요해져서 "임시표 없음" 의도와 모순됨).
    """
    ts_num, ts_den = random.choices(TIME_SIGS, weights=TS_WEIGHTS)[0]
    ks_sharps, ks_name = ((0, 'C') if (force_c_major or natural_only)
                          else random.choices(KEY_SIGS, weights=KS_WEIGHTS)[0])
    n_measures = random.randint(MIN_MEASURES, MAX_MEASURES)
    hide_ts = random.random() < HIDE_TIMESIG_PROB   # 치/베이스 동시 적용(같은 시스템이므로)

    # 반복기호 (treble에만 적용, bass는 같은 barline 구조를 따름)
    use_repeat  = random.random() < REPEAT_PROB and n_measures >= 2
    rp_start_m  = random.randint(0, n_measures // 2) if use_repeat else -1
    rp_end_m    = random.randint(max(rp_start_m, 1), n_measures - 1) if use_repeat else -1

    treble_pool = TREBLE_PITCHES_NATURAL if natural_only else TREBLE_PITCHES
    bass_pool   = BASS_PITCHES_NATURAL if natural_only else BASS_PITCHES
    if CROSS_REGISTER_PROB > 0 and random.random() < CROSS_REGISTER_PROB:
        # range.mscz 기준 상/하한(치=C3~B6, 베이스=E1~B4) 안으로 clamp된 전용 풀 사용
        # -- 정상 풀을 그대로 맞바꾸면(예전 방식) 치가 C2까지, 베이스가 B5까지 내려가서/
        # 올라가서 사용자가 지정한 범위를 벗어났었음.
        treble_low_pool  = TREBLE_LOW_PITCHES_NATURAL if natural_only else TREBLE_LOW_PITCHES
        bass_high_pool   = BASS_HIGH_PITCHES_NATURAL if natural_only else BASS_HIGH_PITCHES
        mode = random.choice(['swap', 'both_high', 'both_low'])
        if mode == 'swap':
            treble_pool, bass_pool = treble_low_pool, bass_high_pool
        elif mode == 'both_high':
            bass_pool = bass_high_pool
        else:  # both_low
            treble_pool = treble_low_pool

    treble_preferred = TREBLE_PREFERRED_NATURAL if natural_only else TREBLE_PREFERRED_PITCHES
    bass_preferred    = BASS_PREFERRED_NATURAL if natural_only else BASS_PREFERRED_PITCHES

    # 같은-clef 대보표(SAME_CLEF_PROB) -- 트리거되면 위/아래 보표가 둘 다 같은 clef로
    # 나옴. 피치 풀은 CROSS_REGISTER_PROB용 풀을 재사용해 같은 clef 안에서도 위/아래가
    # 음역으로 구분되게 함(둘 다 정확히 같은 음역이면 시각적으로 위/아래 구분이 안 됨).
    same_clef_mode = None  # None | 'both_treble' | 'both_bass'
    if SAME_CLEF_PROB > 0 and random.random() < SAME_CLEF_PROB:
        same_clef_mode = random.choice(['both_treble', 'both_bass'])
    if same_clef_mode == 'both_treble':
        top_clef_obj, top_clef_tok = clef.TrebleClef(), 'clef-G'
        bot_clef_obj, bot_clef_tok = clef.TrebleClef(), 'clef-G'
        top_pool = treble_pool
        bot_pool = TREBLE_LOW_PITCHES_NATURAL if natural_only else TREBLE_LOW_PITCHES
    elif same_clef_mode == 'both_bass':
        top_clef_obj, top_clef_tok = clef.BassClef(), 'clef-F'
        bot_clef_obj, bot_clef_tok = clef.BassClef(), 'clef-F'
        top_pool = BASS_HIGH_PITCHES_NATURAL if natural_only else BASS_HIGH_PITCHES
        bot_pool = bass_pool
    else:
        top_clef_obj, top_clef_tok = clef.TrebleClef(), 'clef-G'
        bot_clef_obj, bot_clef_tok = clef.BassClef(), 'clef-F'
        top_pool, bot_pool = treble_pool, bass_pool

    # 마디 중간 클렙 전환(CLEF_CHANGE_PROB) -- 대보표에도 적용(2026-07-30 추가, 기존엔
    # build_score_single_staff에만 있었음). 치/베이스 "둘 중 하나"에만(양쪽 동시 전환은
    # 안 함) -- 어느 쪽이 될지는 50/50, _build_part()는 원래 clef_events를 받는
    # 범용 함수라 대보표 쪽 호출에 그대로 넘기기만 하면 동일하게 동작한다. 클렙 전환이
    # 걸린 파트는 tie_enabled가 자동으로 꺼짐(_build_part의 기존 규칙, 단일오선과 동일).
    treble_clef_events, bass_clef_events = [], []
    measure_ql = ts_num * (4.0 / ts_den)
    if CLEF_CHANGE_PROB > 0 and same_clef_mode is None and n_measures >= 2 and random.random() < CLEF_CHANGE_PROB:
        change_m = random.randint(0, n_measures - 2)
        revert_m = random.randint(change_m + 1, n_measures - 1)
        change_off = random.uniform(0.25, max(0.25, measure_ql - 0.25))
        revert_off = (random.uniform(0.25, max(0.25, measure_ql - 0.25))
                      if revert_m > change_m else measure_ql)
        if random.random() < 0.5:
            treble_clef_events = [
                {'measure_idx': change_m, 'offset_ql': change_off,
                 'clef_obj': clef.BassClef(), 'clef_tok': 'clef-F', 'pitch_pool': bass_pool},
                {'measure_idx': revert_m, 'offset_ql': revert_off,
                 'clef_obj': clef.TrebleClef(), 'clef_tok': 'clef-G', 'pitch_pool': treble_pool},
            ]
        else:
            bass_clef_events = [
                {'measure_idx': change_m, 'offset_ql': change_off,
                 'clef_obj': clef.TrebleClef(), 'clef_tok': 'clef-G', 'pitch_pool': treble_pool},
                {'measure_idx': revert_m, 'offset_ql': revert_off,
                 'clef_obj': clef.BassClef(), 'clef_tok': 'clef-F', 'pitch_pool': bass_pool},
            ]

    treble_part, t_measure_toks, t_barline_toks, t_measure_objs = _build_part(
        top_pool,
        top_clef_obj, top_clef_tok,
        ks_sharps, ks_name,
        ts_num, ts_den, n_measures,
        use_ottava=True,
        clef_events=treble_clef_events,
        preferred_pool=treble_preferred,
        hide_timesig=hide_ts,
    )
    # 왼손 반주(브로큰코드)/오른손 멜로디 구조(2026-07-31 사용자 요청, exactPicture
    # newage01~03 참고) -- ACCOMPANIMENT_PROB 확률로 베이스 성부를 무작위 생성 대신
    # 전용 아르페지오 반주 생성기로 대체. 클렙 중간 전환(bass_clef_events)은 반주
    # 패턴과 상호작용을 검증하지 않아 이 모드에서는 적용하지 않음(자동으로 일반
    # 생성 경로로 폴백). same_clef_mode도 동일한 이유로 제외(_build_accompaniment_part는
    # clef.BassClef()/bass_pool을 하드코딩해서 top_clef_obj/top_pool을 반영 못 함).
    use_accompaniment = (ACCOMPANIMENT_PROB > 0 and not bass_clef_events and same_clef_mode is None
                          and random.random() < ACCOMPANIMENT_PROB)
    if use_accompaniment:
        bass_part, b_measure_toks, _, b_measure_objs = _build_accompaniment_part(
            bass_pool, clef.BassClef(), 'clef-F', ks_sharps, ts_num, ts_den, n_measures,
            hide_timesig=hide_ts,
        )
    else:
        bass_part, b_measure_toks, _, b_measure_objs = _build_part(
            bot_pool,
            bot_clef_obj, bot_clef_tok,
            ks_sharps, ks_name,
            ts_num, ts_den, n_measures,
            use_ottava=False,
            clef_events=bass_clef_events,
            preferred_pool=bass_preferred,
            hide_timesig=hide_ts,
        )

    # 시스템(줄) 나누기: 밀도 기반(DENSITY_BREAK) 또는 고정 마디수 기반(기존 방식) 중 선택.
    # 치/베이스 양쪽 마디 객체를 다 받은 뒤에야 정확히 결정 가능.
    if DENSITY_BREAK:
        system_breaks = _decide_system_breaks(t_measure_toks, b_measure_toks, MAX_SYSTEM_WEIGHT)
    else:
        eff_mps = _effective_measures_per_system(n_measures)
        system_breaks = ([i for i in range(1, n_measures) if i % eff_mps == 0]
                         if eff_mps else [])
    for i in system_breaks:
        t_measure_objs[i].insert(0, layout.SystemLayout(isNew=True))
        b_measure_objs[i].insert(0, layout.SystemLayout(isNew=True))

    # 토큰 조합 -- 시작 clef/최초 bass clef 토큰은 same_clef_mode에 따라 top_clef_tok/
    # bot_clef_tok을 씀(기존엔 'clef-G'/'clef-F' 하드코딩이었는데, 같은-clef 모드에서
    # 실제로는 둘 다 clef-F(또는 둘 다 clef-G)여야 하므로 고정값을 쓰면 안 됨).
    tokens = ['<SOS>', top_clef_tok, f'key-{ks_name}', f'time-{ts_num}/{ts_den}']

    bass_clef_emitted = False
    for m_idx in range(n_measures):
        # 반복기호 시작 (treble 마디 시작 앞)
        if use_repeat and m_idx == rp_start_m:
            tokens.append('barline-start-repeat')

        # Treble 마디 내용
        tokens.extend(t_measure_toks[m_idx])

        # staff-bass 구분자 + bass clef (최초 1회)
        tokens.append('staff-bass')
        if not bass_clef_emitted:
            tokens.append(bot_clef_tok)
            bass_clef_emitted = True

        # Bass 마디 내용
        tokens.extend(b_measure_toks[m_idx])

        # 마디선
        if use_repeat and m_idx == rp_end_m:
            tokens.append('barline-end-repeat')
        else:
            tokens.append(t_barline_toks[m_idx])

    tokens.append('<EOS>')

    score = Score()
    # music21 기본 메타데이터("Music21 Fragment" 제목 + 우상단 "Music21" 워터마크)를
    # 그대로 두면 MuseScore 렌더링에 그대로 찍혀나온다 -- build_score_single_staff에는
    # 이미 있던 blanking 처리가 대보표 경로에는 빠져 있었음(2026-07-28, page_noise_and_redetect
    # 350px 크롭이 우상단 워터맠을 캔버스에 끌어들이는 걸 보고 발견).
    score.metadata = metadata.Metadata()
    score.metadata.title = ' '
    score.metadata.composer = ' '
    score.insert(0, treble_part)
    score.insert(0, bass_part)
    return score, tokens, system_breaks


def build_score_single_staff(score_id: int, natural_only: bool = False,
                             atom_only: bool = False) -> tuple:
    """
    단일 오선(treble 또는 bass 둘 중 하나만) 생성 — 커리큘럼 초기 단계용.
    조표는 항상 C장조(vocab엔 natural 전용 토큰이 없어 임시표는 음이름(#/b)에
    내재됨 -- key signature 축이 아니라 pitch pool 축으로 난이도를 조절한다).

    atom_only=True: 4/4 한 마디를 통째로 채우는 음표/쉼표 딱 1개만 생성
    (호출 전 DURATIONS를 온음(1/1)만 남도록 override해서 사용, 항상 1마디 고정).
    natural_only=True: 임시표(#, b) 없는 자연음 피치만 사용.
    atom_only이 아니면 MIN_MEASURES~MAX_MEASURES 범위에서 실제 촬영 시 오선 1개 프레임에
    여러 마디가 걸리는 경우(가이드 카메라 "오선 1개" 모드)를 반영해 여러 마디로 생성한다
    (기존엔 이 경로가 항상 1마디로 고정돼있어 실사용 조건과 안 맞았음, 2026-07-28 확인).
    """
    ts_num, ts_den = (4, 4) if atom_only else random.choices(TIME_SIGS, weights=TS_WEIGHTS)[0]
    ks_sharps, ks_name = 0, 'C'
    n_measures = 1 if atom_only else random.randint(MIN_MEASURES, MAX_MEASURES)
    hide_ts = (not atom_only) and random.random() < HIDE_TIMESIG_PROB

    is_treble = random.random() < 0.5
    if is_treble:
        pitch_pool, clef_cls, clef_tok = (TREBLE_PITCHES_NATURAL if natural_only else TREBLE_PITCHES), clef.TrebleClef, 'clef-G'
        other_pool, other_clef_cls, other_clef_tok = (BASS_PITCHES_NATURAL if natural_only else BASS_PITCHES), clef.BassClef, 'clef-F'
    else:
        pitch_pool, clef_cls, clef_tok = (BASS_PITCHES_NATURAL if natural_only else BASS_PITCHES), clef.BassClef, 'clef-F'
        other_pool, other_clef_cls, other_clef_tok = (TREBLE_PITCHES_NATURAL if natural_only else TREBLE_PITCHES), clef.TrebleClef, 'clef-G'
    clef_obj = clef_cls()

    # 마디 중간 클렙 전환(CLEF_CHANGE_PROB) — 반대쪽 클렙으로 전환했다가 이후 마디에서
    # 되돌아오는 두 이벤트 쌍을 생성. atom_only(1마디 고정)에는 적용하지 않음(되돌아올
    # 마디 자체가 없음).
    clef_events = []
    measure_ql = ts_num * (4.0 / ts_den)
    if (not atom_only and CLEF_CHANGE_PROB > 0 and n_measures >= 2
            and random.random() < CLEF_CHANGE_PROB):
        change_m = random.randint(0, n_measures - 2)
        revert_m = random.randint(change_m + 1, n_measures - 1)
        # offset 0은 원래 클렙과 구분이 안 되므로(마디 시작=파트 기본 클렙과 시각적으로 동일)
        # 살짝 뒤로 미뤄 "마디 중간"이라는 걸 분명히 함.
        change_off = random.uniform(0.25, max(0.25, measure_ql - 0.25))
        revert_off = (random.uniform(0.25, max(0.25, measure_ql - 0.25))
                      if revert_m > change_m else measure_ql)
        clef_events.append({'measure_idx': change_m, 'offset_ql': change_off,
                             'clef_obj': other_clef_cls(), 'clef_tok': other_clef_tok,
                             'pitch_pool': other_pool})
        clef_events.append({'measure_idx': revert_m, 'offset_ql': revert_off,
                             'clef_obj': clef_cls(), 'clef_tok': clef_tok,
                             'pitch_pool': pitch_pool})

    own_preferred = ((TREBLE_PREFERRED_NATURAL if natural_only else TREBLE_PREFERRED_PITCHES) if is_treble
                     else (BASS_PREFERRED_NATURAL if natural_only else BASS_PREFERRED_PITCHES))
    part, measure_tok_lists, barline_toks, _ = _build_part(
        pitch_pool, clef_obj, clef_tok, ks_sharps, ks_name,
        ts_num, ts_den, n_measures=n_measures, use_ottava=False,
        clef_events=clef_events, preferred_pool=own_preferred,
        hide_timesig=hide_ts,
    )
    # 마디 수/밀도(SHORT_NOTE_BIAS)가 오르면 MuseScore가 기본 페이지 폭 기준으로 자동
    # 2줄로 줄바꿈해버림 -- 라벨은 한 줄짜리 평평한 토큰 시퀀스라 줄바꿈되면 라벨-이미지가
    # 어긋남(2026-07-28 실측: 5마디+밀도 올린 상태에서 15장 중 4장 줄바꿈 확인). 페이지 폭을
    # 넉넉히 넓게 지정해서 몇 마디든 항상 한 줄에 들어가도록 강제.
    part.insert(0, layout.PageLayout(pageWidth=12000, pageHeight=1000))

    tokens = ['<SOS>', clef_tok, f'key-{ks_name}', f'time-{ts_num}/{ts_den}']
    for m_tok, bar_tok in zip(measure_tok_lists, barline_toks):
        tokens.extend(m_tok)
        tokens.append(bar_tok)
    tokens.append('<EOS>')

    score = Score()
    # 기본 타이틀("Music21 Fragment")이 와이드 페이지(wide_page.mss)에서는 중앙 정렬돼
    # 오선에서 멀리 떨어진 곳에 찍힘 -- preprocess()의 콘텐츠 바운딩박스가 오선~타이틀
    # 사이 빈 공간까지 다 포함해버려서 리사이즈 후 오선이 몇 픽셀로 짜부라져 오선 검출이
    # 전부 실패하는 원인이었음(2026-07-28, 학습 로그에서 단일 오선 4800개 전부 skip로
    # 확인). 타이틀 자체를 비워서 콘텐츠 바운딩박스가 오선에만 딱 맞도록 함.
    score.metadata = metadata.Metadata()
    score.metadata.title = ' '
    score.metadata.composer = ' '
    score.insert(0, part)
    return score, tokens


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Round 3 대보표 데이터 생성')
    p.add_argument('--count',     type=int, default=4000)
    p.add_argument('--output',    default='train/Round3')
    p.add_argument('--seed',      type=int, default=None)
    p.add_argument('--musescore', default=None)
    p.add_argument('--no-png',    action='store_true')
    p.add_argument('--start-idx', type=int, default=1)
    p.add_argument('--difficulty', choices=['easy', 'medium', 'full'], default='full',
                    help='기호 등장 확률 프로파일 (커리큘럼 학습용). easy=음표/쉼표 위주, '
                         'medium=절반 밀도, full=기존 기본값 (기본값)')
    p.add_argument('--min-measures', type=int, default=2)
    p.add_argument('--max-measures', type=int, default=4)
    p.add_argument('--single-staff', action='store_true',
                    help='대보표 대신 단일 오선(treble/bass 중 하나)만 생성 (커리큘럼 초기 단계용)')
    p.add_argument('--natural-only', action='store_true',
                    help='임시표(#, b) 없는 자연음 피치만 사용 (커리큘럼 초기 단계용). '
                         '단일 오선/대보표 둘 다 적용됨 -- 대보표는 조표도 C장조로 함께 강제됨 '
                         '(임시표 취소용 natural 기호가 재도입되는 모순 방지)')
    p.add_argument('--atom-only', action='store_true',
                    help='4/4 한 마디를 통째로 채우는 음표/쉼표 1개짜리 초단문만 생성 '
                         '(--single-staff와 함께 사용)')
    p.add_argument('--force-c-major', action='store_true',
                    help='대보표(--single-staff 미사용) 생성 시 조표를 C장조로 고정 '
                         '(커리큘럼 중간 단계용 -- 오선 개수 축만 격리)')
    p.add_argument('--quarter-only', action='store_true',
                    help='박자를 4/4로 고정하고 음표/쉼표를 4분음표 길이로만 채워 '
                         '오선 하나당 정확히 4개의 note/rest만 생성 (대보표 진단용 -- '
                         'duration 종류를 최소화해 오선 개수 축만 격리)')
    p.add_argument('--duration-subset', default=None,
                    help='허용할 duration 종류를 콤마로 지정 (예: "1/4,1/2,1/1,1/8,1/16"). '
                         '지정 시 DURATIONS를 해당 종류만으로 필터링(원래 가중치 비율 유지), '
                         '박자는 4/4로 고정, 마디는 1개로 고정 (대보표 duration 커리큘럼 단계용 -- '
                         '--quarter-only의 일반화 버전)')
    p.add_argument('--no-rests', action='store_true',
                    help='쉼표를 전혀 생성하지 않고 음표로만 채움 (REST_PROB=0). '
                         'duration 판별 자체(예: 1/8, 1/16)를 note-vs-rest 혼동 없이 '
                         '순수하게 학습시키고 싶을 때 사용')
    p.add_argument('--narrow-pitch', action='store_true',
                    help='피치 범위를 오선 중간선 아래로 좁혀서(treble C4-A4, bass E2-C3) '
                         '기둥(stem) 방향을 항상 위쪽으로 고정 -- 음높이별 깃발(flag) 방향/'
                         '모양 차이를 배제하고 duration 판별 자체만 순수하게 테스트하고 싶을 때 사용')
    p.add_argument('--time-sig', default=None,
                    help='--duration-subset이 고정하는 4/4 대신 사용할 박자 (예: "2/4"). '
                         'duration이 짧을수록(1/8, 1/16...) 4/4 한 마디에 더 많은 음표가 '
                         '들어차 음표 간격이 좁아지는 밀도 문제를 격리 테스트하고 싶을 때, '
                         '마디당 음표 개수를 4a(4/4+4분음표=마디당 4개)와 맞추는 용도로 사용')
    p.add_argument('--measures-per-system', type=int, default=None,
                    help='지정 시 매 N마디마다 강제 줄바꿈(SystemLayout isNew=True) 삽입 '
                         '-- 여러 시스템(줄)로 나뉜 대보표 페이지를 생성. MuseScore의 자동 '
                         '줄바꿈은 마디를 균등하게 안 나누므로(예: 8마디->6+2), dataset.py의 '
                         '시스템별 균등분할 가정이 실제로 맞도록 항상 정확히 N마디씩 나눔')
    p.add_argument('--auto-measures-per-system', action='store_true',
                    help='--min/--max-measures를 범위(예: 1~4)로 줄 때, 샘플마다 실제 뽑힌 '
                         'n_measures에 맞춰 시스템당 마디 수를 자동 결정(3마디 이하는 한 '
                         '시스템, 4마디부터 절반씩 나눠 줄바꿈) -- 홀수 마디가 섞여도 균등분할 '
                         '가정이 항상 성립하도록 보장. --measures-per-system과 동시 사용 시 '
                         '이 옵션이 우선')
    p.add_argument('--repeat-prob', type=float, default=None,
                    help='REPEAT_PROB(기본 난이도별 0.10~0.12)를 덮어씀. barline-start-repeat/'
                         'barline-end-repeat 토큰 노출을 인위적으로 높여 격리 진단하고 싶을 때 '
                         '사용 (예: 1.0이면 2마디 이상인 샘플은 항상 반복기호 포함)')
    p.add_argument('--hide-timesig-prob', type=float, default=None,
                   help='HIDE_TIMESIG_PROB 덮어씀 -- 이 확률로 박자표 기호를 렌더링에서만 숨김(라벨은 유지)')
    p.add_argument('--chord-prob',    type=float, default=None, help='CHORD_PROB 덮어씀')
    p.add_argument('--chord-min-notes', type=int, default=None, help='화음 최소 음 수(기본 2)')
    p.add_argument('--chord-max-notes', type=int, default=None, help='화음 최대 음 수(기본 3)')
    p.add_argument('--chord-2note-prob', type=float, default=None,
                    help='CHORD_TWO_NOTE_PROB(기본 0.5=균등) 덮어씀 -- 화음을 2개 음으로 '
                         '고정할 확률, 올릴수록 2음 화음 비중 증가. CHORD_SIZE_WEIGHTS가 '
                         '지정되면 이 옵션은 무시됨')
    p.add_argument('--chord-size-weights', type=str, default=None,
                    help='CHORD_SIZE_WEIGHTS 덮어씀 -- "2:60,3:30,4:10" 형식으로 화음 '
                         '노트 개수(2/3/4개)별 상대 가중치 지정(CHORD_MIN/MAX_NOTES 범위로 '
                         '클램프). 지정 시 --chord-2note-prob보다 우선 적용')
    p.add_argument('--chord-interval-weights', type=str, default=None,
                    help='CHORD_INTERVAL_WEIGHTS 덮어씀 -- "2:60,3:30,4:10" 형식으로 '
                         '루트 기준 화음 후보음의 음정 도수(2도/3도/4도)별 상대 가중치 지정. '
                         '기본(미지정)은 도수 무관 균등 샘플(기존 동작)')
    p.add_argument('--chord-progression-bias', type=float, default=None,
                    help='CHORD_PROGRESSION_BIAS(기본 0=끔) 덮어씀 -- 마디마다 정해지는 '
                         '암묵적 화성 진행의 구성음 쪽으로 이 확률만큼 음표 후보를 좁힘')
    p.add_argument('--accompaniment-prob', type=float, default=None,
                    help='ACCOMPANIMENT_PROB(기본 0=끔) 덮어씀 -- 대보표 곡에서 이 확률로 '
                         '베이스 성부 전체가 브로큰코드(아르페지오) 반주 패턴으로 생성됨 '
                         '(왼손 반주/오른손 멜로디 구조, exactPicture newage 참고)')
    p.add_argument('--chord-max-interval', type=int, default=None,
                    help='화음 내 최저-최고음 간격 상한(반음 수, 기본 12=1옥타브/8도). '
                         '화음 음정이 비현실적으로 넓게 벌어지는 것을 방지')
    p.add_argument('--diatonic-bias', type=float, default=None,
                    help='DIATONIC_BIAS(기본 0.75) 덮어씀 -- 조표 음계 안의 음을 우선 고를 '
                         '확률. 0이면 완전 무작위(조표 도입 전 기존 동작), 1이면 항상 조표 '
                         '음계 안의 음만. 조표가 다양해질 때 "같은 음 토큰이 조표에 따라 '
                         '임시표 유무가 달라지는" 낯선 조합이 너무 잦아지는 것을 완화')
    p.add_argument('--density-break', action='store_true',
                    help='마디 개수 대신 실제 내용 밀도(음표/쉼표/화음 이벤트 수) 기준으로 '
                         '시스템(줄) 나누기를 결정 -- 실제 조판자처럼 내용이 빽빽하면 한 줄에 '
                         '마디를 적게, 단순하면 많이 담음. 마디 수가 짝수로 안 나눠떨어져도 '
                         '안전(각 시스템 마디 수가 달라질 수 있음). system_breaks가 JSON에 '
                         '함께 저장되어 dataset.py가 실제 줄바꿈 지점을 정확히 알 수 있음')
    p.add_argument('--max-system-weight', type=float, default=None,
                    help='MAX_SYSTEM_WEIGHT(기본 8.0) 덮어씀 -- --density-break일 때 시스템 '
                         '하나에 담을 수 있는 밀도 상한(음표/쉼표=1, 화음 추가음=0.5)')
    p.add_argument('--dynamics-subset', default=None,
                    help='허용할 다이나믹 종류를 콤마로 지정 (예: "f,p"). 지정 시 DYNAMICS_LIST를 '
                         '해당 종류만으로 필터링 -- 다이나믹을 한꺼번에 7종 도입하는 대신 '
                         'f/p부터 최소로 시작하는 커리큘럼 단계용 (--duration-subset과 동일 패턴)')
    p.add_argument('--dynamic-prob',  type=float, default=None, help='DYNAMIC_PROB 덮어씀')
    p.add_argument('--hairpin-prob',  type=float, default=None, help='HAIRPIN_PROB 덮어씀')
    p.add_argument('--artic-prob',    type=float, default=None, help='ARTIC_PROB 덮어씀')
    p.add_argument('--ornament-prob', type=float, default=None, help='ORNAMENT_PROB 덮어씀')
    p.add_argument('--fermata-prob',  type=float, default=None, help='FERMATA_PROB 덮어씀')
    p.add_argument('--slur-prob',     type=float, default=None, help='SLUR_PROB 덮어씀')
    p.add_argument('--tuplet-prob',   type=float, default=None, help='TUPLET_PROB 덮어씀')
    p.add_argument('--tuplet-ledger-prob', type=float, default=None,
                    help='TUPLET_LEDGER_PROB(기본 0=끔) 덮어씀 -- 3연음 전체가 이 확률로 '
                         '오선 밖(최대 두 줄) 확장 음역에서 뽑힘, 나머지는 정상 음역 고정')
    p.add_argument('--tuplet-rest-prob', type=float, default=None,
                    help='TUPLET_REST_PROB(기본 0=끔) 덮어씀 -- 3연음 세 슬롯 중 하나가 '
                         '이 확률로 쉼표(rest-1/8)가 됨')
    p.add_argument('--ottava-prob',   type=float, default=None, help='OTTAVA_PROB 덮어씀')
    p.add_argument('--melodic-bias', type=float, default=None,
                    help='MELODIC_BIAS(기본 0=끔) 덮어씀 -- 이 확률만큼 다음 음을 이전 음 '
                         '근처(--melodic-max-step 반음 이내)에서 고름. 기존에는 매 음을 '
                         '이전 음과 무관하게 독립 추첨해서 실제 곡 특유의 반음계 진행/순차 '
                         '진행이 전혀 안 나왔음 -- 실사 쇼팽 곡 GT와 가까운 분포를 만들고 '
                         '싶을 때 0.5~0.8 정도로 사용')
    p.add_argument('--melodic-max-step', type=int, default=None,
                    help='MELODIC_MAX_STEP(기본 4반음)덮어씀 -- --melodic-bias 적용 시 '
                         '"가까운 음"으로 취급할 반음 상한')
    p.add_argument('--markov-bias', type=float, default=None,
                    help='MARKOV_BIAS(기본 0=끔) 덮어씀 -- 이 확률만큼 다음 음을 --markov-table의 '
                         'PDMX 실제곡 음정 전이 통계로 가중 추첨. --markov-table을 같이 줘야 '
                         '실제로 켜짐(테이블 없으면 이 값과 무관하게 기존 균등 무작위)')
    p.add_argument('--markov-table', type=str, default=None,
                    help='build_markov_transitions.py가 만든 markov_transitions.json 경로 -- '
                         '지정 안 하면 --markov-bias가 있어도 적용 안 됨')
    p.add_argument('--short-note-bias', type=float, default=None,
                    help='SHORT_NOTE_BIAS(기본 0=끔) 덮어씀 -- 1/8 이하 짧은 음 가중치를 '
                         '(1+값)배로 올려 마디당 음표 밀도를 높임(예: 1.0 = 2배). 실제 악보처럼 '
                         '마디 폭이 들쭉날쭉(밀집/희소 마디 혼재)한 데이터 비율을 늘리고 싶을 '
                         '때 사용')
    p.add_argument('--long-note-bias', type=float, default=None,
                    help='LONG_NOTE_BIAS(기본 0=끔) 덮어씀 -- 1/4 이상 긴 음(1/4,1/2,1/1) '
                         '가중치를 (1+값)배로 올려 마디당 음표 개수를 줄임(SHORT_NOTE_BIAS '
                         '반대 방향)')
    p.add_argument('--eighth-bias', type=float, default=None,
                    help='EIGHTH_BIAS(기본 0=끔) 덮어씀 -- 8분음표(dur-1/8, 기본 가중치 15퍼센트) '
                         '가중치만 (1+값)배로 올림. SHORT_NOTE_BIAS와 달리 1/16은 안 건드림')
    p.add_argument('--sixteenth-bias', type=float, default=None,
                    help='SIXTEENTH_BIAS(기본 0=끔) 덮어씀 -- 16분음표(dur-1/16, 기본 가중치 '
                         '8퍼센트) 가중치만 (1+값)배로 올림. EIGHTH_BIAS와 개별 조정용')
    p.add_argument('--dotted8-bias', type=float, default=None,
                    help='DOTTED8_BIAS(기본 0=끔) 덮어씀 -- 점8분음표(dur-3/16, 기본 가중치 '
                         '0.5 퍼센트) 가중치를 (1+값)배로 올림(예: 10.0 = 11배 -> 약 5.5 퍼센트). '
                         '"점8분+16분음표" 리듬 셀 노출을 늘리는 커리큘럼 단계용')
    p.add_argument('--rare-long-bias', type=float, default=None,
                    help='RARE_LONG_BIAS(기본 0=끔) 덮어씀 -- 온음표(1/1, 기본 4퍼센트)와 '
                         '점2분음표(3/4, 기본 2퍼센트)만 (1+값)배로 올림. LONG_NOTE_BIAS와 달리 '
                         '4분/8분음표는 안 건드림')
    p.add_argument('--eighth-run-prob', type=float, default=None,
                    help='EIGHTH_RUN_PROB(기본 0=끔) 덮어씀 -- 박 경계에서 이 확률로 '
                         '8분음표 4개(2박) 강제 배치, 한 마디에 여러 번 걸릴 수 있음')
    p.add_argument('--sixteenth-run-prob', type=float, default=None,
                    help='SIXTEENTH_RUN_PROB(기본 0=끔) 덮어씀 -- 박 경계에서 이 확률로 '
                         '16분음표 4개(1박) 강제 배치, 한 마디에 여러 번 걸릴 수 있음')
    p.add_argument('--eighth-run-prob-2-4', type=float, default=None,
                    help='EIGHTH_RUN_PROB_2_4(기본 None=EIGHTH_RUN_PROB와 동일) 덮어씀 -- '
                         '2/4 박자 파트에서만 적용되는 오버라이드')
    p.add_argument('--sixteenth-run-prob-2-4', type=float, default=None,
                    help='SIXTEENTH_RUN_PROB_2_4(기본 None=SIXTEENTH_RUN_PROB와 동일) 덮어씀 '
                         '-- 2/4 박자 파트에서만 적용되는 오버라이드')
    p.add_argument('--courtesy-accidental-prob', type=float, default=None,
                    help='COURTESY_ACCIDENTAL_PROB(기본 0=끔) 덮어씀 -- 음표(화음 구성음 '
                         '포함)마다 이 확률로 굳이 임시표/내추럴 기호를 시각적으로 강제 '
                         '표시(토큰/라벨에는 영향 없는 순수 시각 증강)')
    p.add_argument('--cross-register-prob', type=float, default=None,
                    help='CROSS_REGISTER_PROB(기본 0=끔) 덮어씀 -- 이 확률로 대보표 치/베이스 '
                         '피치 풀을 비정상 조합(swap/both_high/both_low)으로 바꿔 덧줄이 많은 '
                         '"교차 음역" 케이스를 생성. 클렙 표기 자체는 항상 정상(치 위/베이스 '
                         '아래)이고 그 안의 실제 피치 풀만 바뀜')
    p.add_argument('--same-clef-prob', type=float, default=None,
                    help='SAME_CLEF_PROB(기본 0=끔) 덮어씀 -- 이 확률로 대보표 시스템 전체가 '
                         '같은 clef(둘 다 clef-G 또는 둘 다 clef-F)로 나옴(--cross-register-prob와 '
                         '달리 clef 기호 자체가 바뀜). --clef-change-prob(마디 중간 일시 전환)과 '
                         '동시 적용 시 이 옵션이 뽑히면 --clef-change-prob 쪽은 그 샘플에서 자동으로 꺼짐')
    p.add_argument('--clef-change-prob', type=float, default=None,
                    help='CLEF_CHANGE_PROB(기본 0=끔) 덮어씀 -- 이 확률로 한 오선 안에서 마디 '
                         '중간에 반대쪽 클렙으로 전환했다가 이후 마디에서 되돌아오는 표기를 '
                         '생성. --single-staff와 대보표 둘 다 지원(대보표는 치/베이스 중 '
                         '하나에만, 50/50). 새 vocab 토큰 불필요 -- 기존 clef-G/clef-F 재사용')
    p.add_argument('--tie-prob', type=float, default=None,
                    help='TIE_PROB(기본 0=끔) 덮어씀 -- 이 확률로 마디 끝 음표와 다음 마디 첫 '
                         '음표를 같은 피치로 강제하고 붙임줄(tie)로 이어줌. 새 vocab 토큰 `tie` '
                         '1개 필요(tokenizer258.json에 이미 추가돼 있어야 함). 마디 중간 클렙 '
                         '전환(--clef-change-prob)과 동시 사용 시 해당 파트는 tie가 자동으로 꺼짐')
    p.add_argument('--preferred-register-prob', type=float, default=None,
                    help='PREFERRED_REGISTER_PROB(기본 0=끔) 덮어씀 -- 이 확률로 음 후보를 '
                         '"덧줄이 많이 필요한" 선호 구간(치 D3~A3/A5~A6, 베이스 C2~E2/F4~B4, '
                         'range.mscz 참고)과의 교집합으로 좁혀서 뽑음(교집합이 비면 원래 풀 '
                         '유지 -- cross-register 극단 풀과 함께 써도 안전)')
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    globals().update(DIFFICULTY_PROFILES[args.difficulty])
    global MIN_MEASURES, MAX_MEASURES, DURATIONS, CHORD_PROB, TIME_SIGS, TS_WEIGHTS, REST_PROB, MEASURES_PER_SYSTEM
    global REPEAT_PROB, DYNAMIC_PROB, HAIRPIN_PROB, ARTIC_PROB, ORNAMENT_PROB, FERMATA_PROB, SLUR_PROB, TUPLET_PROB, OTTAVA_PROB
    global TUPLET_LEDGER_PROB, TUPLET_REST_PROB
    global AUTO_MEASURES_PER_SYSTEM, CHORD_MIN_NOTES, CHORD_MAX_NOTES, CHORD_MAX_INTERVAL, CHORD_TWO_NOTE_PROB, CHORD_SIZE_WEIGHTS, CHORD_INTERVAL_WEIGHTS, DIATONIC_BIAS
    global DENSITY_BREAK, MAX_SYSTEM_WEIGHT, DYNAMICS_LIST, MELODIC_BIAS, MELODIC_MAX_STEP, SHORT_NOTE_BIAS, LONG_NOTE_BIAS, EIGHTH_BIAS, SIXTEENTH_BIAS, DOTTED8_BIAS, RARE_LONG_BIAS
    global EIGHTH_RUN_PROB, SIXTEENTH_RUN_PROB, EIGHTH_RUN_PROB_2_4, SIXTEENTH_RUN_PROB_2_4, COURTESY_ACCIDENTAL_PROB
    global ACCOMPANIMENT_PROB
    global MARKOV_BIAS, MARKOV_TABLE, MARKOV_MAX_INTERVAL
    global CROSS_REGISTER_PROB, CLEF_CHANGE_PROB, TIE_PROB, PREFERRED_REGISTER_PROB, CHORD_PROGRESSION_BIAS, SAME_CLEF_PROB
    global HIDE_TIMESIG_PROB
    MIN_MEASURES, MAX_MEASURES = args.min_measures, args.max_measures
    MEASURES_PER_SYSTEM = args.measures_per_system
    AUTO_MEASURES_PER_SYSTEM = args.auto_measures_per_system
    DENSITY_BREAK = args.density_break
    if args.max_system_weight is not None: MAX_SYSTEM_WEIGHT = args.max_system_weight
    if args.hide_timesig_prob is not None: HIDE_TIMESIG_PROB = args.hide_timesig_prob
    if args.repeat_prob    is not None: REPEAT_PROB    = args.repeat_prob
    if args.chord_prob     is not None: CHORD_PROB     = args.chord_prob
    if args.chord_min_notes    is not None: CHORD_MIN_NOTES    = args.chord_min_notes
    if args.chord_max_notes    is not None: CHORD_MAX_NOTES    = args.chord_max_notes
    if args.chord_max_interval is not None: CHORD_MAX_INTERVAL = args.chord_max_interval
    if args.chord_2note_prob is not None: CHORD_TWO_NOTE_PROB = args.chord_2note_prob
    if args.chord_size_weights is not None:
        CHORD_SIZE_WEIGHTS = {}
        for pair in args.chord_size_weights.split(','):
            n, wt = pair.split(':')
            CHORD_SIZE_WEIGHTS[int(n)] = float(wt)
    if args.chord_interval_weights is not None:
        CHORD_INTERVAL_WEIGHTS = {}
        for pair in args.chord_interval_weights.split(','):
            deg, wt = pair.split(':')
            CHORD_INTERVAL_WEIGHTS[int(deg)] = float(wt)
    if args.chord_progression_bias is not None: CHORD_PROGRESSION_BIAS = args.chord_progression_bias
    if args.diatonic_bias  is not None: DIATONIC_BIAS  = args.diatonic_bias
    if args.melodic_bias    is not None: MELODIC_BIAS    = args.melodic_bias
    if args.melodic_max_step is not None: MELODIC_MAX_STEP = args.melodic_max_step
    if args.markov_bias    is not None: MARKOV_BIAS = args.markov_bias
    if args.markov_table   is not None:
        with open(args.markov_table, encoding='utf-8') as _mf:
            _markov_json = json.load(_mf)
        MARKOV_TABLE = {int(k): v for k, v in _markov_json['table'].items()}
        MARKOV_MAX_INTERVAL = _markov_json.get('max_interval', MARKOV_MAX_INTERVAL)
    if args.short_note_bias is not None: SHORT_NOTE_BIAS = args.short_note_bias
    if args.long_note_bias is not None: LONG_NOTE_BIAS = args.long_note_bias
    if args.eighth_bias     is not None: EIGHTH_BIAS     = args.eighth_bias
    if args.sixteenth_bias  is not None: SIXTEENTH_BIAS  = args.sixteenth_bias
    if args.dotted8_bias    is not None: DOTTED8_BIAS    = args.dotted8_bias
    if args.rare_long_bias  is not None: RARE_LONG_BIAS  = args.rare_long_bias
    if args.eighth_run_prob is not None: EIGHTH_RUN_PROB = args.eighth_run_prob
    if args.sixteenth_run_prob is not None: SIXTEENTH_RUN_PROB = args.sixteenth_run_prob
    if args.eighth_run_prob_2_4 is not None: EIGHTH_RUN_PROB_2_4 = args.eighth_run_prob_2_4
    if args.sixteenth_run_prob_2_4 is not None: SIXTEENTH_RUN_PROB_2_4 = args.sixteenth_run_prob_2_4
    if args.courtesy_accidental_prob is not None: COURTESY_ACCIDENTAL_PROB = args.courtesy_accidental_prob
    if args.accompaniment_prob is not None: ACCOMPANIMENT_PROB = args.accompaniment_prob
    if args.cross_register_prob is not None: CROSS_REGISTER_PROB = args.cross_register_prob
    if args.clef_change_prob is not None: CLEF_CHANGE_PROB = args.clef_change_prob
    if args.same_clef_prob is not None: SAME_CLEF_PROB = args.same_clef_prob
    if args.tie_prob is not None: TIE_PROB = args.tie_prob
    if args.preferred_register_prob is not None: PREFERRED_REGISTER_PROB = args.preferred_register_prob
    if args.dynamic_prob   is not None: DYNAMIC_PROB   = args.dynamic_prob
    if args.hairpin_prob   is not None: HAIRPIN_PROB   = args.hairpin_prob
    if args.artic_prob     is not None: ARTIC_PROB     = args.artic_prob
    if args.ornament_prob  is not None: ORNAMENT_PROB  = args.ornament_prob
    if args.fermata_prob   is not None: FERMATA_PROB   = args.fermata_prob
    if args.slur_prob      is not None: SLUR_PROB      = args.slur_prob
    if args.tuplet_prob    is not None: TUPLET_PROB    = args.tuplet_prob
    if args.tuplet_ledger_prob is not None: TUPLET_LEDGER_PROB = args.tuplet_ledger_prob
    if args.tuplet_rest_prob is not None: TUPLET_REST_PROB = args.tuplet_rest_prob
    if args.ottava_prob    is not None: OTTAVA_PROB    = args.ottava_prob
    if args.dynamics_subset:
        allowed = set(args.dynamics_subset.split(','))
        unknown = allowed - set(DYNAMICS_LIST)
        if unknown:
            raise SystemExit(f"--dynamics-subset에 알 수 없는 다이나믹: {sorted(unknown)} "
                              f"(허용 가능: {DYNAMICS_LIST})")
        DYNAMICS_LIST = [d for d in DYNAMICS_LIST if d in allowed]
    if args.atom_only:
        DURATIONS  = [(4.0, '1/1', 1.0)]
        CHORD_PROB = 0.0
        MIN_MEASURES = MAX_MEASURES = 1
    if args.quarter_only:
        DURATIONS  = [(1.0, '1/4', 1.0)]
        CHORD_PROB = 0.0
        TIME_SIGS  = [(4, 4)]
        TS_WEIGHTS = [1.0]
        MIN_MEASURES = MAX_MEASURES = 1
    if args.duration_subset:
        allowed = set(args.duration_subset.split(','))
        known   = {tok for _, tok, _ in DURATIONS}
        unknown = allowed - known
        if unknown:
            raise SystemExit(f"--duration-subset에 알 수 없는 duration: {sorted(unknown)} "
                              f"(허용 가능: {sorted(known)})")
        filtered = [(ql, tok, w) for ql, tok, w in DURATIONS if tok in allowed]
        DURATIONS  = filtered
        CHORD_PROB = 0.0
        TIME_SIGS  = [(4, 4)]
        TS_WEIGHTS = [1.0]
        MIN_MEASURES = MAX_MEASURES = 1
    if args.time_sig:
        ts_num_s, ts_den_s = args.time_sig.split('/')
        TIME_SIGS  = [(int(ts_num_s), int(ts_den_s))]
        TS_WEIGHTS = [1.0]
    if args.no_rests:
        REST_PROB = 0.0
    if args.narrow_pitch:
        global TREBLE_PITCHES, BASS_PITCHES, TREBLE_PITCHES_NATURAL, BASS_PITCHES_NATURAL
        TREBLE_PITCHES = _pitch_pool("C4", "A4")
        BASS_PITCHES   = _pitch_pool("E2", "C3")
        TREBLE_PITCHES_NATURAL = _naturals_only(TREBLE_PITCHES)
        BASS_PITCHES_NATURAL   = _naturals_only(BASS_PITCHES)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    musescore_path = None
    if not args.no_png:
        musescore_path = find_musescore(args.musescore)
        if musescore_path:
            print(f"MuseScore: {musescore_path}")
        else:
            print("WARNING: MuseScore not found — XML+JSON만 생성")

    print(f"Round 3 생성: {args.count}개 → {out_dir.resolve()}")
    if args.single_staff:
        print(f"단일 오선: MEASURES {MIN_MEASURES}~{MAX_MEASURES}, difficulty={args.difficulty}, "
              f"natural_only={args.natural_only}, atom_only={args.atom_only}")
    else:
        print(f"대보표: treble + staff-bass + bass, MEASURES {args.min_measures}~{args.max_measures}, "
              f"difficulty={args.difficulty}, natural_only={args.natural_only}")

    try:
        from tqdm import tqdm as _tqdm
        _iter = lambda x: _tqdm(x, desc="Round3", unit="score")
    except ImportError:
        _iter = lambda x: x

    # 2026-07-30: 1단계(악보 구성+XML/라벨 저장)와 2단계(MuseScore 렌더링)를 분리 --
    # 렌더링을 이미지 1장마다 하지 않고 전부 모았다가 render_batch_png()로 한꺼번에
    # 처리(-j 배치 모드)해서 이미지마다 앱을 재시작하는 오버헤드를 없앤다(실측 약 1.9배
    # 단축). wide_page=True는 기존과 동일하게 항상 적용 -- 대보표(단일 오선 아님)도 예전엔
    # 기본 페이지 폭 기준으로 MuseScore가 조용히 2번째 시스템으로 줄바꿈하는 경우가 있었음
    # (2026-07-30 실측: --density-break 없이도 --short-note-bias 등으로 내용이 조금만
    # 조밀해지면 발생, system_breaks는 여전히 []로 기록돼 라벨-이미지가 어긋남). 실제 카메라
    # 캡처가 항상 시스템 1개만 담으므로(guided_camera_screen.dart) 학습 이미지도 항상 한
    # 시스템이어야 함.
    ok_xml = 0
    jobs = []  # (stem, xml_path, png_path) -- 렌더링 대상
    for i in _iter(range(args.start_idx, args.start_idx + args.count)):
        stem     = f"num{i}"
        xml_path = out_dir / f"{stem}.musicxml"
        png_path = out_dir / f"{stem}.png"
        lbl_path = out_dir / f"{stem}.json"

        system_breaks: list = []
        try:
            if args.single_staff:
                score, tokens = build_score_single_staff(
                    i, natural_only=args.natural_only, atom_only=args.atom_only)
            else:
                score, tokens, system_breaks = build_score_r3(
                    i, force_c_major=args.force_c_major, natural_only=args.natural_only)
        except Exception as exc:
            print(f"  [ERROR] {stem}: {exc}")
            continue

        try:
            score.write("musicxml", fp=str(xml_path))
            ok_xml += 1
        except Exception as exc:
            print(f"  [ERROR] {stem} XML: {exc}")
            continue

        lbl_path.write_text(
            json.dumps({"id": stem, "tokens": tokens, "system_breaks": system_breaks}, ensure_ascii=False),
            encoding='utf-8'
        )

        if musescore_path:
            jobs.append((stem, xml_path, png_path))

    ok_png = 0
    if musescore_path and jobs:
        print(f"\n렌더링 중 ({len(jobs)}장, 배치 모드)...")
        ok_stems = render_batch_png(musescore_path, jobs, wide_page=True,
                                     grand=not args.single_staff, n_measures=MAX_MEASURES)
        ok_png = len(ok_stems)

    print(f"\nRound 3 완료: XML={ok_xml}/{args.count}, PNG={ok_png}/{ok_xml}")
    print(f"출력: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
