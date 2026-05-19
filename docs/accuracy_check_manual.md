# OMR 라운드별 학습 & 정확도 체크 매뉴얼

각 라운드의 전체 흐름: **데이터 생성 → 학습 → 정확도 체크**  
Round 1 기준으로 상세히 설명하며, Round 2·3·4는 경로와 `--round` 번호만 바꾸면 동일합니다.

---

## 사전 준비 (최초 1회)

### 패키지 설치

```bash
# 데이터 생성용
pip install -r omr/data_gen/requirements.txt
# music21>=9.1, numpy>=1.26, Pillow>=10.0

# 학습 및 정확도 체크용
pip install -r omr/training/requirements.txt
# torch>=2.2.0, torchvision>=0.17.0, opencv-python>=4.9, numpy>=1.26, tqdm
```

### MuseScore 설치 확인 (PNG 렌더링에 필요)

```
Windows : C:\Program Files\MuseScore 4\bin\MuseScore4.exe
Linux   : sudo apt install musescore4
```

---

## 전체 디렉토리 구조

```
musicscore_flutter/
├── data/
│   ├── tokenizer.json            ← 공통 tokenizer (모든 라운드 공유, 수정 금지)
│   ├── round_tokens/
│   │   ├── round1_tokens.json    ← Round 1 허용 토큰 정의
│   │   ├── round2_tokens.json
│   │   ├── round3_tokens.json
│   │   └── round4_tokens.json
│   ├── Round1/                   ← 생성 후 학습 데이터가 저장되는 곳
│   │   ├── num1.png
│   │   ├── num1.json             ← {"tokens": ["clef-G", "key-C", ...]}
│   │   ├── num1.musicxml
│   │   └── test/                 ← 정확도 체크용 테스트 셋 (학습에 미사용)
│   ├── Round2/
│   ├── Round3/
│   └── Round4/
├── models/
│   ├── round1/
│   │   ├── segnet_best.pt
│   │   └── seq2seq_best.pt
│   ├── round2/
│   ├── round3/
│   └── round4/
├── reports/                      ← 정확도 CSV 리포트 저장
├── scripts/
│   ├── train_round.py
│   └── evaluate_round_accuracy.py
└── omr/
    ├── data_gen/generate_dataset.py
    └── training/train.py
```

---

## Round 1 전체 파이프라인

### Step 1. 데이터 생성

```bash
python data/generate_random_scores.py \
    --round  1 \
    --count  3000 \
    --output data/Round1
```

- `--round 1` : dynamics·hairpin·articulation 등 R2+ 심볼 확률을 0으로 설정해 순수 기본 표기만 생성
- `--count 3000` : 학습용 3000장 (권장 최소 2000장, 이상적 5000장)
- 생성 결과: `data/Round1/num{1..3000}.{png, json, musicxml}`

테스트 셋은 별도로 생성해 `data/Round1/test/`에 배치합니다:

```bash
python data/generate_random_scores.py \
    --round  1 \
    --count  200 \
    --output data/Round1/test
```

> **주의**: 테스트 셋은 학습 데이터(`data/Round1/`)와 seed를 다르게 사용해야 합니다.  
> `--seed 42`(학습) / `--seed 999`(테스트) 처럼 구분하세요.

### Step 2. 학습 (3-Phase 자동 실행)

```bash
python scripts/train_round.py --round 1
```

내부적으로 아래 세 단계를 순서대로 실행합니다:

```
Phase 1 (SegNet)         : 픽셀 세그멘테이션 학습  → models/round1/segnet_best.pt
Phase 2 (Encoder+Decoder): 토큰 시퀀스 학습        → models/round1/seq2seq_best.pt
Phase 3 (End-to-End)     : 전체 파인튜닝            → models/round1/seq2seq_best.pt (갱신)
```

GPU 메모리 등 옵션 조정이 필요하면:

```bash
python scripts/train_round.py --round 1 \
    --epochs-p1 30 --epochs-p2 80 --epochs-p3 20 \
    --batch-p1 16 --batch-p2 8 --batch-p3 4 \
    --device cuda
```

### Step 3. 정확도 체크

```bash
python scripts/evaluate_round_accuracy.py \
    --round    1 \
    --weights  models/round1/seq2seq_best.pt \
    --test-dir data/Round1/test \
    --tokenizer data/tokenizer.json \
    --report   reports/round1_eval.csv
```

#### 출력 예시

```
======================================================================
  OMR Round-1 Accuracy Report
======================================================================
  Weights    : models/round1/seq2seq_best.pt
  Test dir   : data/Round1/test
  Samples    : 200 found  /  200 evaluated
  Threshold  : 90.0%

  RESULT     : PASS
  Overall    : 93.5%  (TER 6.5%)
  Note-only  : 95.2%
  Per-sample : 186/200 pass threshold
======================================================================
```

#### PASS 기준

| 라운드 | Overall Acc 목표 |
|--------|----------------|
| Round 1 | ≥ 90% |
| Round 2 | ≥ 88% |
| Round 3 | ≥ 85% |
| Round 4 | ≥ 82% |

---

## Round 2·3·4 — 동일 패턴, 경로만 변경

### Round 2

```bash
# 1. 데이터 생성 (R2 심볼 포함: 임시표·코드·셈여림·슬러·잇단음표 등)
python data/generate_random_scores.py --round 2 --count 3000 --output data/Round2
python data/generate_random_scores.py --round 2 --count 200  --output data/Round2/test --seed 999

# 2. 학습 (Round 1 가중치 이어받기)
python scripts/train_round.py --round 2 \
    --prev-segnet  models/round1/segnet_best.pt \
    --prev-seq2seq models/round1/seq2seq_best.pt

# 3. 정확도 체크
python scripts/evaluate_round_accuracy.py \
    --round 2 --weights models/round2/seq2seq_best.pt \
    --test-dir data/Round2/test --report reports/round2_eval.csv
```

### Round 3

```bash
python data/generate_random_scores.py --round 3 --count 3000 --output data/Round3
python data/generate_random_scores.py --round 3 --count 200  --output data/Round3/test --seed 999

python scripts/train_round.py --round 3 \
    --prev-segnet  models/round2/segnet_best.pt \
    --prev-seq2seq models/round2/seq2seq_best.pt

python scripts/evaluate_round_accuracy.py \
    --round 3 --weights models/round3/seq2seq_best.pt \
    --test-dir data/Round3/test --report reports/round3_eval.csv
```

### Round 4

```bash
python data/generate_random_scores.py --round 4 --count 3000 --output data/Round4
python data/generate_random_scores.py --round 4 --count 200  --output data/Round4/test --seed 999

python scripts/train_round.py --round 4 \
    --prev-segnet  models/round3/segnet_best.pt \
    --prev-seq2seq models/round3/seq2seq_best.pt

python scripts/evaluate_round_accuracy.py \
    --round 4 --weights models/round4/seq2seq_best.pt \
    --test-dir data/Round4/test --report reports/round4_eval.csv
```

---

## Round 5 — 실제 촬영 이미지 도메인 적응

Round 1–4는 합성 악보(MuseScore 렌더링)로 학습합니다.  
Round 5는 **실제 카메라로 촬영한 악보 이미지**에 모델을 적응시키는 단계입니다.

### 필요한 것

| 항목 | 설명 |
|------|------|
| 촬영 PNG | 실제 종이 악보를 카메라로 찍은 이미지 |
| .mscz 파일 | 해당 악보의 MuseScore 파일 (정답 토큰 추출용) |
| JSON 레이블 | .mscz → MusicXML → 토큰 변환 결과 (자동 생성) |

> **.mscz 파일만 준비해서는 바로 학습할 수 없습니다.**  
> `.mscz` → MusicXML → JSON 토큰 변환 단계가 반드시 필요합니다.

### Step 1. .mscz → MusicXML 변환 (MuseScore CLI)

```bash
# 단일 파일
MuseScore4.exe -o output/score.musicxml input/score.mscz

# 폴더 전체 일괄 변환 (Windows PowerShell)
Get-ChildItem data/Round5/raw/*.mscz | ForEach-Object {
    & "C:\Program Files\MuseScore 4\bin\MuseScore4.exe" `
      -o "data/Round5/$($_.BaseName).musicxml" $_.FullName
}
```

### Step 2. MusicXML → JSON 토큰 변환

```bash
python scripts/mscz_to_label.py \
    --input-dir  data/Round5 \
    --output-dir data/Round5
```

> `scripts/mscz_to_label.py`는 별도 작성 필요합니다.  
> music21로 MusicXML을 읽어 `generate_dataset.py`와 동일한 토큰 규칙으로 JSON을 생성합니다.

### Step 3. 학습 (Round 4 가중치에서 파인튜닝)

```bash
python omr/training/train.py \
    --phase    2 \
    --data_dir data/Round5 \
    --tokenizer data/tokenizer.json \
    --out_dir   models/round5 \
    --resume    models/round4/seq2seq_best.pt \
    --epochs    30 \
    --batch     8 \
    --lr        1e-5
```

- 실제 촬영 이미지는 적은 수(수백 장)로도 효과적입니다
- 학습률을 낮게 (`1e-5`) 설정해 기존 Round 1–4 지식을 보존합니다

### Step 4. 정확도 체크

```bash
python scripts/evaluate_round_accuracy.py \
    --round    4 \
    --weights  models/round5/seq2seq_best.pt \
    --test-dir data/Round5/test \
    --report   reports/round5_eval.csv
```

---

## 정확도 체크 옵션 전체 목록

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--round` | 라운드 번호 (1–4) | 필수 |
| `--weights` | `.pt` 가중치 경로 | 필수 |
| `--test-dir` | 테스트 이미지+JSON 디렉토리 | 필수 |
| `--tokenizer` | tokenizer.json 경로 | `data/tokenizer.json` |
| `--threshold` | PASS 기준 정확도 (0.0–1.0) | `0.90` |
| `--max-samples` | 평가 샘플 수 상한 | 전체 |
| `--report` | CSV 저장 경로 | 없음 |
| `--confusion-top` | 혼동 토큰 쌍 출력 개수 | 20 |
| `--segnet-weights` | SegNet `.pt` 경로 (생략 가능) | 없음 |
| `--device` | `auto` / `cuda` / `cpu` | `auto` |
| `--quiet` | 샘플별 진행 로그 숨김 | False |

---

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `ERROR: tokenizer not found` | 경로 오류 | `--tokenizer data/tokenizer.json` 명시 |
| `ERROR: no image+json pairs found` | test 디렉토리에 JSON 없음 | `.png`와 동일 이름의 `.json` 필요 |
| `inference error: size mismatch` | vocab 크기 불일치 | 라운드에 맞는 `.pt` 파일인지 확인 |
| GPU 메모리 부족 | 이미지가 크거나 배치 큼 | `--device cpu` 또는 `--max-samples 30` |
| Round 5 정확도 급락 | 학습률이 너무 높음 | `--lr 1e-5` 이하로 재학습 |
| Round 5 정확도 낮음 | 데이터 수 부족 | 촬영 데이터 100장 이상 확보 권장 |
