# OMR 라운드별 정확도 체크 매뉴얼

라운드별 학습이 완료된 `.pt` 가중치 파일의 정확도를 검증하는 절차입니다.  
Round 1 기준으로 설명하며, Round 2·3·4도 경로만 바꾸면 동일하게 적용됩니다.

---

## 사전 준비

### 1. 패키지 설치 확인

```bash
pip install torch torchvision opencv-python numpy tqdm
```

### 2. 디렉토리 구조 확인

```
musicscore_flutter/
├── data/
│   ├── tokenizer.json          ← 공통 tokenizer (모든 라운드 공유)
│   ├── Round1/
│   │   ├── test/               ← 테스트 이미지 + JSON 쌍 (아래 형식 참고)
│   │   │   ├── score_0001.png
│   │   │   ├── score_0001.json  ← {"tokens": ["clef-G", "key-C", ...]}
│   │   │   └── ...
│   │   └── ...
│   └── round_tokens/
│       └── round1_tokens.json
├── models/
│   └── round1/
│       ├── seq2seq_best.pt     ← 팀원이 완료한 가중치 파일
│       └── segnet_best.pt
└── scripts/
    └── evaluate_round_accuracy.py
```

> **테스트 이미지 형식**: `score_XXXX.png` + `score_XXXX.json` 쌍.  
> JSON 안에 `"tokens"` 키로 정답 토큰 리스트가 있어야 합니다.

---

## Round 1 정확도 체크

### 기본 실행

```bash
python scripts/evaluate_round_accuracy.py \
    --round 1 \
    --weights   models/round1/seq2seq_best.pt \
    --test-dir  data/Round1/test \
    --tokenizer data/tokenizer.json
```

### 출력 예시

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

  Sample                    GT  Pred    TER  NoteAcc
  ----------------------  ----  ----  -----  -------
  score_0001               24    23   4.2%   96.8%  PASS
  score_0002               31    33   9.7%   89.2%  FAIL
  ...
======================================================================
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--round` | 라운드 번호 (1–4) | 필수 |
| `--weights` | `.pt` 가중치 파일 경로 | 필수 |
| `--test-dir` | 테스트 이미지+JSON 디렉토리 | 필수 |
| `--tokenizer` | tokenizer.json 경로 | `data/tokenizer.json` |
| `--threshold` | PASS 기준 정확도 (0.0–1.0) | `0.90` (90%) |
| `--max-samples` | 평가 샘플 수 상한 | 전체 |
| `--report` | CSV 리포트 저장 경로 | 없음 |
| `--confusion-top` | 혼동 토큰 쌍 출력 개수 | 20 |
| `--device` | `auto` / `cuda` / `cpu` | `auto` |
| `--quiet` | 샘플별 진행 로그 숨김 | False |

---

## Round 2·3·4 적용 방법

경로만 바꾸면 동일하게 동작합니다.

### Round 2

```bash
python scripts/evaluate_round_accuracy.py \
    --round 2 \
    --weights   models/round2/seq2seq_best.pt \
    --test-dir  data/Round2/test \
    --tokenizer data/tokenizer.json
```

### Round 3

```bash
python scripts/evaluate_round_accuracy.py \
    --round 3 \
    --weights   models/round3/seq2seq_best.pt \
    --test-dir  data/Round3/test \
    --tokenizer data/tokenizer.json
```

### Round 4

```bash
python scripts/evaluate_round_accuracy.py \
    --round 4 \
    --weights   models/round4/seq2seq_best.pt \
    --test-dir  data/Round4/test \
    --tokenizer data/tokenizer.json
```

---

## 권장 워크플로우

### 1. 기본 정확도 확인

```bash
python scripts/evaluate_round_accuracy.py \
    --round 1 \
    --weights models/round1/seq2seq_best.pt \
    --test-dir data/Round1/test
```

### 2. CSV 리포트 저장

결과를 파일로 저장해 팀원과 공유할 때:

```bash
python scripts/evaluate_round_accuracy.py \
    --round 1 \
    --weights   models/round1/seq2seq_best.pt \
    --test-dir  data/Round1/test \
    --report    reports/round1_eval.csv \
    --confusion-top 30
```

### 3. 빠른 샘플 검증 (50장만)

전체 실행 전 빠르게 확인:

```bash
python scripts/evaluate_round_accuracy.py \
    --round 1 \
    --weights     models/round1/seq2seq_best.pt \
    --test-dir    data/Round1/test \
    --max-samples 50 \
    --quiet
```

### 4. SegNet 포함 End-to-End 평가

SegNet 가중치가 있을 때 전체 파이프라인 평가:

```bash
python scripts/evaluate_round_accuracy.py \
    --round 1 \
    --weights        models/round1/seq2seq_best.pt \
    --segnet-weights models/round1/segnet_best.pt \
    --test-dir       data/Round1/test
```

> **참고**: `--segnet-weights` 생략 시 간단한 canvas crop 방식으로 이미지 전처리를 대체합니다.  
> Full 파이프라인(TFLite)이 아닌 PyTorch 기반 평가이므로 실제 앱 정확도와 소폭 차이가 있을 수 있습니다.

---

## 결과 해석

| 지표 | 설명 | 라운드별 목표 |
|------|------|--------------|
| **Overall Acc** | 전체 토큰 정확도 (1 - TER) | R1: ≥90% / R2: ≥88% / R3: ≥85% / R4: ≥82% |
| **Note-only Acc** | 음표·쉼표만의 정확도 | Overall보다 3–5%p 높으면 정상 |
| **TER** | Token Error Rate (낮을수록 좋음) | <10% 목표 |
| **Per-Token Recall** | 토큰 유형별 재현율 | 빈도 높은 토큰(barline, note-X4-1/4 등) ≥95% |

### PASS / FAIL 기준 (기본 threshold=0.90)

- **PASS**: 평균 Overall Acc ≥ 90%
- **FAIL**: 평균 Overall Acc < 90% → 추가 학습 또는 데이터 보강 필요

PASS 기준을 조정하려면 `--threshold 0.85` 처럼 지정합니다.

---

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `ERROR: tokenizer not found` | tokenizer.json 경로 오류 | `--tokenizer` 옵션으로 정확한 경로 지정 |
| `ERROR: no image+json pairs found` | test 디렉토리에 JSON 없음 | `.png`와 동일 이름의 `.json` 파일 필요 |
| `inference error: size mismatch` | vocab 크기 불일치 | 라운드에 맞는 가중치 파일인지 확인 |
| GPU 메모리 부족 | 큰 배치 / 큰 이미지 | `--device cpu` 또는 `--max-samples 30` |
| 정확도가 예상보다 낮음 | 이전 라운드 가중치 로드 실패 | `--segnet-weights` 경로 확인 |

---

## 다음 라운드 학습 시작

Round 1 정확도 확인 후 Round 2 학습을 시작하려면:

```bash
python scripts/train_round.py \
    --round 2 \
    --prev-segnet  models/round1/segnet_best.pt \
    --prev-seq2seq models/round1/seq2seq_best.pt
```

자세한 학습 옵션은 `python scripts/train_round.py --help` 참고.
