# sheet-music-qa.md

> 담당 에이전트: `sheet-music-qa`  
> 관련 코드: `ml/scripts/evaluate_round_accuracy.py`, `ml/reports/`

## 역할

각 Round 학습 완료 후 OMR 인식 정확도를 평가하고 PASS/FAIL을 판정한다.  
단순 기호에서 복잡한 악보까지 누적 학습 진행 상황을 체크포인트별로 검증한다.

---

## 평가 지표

| 지표 | 설명 | 목표 |
|---|---|---|
| **TER** (Token Error Rate) | (삽입+삭제+치환) / 정답 토큰 수 | 0.10 이하 |
| **Note-only TER** | 음표·쉼표 토큰만 추출 후 TER | TER보다 엄격하게 추적 |
| **Sequence Exact Match** | 전체 시퀀스 완벽 일치 악보 비율 | 50%+ (초기), 80%+ (최종) |
| **Pitch Accuracy** | 음높이만 비교 | 95%+ |
| **Duration Accuracy** | 음가만 비교 | 90%+ |

### Round별 PASS 기준

| 라운드 | Overall Acc 목표 |
|---|---|
| Round 1 | ≥ 90% |
| Round 2 | ≥ 88% |
| Round 3 | ≥ 85% |
| Round 4 | ≥ 82% |
| Round 5 | ≥ 80% |

---

## 평가 실행 명령

```bash
python ml/scripts/evaluate_round_accuracy.py \
    --round    1 \
    --weights  ml/models/round1/seq2seq_best.pt \
    --test-dir ml/data/Round1/test \
    --tokenizer ml/data/tokenizer.json \
    --report   ml/reports/round1_eval.csv
```

### 출력 예시

```
======================================================================
  OMR Round-1 Accuracy Report
======================================================================
  Weights    : ml/models/round1/seq2seq_best.pt
  Test dir   : ml/data/Round1/test
  Samples    : 200 found  /  200 evaluated
  Threshold  : 90.0%

  RESULT     : PASS
  Overall    : 93.5%  (TER 6.5%)
  Note-only  : 95.2%
  Per-sample : 186/200 pass threshold
======================================================================
```

---

## Round 2·3·4·5 — 동일 패턴

```bash
# Round 2
python ml/scripts/evaluate_round_accuracy.py \
    --round 2 --weights ml/models/round2/seq2seq_best.pt \
    --test-dir ml/data/Round2/test --report ml/reports/round2_eval.csv

# Round 3
python ml/scripts/evaluate_round_accuracy.py \
    --round 3 --weights ml/models/round3/seq2seq_best.pt \
    --test-dir ml/data/Round3/test --report ml/reports/round3_eval.csv

# Round 4
python ml/scripts/evaluate_round_accuracy.py \
    --round 4 --weights ml/models/round4/seq2seq_best.pt \
    --test-dir ml/data/Round4/test --report ml/reports/round4_eval.csv

# Round 5 (실사 사진)
python ml/scripts/evaluate_round_accuracy.py \
    --round 4 --weights ml/models/round5/seq2seq_best.pt \
    --test-dir ml/data/Round5/test --report ml/reports/round5_eval.csv
```

---

## PyTorch 체크포인트 직접 평가 파이프라인 (TFLite 변환 전)

```
test.png
  ↓ [ml/omr/utils/pt_predict.py] ← round1_best.pth + tokenizer.json
predicted.json
  ↓ [ml/omr/utils/tokens_to_musicxml.py]
predicted.musicxml ──→ [compare_musicxml.py] ──→ 정확도 리포트
                   ──→ [evaluate.py TER]      ──→ TER / note-only TER
```

### 일괄 평가

```bash
for f in ml/data/test/*.png; do
    stem=$(basename $f .png)
    python ml/omr/utils/pt_predict.py \
        --image $f --ckpt checkpoints/round1_best.pth \
        --tokenizer ml/data/tokenizer.json \
        --out_xml ml/output/${stem}_pred.musicxml
done
python ml/omr/utils/evaluate.py \
    --test_dir ml/data/test --pred_dir ml/output \
    --tokenizer ml/data/tokenizer.json
```

---

## 평가 옵션 전체 목록

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--round` | 라운드 번호 | 필수 |
| `--weights` | `.pt` 가중치 경로 | 필수 |
| `--test-dir` | 테스트 이미지+JSON 디렉토리 | 필수 |
| `--tokenizer` | tokenizer.json 경로 | `ml/data/tokenizer.json` |
| `--threshold` | PASS 기준 정확도 | `0.90` |
| `--max-samples` | 평가 샘플 수 상한 | 전체 |
| `--report` | CSV 저장 경로 | 없음 |
| `--confusion-top` | 혼동 토큰 쌍 출력 개수 | 20 |
| `--device` | `auto` / `cuda` / `cpu` | `auto` |

---

## 측정 시점

| 시점 | 측정 항목 |
|---|---|
| 매 epoch | train loss, val loss (TensorBoard) |
| 매 5 epoch | val TER, note-only TER |
| Round 완료 시 | Exact Match + Pitch/Duration + Confusion Matrix |
| TFLite 변환 후 | FP32 대비 INT8 TER 차이 |

---

## 시각적 검증

`ml/omr/utils/render_notation.py`로 예측 토큰 → 커스텀 악보 PNG 렌더링 후 정답 라벨과 나란히 비교.

---

## 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `ERROR: tokenizer not found` | 경로 오류 | `--tokenizer ml/data/tokenizer.json` 명시 |
| `ERROR: no image+json pairs found` | test 디렉토리에 JSON 없음 | `.png`와 동일 이름의 `.json` 필요 |
| `inference error: size mismatch` | vocab 크기 불일치 | 라운드에 맞는 `.pt` 파일인지 확인 |
| GPU 메모리 부족 | 이미지 크기 또는 배치 큼 | `--device cpu` 또는 `--max-samples 30` |
| Round 5 정확도 급락 | 학습률이 너무 높음 | `--lr 1e-5` 이하로 재학습 |
