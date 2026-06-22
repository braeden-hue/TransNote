# Round 1 학습 완료 후 검증 가이드

Round 1 학습이 끝난 뒤 이 가이드 순서대로 실행해 PASS/FAIL 판정을 내린다.

---

## 합격 기준 (PASS)

| 지표 | PASS | 재학습 검토 | FAIL |
|---|---|---|---|
| val_TER | < 0.25 (Acc > 75%) | 0.25~0.35 | > 0.35 |
| Note Acc | > 80% | 70~80% | < 70% |
| Pass rate (TER=0) | > 5% | 1~5% | < 1% |

---

## 1단계 — 학습 로그 확인

학습 완료 직후 `round1/models/` 에 생성된 CSV 로그를 확인한다.

```
round1/models/
  seq2seq_best.pt    ← 가장 낮은 val_TER 에폭
  segnet_best.pt     ← 가장 낮은 segnet val_loss 에폭
  train_log.csv      ← 에폭별 loss / TER 기록 (있는 경우)
```

**확인 사항**
- `seq2seq_best.pt` 의 `val_TER` 값 (체크포인트 파일 내 `best_ter` 키)
- 학습 로그에서 TER이 에폭이 늘수록 감소하는 추세인지

```python
import torch
ckpt = torch.load("round1/models/seq2seq_best.pt", map_location="cpu", weights_only=False)
print("best epoch :", ckpt.get("epoch"))
print("val TER    :", ckpt.get("best_ter"))
print("vocab size :", ckpt.get("vocab_size"))
```

---

## 2단계 — 테스트 데이터 생성 (300장)

학습 데이터와 겹치지 않도록 `--start-idx 9001` 사용.

```bash
python round1/generate_scores.py \
    --count 300 \
    --start-idx 9001 \
    --output round1/Round1_test \
    --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
```

> MuseScore가 없으면 `--no-png` 플래그 추가 (XML+JSON만 생성). PNG 없이는 inference를 실행할 수 없으므로 반드시 MuseScore 설치 후 재실행.

---

## 3단계 — 배치 평가 실행

```bash
python round1/inference.py \
    --seq2seq    round1/models/seq2seq_best.pt \
    --tokenizer  round1/tokenizer.json \
    --eval_dir   round1/Round1_test \
    --n_eval     200
```

**출력 예시 (PASS 기준)**
```
──────────────────────────────────────────────────
샘플       : 200
TER        : 22.4%
Acc (1-TER): 77.6%        ← 75% 이상이면 PASS
Note Acc   : 83.1%        ← 80% 이상이면 PASS
Pass (TER=0): 6.5%  (13/200)
소요 시간  : 94.2s (0.47s/샘플)
```

---

## 4단계 — 토큰 카테고리별 점검

inference 로그에서 오류 패턴을 확인한다. 주요 체크 항목:

| 토큰 유형 | 예상 오류 원인 | 확인 방법 |
|---|---|---|
| `clef-G` / `clef-F` | 보표 시작 누락 | 첫 토큰이 clef인지 확인 |
| `key-*` | 조성 혼동 | key 토큰 앞뒤 순서 점검 |
| `barline` | 마디선 누락/삽입 | TER 분해 후 barline 오류 비율 |
| `note-*-1/4` | 음가 오인식 | 8분음표와 4분음표 혼동 여부 |
| `barline-start/end-repeat` | 도돌이표 혼동 | 반복 기호 포함 샘플만 별도 평가 |

**단일 이미지 추론 (디버깅용)**
```bash
python round1/inference.py \
    --seq2seq   round1/models/seq2seq_best.pt \
    --tokenizer round1/tokenizer.json \
    round1/Round1_test/num9001.png
```

---

## 5단계 — 판정 및 다음 단계

### PASS (val_TER < 0.25, Acc > 75%)

Round 2 재학습 진행.

```bash
# Round 2 학습 명령 예시 (누적 데이터 사용)
python round2/train.py \
    --phase 2 \
    --data_dir  "round2/Round1+Round2" \
    --out_dir   round2/models \
    --resume    round1/models/seq2seq_best.pt
```

### 재학습 검토 (TER 0.25~0.35)

아래 항목 중 하나 이상 적용 후 Round 1 재학습:

- `--tiles_per_staff` 증가 (데이터 증강 강화)
- 에폭 수 증가 (`--epochs 30`)
- 학습률 낮추기 (`--lr 5e-5`)
- 데이터 2000장 → 4000장으로 증가

### FAIL (TER > 0.35)

- 데이터 품질 점검: `round1/Round1_test/` 에서 PNG 육안 확인
- SegNet 단독 학습 추가 (`--phase 1` 먼저 실행)
- `dataset.py` 의 `tiles_per_staff`, augmentation 파라미터 조정

---

## 참고: 학습 재시작 명령

```bash
# Round 1 처음부터 (2000장 이상)
python round1/generate_scores.py \
    --count 2000 \
    --output round1/Round1 \
    --musescore "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"

python round1/train.py \
    --phase 2 \
    --data_dir round1/Round1 \
    --out_dir  round1/models
```

---

> 최종 수정: 2026-06-22  
> 기준 가중치: `seq2seq_best.pt` (vocab=1012, Round 1 기호 완성)
