# score-training-agent.md

> 담당 에이전트: `score-training-agent`  
> 관련 코드: `ml/omr/data_gen/`, `ml/omr/training/`, `ml/data/`, `ml/scripts/`

## 역할

OMR 모델 학습을 위한 데이터 생성·증강·학습 파이프라인을 관리한다.  
`.mscz` 파일 및 실제 촬영 악보 이미지를 처리하여 학습 데이터셋을 구성하고,  
SegNet → Encoder → Decoder를 Round별로 누적 학습한다.

---

## 전체 학습 흐름

```
[데이터 생성]
  music21 무작위 악보 생성 (generate_random_scores.py)
  + 직접 촬영 악보 사진 (Round 5)
        ↓
[라벨링]
  DeepScore 토큰 시퀀스 생성 (tokenizer.json 기준)
  실사 사진: .mscz → MusicXML → JSON (ml/scripts/mscz_to_label.py ✅)
        ↓
[학습 — RTX 3080]
  SegNet → Encoder → Decoder
  Focal Loss / CrossEntropy / Teacher Forcing
        ↓
[양자화]
  PyTorch → ONNX → onnx2tf → TFLite INT8
        ↓
[C++ 추론 엔진 배포]
  ml/omr/engine/ 빌드 → Flutter 통합
```

---

## 데이터 생성

### 무작위 악보 생성

```bash
python ml/data/generate_random_scores.py --round 1 --count 3000 --output ml/data/Round1
python ml/data/generate_random_scores.py --round 1 --count 200  --output ml/data/Round1/test --seed 999
```

| 파라미터 | 값 |
|---|---|
| 마디 수 | 4–8 마디 |
| 박자표 | 4/4 (50%), 3/4 (25%), 2/4 (15%), 6/8 (10%) |
| 렌더링 | MuseScore 4 CLI → PNG (150 DPI) |
| 출력물 | `.png` + `.musicxml` + `.json` (DeepScore 토큰) |

### Round별 생성 기호 (누적)

| 기호 | Round 1 | Round 2 | Round 3 | Round 4/5 |
|---|---|---|---|---|
| 음표·쉼표·박자표·조표·도돌이표 | ✅ | ✅ | ✅ | ✅ |
| 셈여림 / 크레센도·디미누엔도 | ❌ | ✅ | ✅ | ✅ |
| 아티큘레이션 / 꾸밈음 / 페르마타 / 이음줄 | ❌ | ✅ | ✅ | ✅ |
| 셋잇단음표 / 긴 트릴 / 8va / 겹올림·내림 | ❌ | ✅ | ✅ | ✅ |
| 2오선(grand staff) | ❌ | ❌ | ✅ | ✅ |
| 실사 촬영 이미지 | ❌ | ❌ | ❌ | ✅ |

### Round 5 실사 데이터 처리

```bash
# .mscz → MusicXML (MuseScore CLI)
Get-ChildItem ml/data/Round5/raw/*.mscz | ForEach-Object {
    & "C:\Program Files\MuseScore 4\bin\MuseScore4.exe" `
      -o "ml/data/Round5/$($_.BaseName).musicxml" $_.FullName
}

# MusicXML → JSON 토큰
python ml/scripts/mscz_to_label.py --input-dir ml/data/Round5 --output-dir ml/data/Round5
```

---

## 모델 아키텍처

| 모듈 | 역할 | 아키텍처 |
|---|---|---|
| **SegNet** | 픽셀별 음악 기호 클래스 분류 | MobileNetV3 + U-Net decoder |
| **Encoder** | feature map → latent sequence | ConvNeXt-Tiny |
| **Decoder** | latent → DeepScore 토큰 시퀀스 | Autoregressive Transformer |

### Loss 구성

| 단계 | Loss |
|---|---|
| SegNet | Focal Loss (클래스 불균형 대응) |
| Decoder | CrossEntropy + Teacher Forcing, Label Smoothing 0.1 |

### 학습 환경

- GPU: RTX 3080 (VRAM 10 GB)
- 프레임워크: PyTorch + torch.cuda.amp (mixed precision)
- Optimizer: AdamW + OneCycleLR
- 학습 코드: `ml/omr/training/train.py`

---

## 팀 학습 결과 (2026-06-08 현재)

### 가중치 파일 위치

| 경로 | 내용 | 크기 |
|---|---|---|
| `ml/models/round1/segnet_best.pt` | Round 1 SegNet (round1_fixed 실험) | 29.7 MB |
| `ml/models/round1/seq2seq_best.pt` | Round 1 Seq2Seq Phase 3 완료 | 186 MB |
| `ml/models/round2/segnet_best.pt` | Round 2 SegNet (round2_fixed 실험) | 29.7 MB |
| `ml/models/round2/seq2seq_best.pt` | Round 2 Seq2Seq Phase 3 완료 | 186 MB |
| `ml/reports/round1_fixed_eval.csv` | Round 1 평가 결과 200샘플 | — |
| `ml/reports/round1_fixed_eval_500.csv` | Round 1 평가 결과 500샘플 | — |

원본 실험 폴더: `team_ml/ml/models/round1_fixed/`, `round2_fixed/`

### SegNet 학습 결과

| 실험 | Epoch | 최종 val_acc | 비고 |
|---|---|---|---|
| round1 (기본) | 14 | **98.9%** | 표준 학습 완료 |
| round1_fixed | N/A (중간 로그 없음) | — | 가중치 있음 |
| round2_fixed | 10 | **98.6%** | Round 1 가중치 계승, 1 epoch부터 98%대 시작 |

SegNet은 안정적으로 수렴. 병목이 아님.

### Seq2Seq 학습 결과

#### Round 1 비교 실험

| 실험 | Phase | Epoch | 시작 val_acc | 최종 val_acc | 비고 |
|---|---|---|---|---|---|
| `round1` | 2 | 100 | 26.6% | ~44% | 수렴 정체 — 60에서 개선 없음 |
| `round1_split_w4` | 2 | 40 | 46.8% | ~42% | 오히려 하락 — 불안정 수렴 |
| **`round1_fixed`** | **2** | **60** | **21.3%** | **70.6%** | **핵심 개선 버전** |
| `round1_fixed` | 3 | 20 | 69.3% | **72.0%** | Phase 2 → Phase 3 파인튜닝 +2.7%p |

`round1`과 `round1_fixed`의 차이(44% vs 70%)는 학습 설정 수정(데이터 전처리 또는 lr 스케줄) 덕분으로 추정. 구체적 변경 사항은 팀원 확인 필요.

#### Round 2 결과

| 실험 | Phase | Epoch | 시작 val_acc | 최종 val_acc | 비고 |
|---|---|---|---|---|---|
| `round2_fixed` | 2 | 68 | 61.0% | **65.2%** | round1_fixed 계승, Round 2 기호 추가 학습 |
| `round2_fixed` | 3 | 30 | 65.1% | **64.4%** | Phase 3에서 소폭 하락 |

Round 1 → Round 2에서 Seq2Seq 성능이 **70%→64%로 역전**. 원인 분석 아래 참조.

### 공식 평가 결과 (round1_fixed 모델)

`ml/reports/round1_fixed_eval.csv` — 200샘플 (어려운 테스트셋)

| 지표 | 결과 |
|---|---|
| Pass Rate (완전 일치) | **64.5%** (129/200) |
| 평균 TER | **0.156** |
| 평균 Accuracy (전체 토큰) | **84.4%** |
| 평균 Note Accuracy (음표만) | **79.2%** |

`ml/reports/round1_fixed_eval_500.csv` — 500샘플

| 지표 | 결과 |
|---|---|
| Pass Rate | **76.8%** (384/500) |
| 평균 TER | **0.083** |
| 평균 Accuracy | **91.7%** |
| 평균 Note Accuracy | **89.0%** |

> 200샘플 세트가 더 어려운 악보를 포함하거나 out-of-distribution일 가능성 있음. 두 세트 간 TER 2배 차이(0.083 vs 0.156)는 평가 데이터 편차로 인한 것.

---

## Round별 학습 계획

### Round 1 ✅ 완료

```bash
# round1_fixed 실험 결과 가중치 사용
# ml/models/round1/segnet_best.pt   — SegNet (98%+)
# ml/models/round1/seq2seq_best.pt  — Seq2Seq Phase 3 (val_acc 72%, TER 0.28)
```

### Round 2 ✅ 완료 (재학습 권장)

```bash
python ml/scripts/train_round.py --round 2 \
    --prev-segnet  ml/models/round1/segnet_best.pt \
    --prev-seq2seq ml/models/round1/seq2seq_best.pt
```

현재 `ml/models/round2/seq2seq_best.pt`는 Round 1보다 낮은 성능(64%). **아래 개선 방안 적용 후 재학습 권장.**

### Round 3

Round 2 체크포인트에서 fine-tuning (lr=1e-5). 2오선 grand staff 추가.

### Round 4

Round 3 체크포인트에서 fine-tuning (lr=5e-6). 실사 사진 도입.

### Round 5 — 실제 촬영 도메인 적응

```bash
python ml/omr/training/train.py \
    --phase 2 --data_dir ml/data/Round5 \
    --resume ml/models/round4/seq2seq_best.pt \
    --epochs 30 --batch 8 --lr 1e-5
```

---

## 데이터 요구량

| 모듈 | 최소 | 권장 | 현재 |
|---|---|---|---|
| SegNet (픽셀 분류) | 2,000 악보 | 20,000+ | 생성 예정 |
| Encoder-Decoder | 5,000 staff | 50,000+ | 생성 예정 |
| 실사 사진 | 200장 | 1,000+ | Round 5 이후 |

증강 ×8 적용 시: 1,000 악보 → 8,000 유효 이미지

---

## 정확도 원인 분석 및 개선 방안

### 현재 낮은 정확도의 근본 원인

#### 1. 데이터 생성 문제

**타일 첫 번째만 학습됨 (`dataset.py:421` `tiles_per_staff=1`)**
`OMRDataset` 생성 시 `tiles_per_staff=1`이 기본값이므로 오선의 첫 번째 1280px 타일만 학습에 사용된다. 4~8 마디 악보 전체가 2~3 타일에 걸쳐 있으면 후반부 마디를 전혀 학습하지 않는 것과 같다.

**증강 불충분 (`dataset.py:498` `perspective=False`)**
`OMRDataset.__getitem__`에서 `augment_image(gray, perspective=False)`로 호출한다. 회전·원근 변환이 빠져 있어 촬영 악보와의 도메인 갭이 크다. `augment_image()`에 JPEG 압축 아티팩트 시뮬레이션도 없다.

**음악적 구조 없는 무작위 음표**
`generate_dataset.py`가 완전 랜덤 음표 배열을 생성하므로, 실제 악보에 나타나는 화성 진행·리듬 패턴·반복 구조를 학습하지 못한다. 모델이 단일 음표 수준은 맞혀도 연속적 문맥을 예측하기 어렵다.

**Round 2에서 val_acc 역전 (원인 미확정)**
`round2_fixed` Phase 2는 Round 1 가중치를 계승했음이 확인된다 (epoch 1에서 이미 61% 시작). `70% → 64%` 역전의 원인은 두 가지 가설이 있으나 `train.py`의 val split 로직 확인이 필요하다:
- **가설 A**: val 세트가 동일한데도 성능 퇴행 → Round 2 데이터로만 fine-tuning하면서 Round 1 패턴이 일부 희석됨
- **가설 B**: val 세트가 Round 2 데이터(더 복잡한 기호 포함)로 바뀌어 수치가 낮게 나온 것 → 실제 퇴행이 아닐 수 있음

#### 2. 추론 파이프라인 문제

**타일 독립 디코딩 (`omr_inference.py:148-153`)**
각 타일을 독립적으로 디코딩해 결과를 단순 concat한다. 타일 경계에서 마디가 잘리면 토큰 중복이나 누락이 발생한다.

**Greedy decoding**
`_run_decoder()`가 greedy argmax만 사용한다. Beam search(beam=3~5)로 전환하면 5~10%p 성능 향상을 기대할 수 있다.

---

### 개선 방안 (우선순위 순)

#### 즉시 적용: 코드 수정만으로 가능 (유료 데이터셋 불필요)

**① 타일 전수 학습** ← 효과 가장 큼, 구현 즉시 가능

`OMRDataset` 생성 시 `tiles_per_staff` 기본값을 높이거나, 타일 전체를 개별 샘플로 등록해야 한다.

```python
# train.py 또는 train_round.py 호출 시
dataset = OMRDataset(data_dir, tokenizer, tiles_per_staff=3)  # 1 → 3
```

단, 타일별 GT 토큰을 분할해야 정확하다. 현재는 타일이 달라도 동일한 전체 JSON을 GT로 사용하므로 타일-토큰 정렬 오류가 발생한다. 단기 해결책: tiles_per_staff를 올리되 GT 토큰 분할은 다음 단계로 미룬다 (성능이 현재보다는 나아질 것).

**② OMRDataset에 perspective 증강 활성화**

```python
# dataset.py:498
gray = augment_image(gray, perspective=True)  # False → True
```

추가로 `augment_image()`에 JPEG 압축 아티팩트를 추가한다:
```python
if random.random() < 0.3:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 90)]
    _, buf = cv2.imencode('.jpg', out, encode_param)
    out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
```

**③ Round 2 재학습 전 원인 확인**

먼저 `ml/omr/training/train.py`의 val split 로직을 확인해야 한다. val 세트가 라운드별로 바뀌는 구조라면 수치 비교 자체가 의미 없을 수 있다.

원인이 **가설 A(실제 퇴행)**로 확인된 경우에만 Round 1 + Round 2 데이터 합산 학습을 적용한다:

```bash
# Round 1 + Round 2 데이터를 하나의 디렉토리로 합쳐서 학습
python ml/scripts/train_round.py --round 2 \
    --data-dir ml/data/Round1_Round2_combined \
    --prev-segnet  ml/models/round1/segnet_best.pt \
    --prev-seq2seq ml/models/round1/seq2seq_best.pt \
    --epochs-p2 80 --epochs-p3 15
```

**④ Phase 3 설정 조정**

`round2_fixed`에서 Phase 3 30 epoch이 오히려 성능을 하락시켰다. Phase 3은 다음 설정을 권장:
- epochs: 30 → **15**
- lr_p3: 2e-5 → **8e-6**
- early stopping: val TER이 3 epoch 연속 개선 없으면 종료

#### 중기: IMSLP 실사 악보 활용 (이미 파이프라인 존재)

`ml/data/imslp_pdf/pd_manifest.txt`와 `ml/scripts/imslp_pdf_to_images.py`가 이미 구현되어 있다. IMSLP 저작권 만료 악보를 PDF → 이미지로 변환하는 파이프라인이 준비됐으나 **미사용** 상태다.

```bash
# 이미 준비된 파이프라인 활용
python ml/scripts/imslp_pdf_to_images.py  # PDF → PNG 변환
# 이후 Round 4 실사 fine-tuning에 포함
```

이 데이터를 Round 4 fine-tuning에 넣으면 합성 데이터와 실제 인쇄 악보 간 도메인 갭이 크게 줄어든다. **유료 데이터셋 구매 전에 이 단계를 먼저 실행해야 한다.**

#### 장기: 아키텍처·디코더 개선

**Beam search 추가** (`omr_inference.py:_run_decoder`)
```python
# 현재: greedy argmax
next_id = int(np.argmax(logits[0, 0]))
# 목표: beam search (beam=3~5)
```

**타일 간 컨텍스트 전달**
현재 타일 독립 디코딩 → 인접 타일 간 이전 타일의 마지막 n개 토큰을 prefix로 넘겨 마디 경계 오류 감소.

---

### 유료 악보 데이터셋이 필요한 시점

아래 순서로 시도한 뒤에도 val_acc가 75% 미만이면 유료 데이터셋을 검토한다:

1. ✅ **먼저**: 타일 전수 학습 + 증강 강화 + 누적 학습 → 기대 효과: +10~15%p
2. ✅ **그 다음**: IMSLP 실사 데이터 fine-tuning (무료) → 기대 효과: +5~10%p
3. ⬜ **필요 시**: [MUSCIMA++](https://ufal.mff.cuni.cz/muscima), [DeepScores v2](https://zenodo.org/record/4812962) 등 수작업 annotation 데이터셋 검토

현재 합성 데이터 4,000장 × 3라운드 = 12,000장 규모에서 데이터 품질 문제를 먼저 해결하는 것이 더 효율적이다.

---

### 증상별 빠른 처방표

| 증상 | 의심 원인 | 처방 |
|---|---|---|
| Round 2가 Round 1보다 낮음 | Catastrophic forgetting | Round 1+2 데이터 합산 누적 학습 |
| train loss 감소, val loss 발산 | 과적합 | 데이터 확대 + 증강 강화 |
| train loss도 수렴 안 함 | 학습률 과다 | max_lr 1e-3 → 5e-4 |
| TER 낮은데 음높이만 틀림 | Decoder 토큰 혼동 | Confusion Matrix → 해당 범위 데이터 비율 상향 |
| TER 낮은데 음가만 틀림 | 음가 분포 불균형 | `_DUR_WEIGHTS` 조정 + 희귀 음가 loss 가중치 ×2 |
| 토큰 아예 없음 | StaffDetector/전처리 실패 | `detect_staffs()` 파라미터 확인 |
| 후반부 마디가 특히 틀림 | 타일 첫 번째만 학습 | `tiles_per_staff` 상향 |
