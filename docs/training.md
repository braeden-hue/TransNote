# training.md — OMR 학습 파이프라인

> 관련 코드: `omr/data_gen/generate_dataset.py`, `omr/training/`

---

## 1. 전체 학습 흐름

```
[데이터 생성]
  music21 무작위 악보 생성 + 직접 촬영 악보 사진
        ↓
[라벨링]
  DeepScore 토큰 시퀀스 생성 (→ docs/custom.md 참조)
        ↓
[학습 — RTX 3080]
  SegNet → Encoder → Decoder
  Focal Loss / CrossEntropy / Teacher Forcing
        ↓
[양자화]
  PyTorch → ONNX → onnx2tf → TFLite INT8 (→ docs/engine.md 참조)
        ↓
[C++ 추론 엔진 배포]
  omr/engine/ 빌드 → Flutter 통합 (→ docs/fluttering.md 참조)
```

---

## 2. 데이터 생성 전략

### 무작위 악보 생성 (`generate_dataset.py`)

| 파라미터 | 값 |
|---|---|
| 마디 수 | 4–8 마디 |
| 박자표 | 4/4 (50%), 3/4 (25%), 2/4 (15%), 6/8 (10%) |
| 렌더링 | MuseScore 4 CLI → PNG (150 DPI, A4 ≈ 1240×1754 px) |
| 출력물 | `.png` + `.musicxml` + `.json` (DeepScore 토큰) |

### Round별 생성 기호 (누적)

| 기호 | Round 1 | Round 2 |
|---|---|---|
| 음표·쉼표·박자표·조표·도돌이표 | ✅ | ✅ |
| 셈여림 / 크레센도·디미누엔도 | ❌ | ✅ |
| 아티큘레이션 / 꾸밈음 / 페르마타 / 이음줄 | ❌ | ✅ |
| 셋잇단음표 / 긴 트릴 / 8va / 겹올림·내림 / 네츄럴 | ❌ | ✅ |

### 생성 확률 (Round 2 현재 설정)

| 기호 | 확률 |
|---|---|
| 도돌이표 | 25% |
| 셈여림 | 1마디당 40% (이후 마디 35% 감소) |
| 크레센도·디미누엔도 | 마디당 25% |
| 아티큘레이션 | 음표당 20% |
| 꾸밈음 | 음표당 8% |
| 페르마타 | 마디당 20% |
| 이음줄 | 마디당 15% |
| 셋잇단음표 | 마디당 20% |
| 긴 트릴 | 마디당 10% |
| 8va | 마디당 10% (높은음자리만) |
| 겹올림·겹내림 | 음표당 8% |
| 마디 내 네츄럴 | 음표당 15% |

### 실사 데이터 확대 (Round 4 이후)

- 직접 촬영 악보 사진 추가
- `xml_to_json.py` (미구현)로 라벨 자동 생성
- 저작권 없는 악보만 사용 (작곡가 사망 70년+)
- 오프라인 증강 `augment.py` (미구현): 원근 왜곡·밝기·노이즈·블러 ×4–8

---

## 3. 입력 이미지 해상도 정책

| 항목 | 값 | 근거 |
|---|---|---|
| 학습 데이터 생성 DPI | 150 DPI | A4 ≈ 1240×1754 px, 스마트폰 중간 해상도와 유사 |
| 추론 입력 최대 너비 | 1280 px | 전처리 단계에서 통일 |
| 종횡비 | 유지 | autocrop → 비율 유지 resize |

---

## 4. 모델 아키텍처

| 모듈 | 역할 | 아키텍처 | 라이선스 |
|---|---|---|---|
| **SegNet** | 픽셀별 음악 기호 클래스 분류 | MobileNetV3 + U-Net decoder | Apache 2.0 |
| **Encoder** | feature map → latent sequence | ConvNeXt-Tiny | MIT |
| **Decoder** | latent → DeepScore 토큰 시퀀스 | Autoregressive Transformer (Vaswani 2017) | 공개 알고리즘 |

### Loss 구성

| 단계 | Loss |
|---|---|
| SegNet | Focal Loss (클래스 불균형 대응) |
| Decoder | CrossEntropy + Teacher Forcing, Label Smoothing 0.1 |
| 전체 | SegNet loss + λ × Decoder loss |

### 학습 환경

- GPU: RTX 3080 (VRAM 10 GB)
- 프레임워크: PyTorch + torch.cuda.amp (mixed precision)
- Optimizer: AdamW + OneCycleLR
- 학습 코드: `omr/training/train.py`

---

## 5. Round별 학습 계획 (한 Round = 변수 하나씩 증가)

```
Round 1: 단선율(1오선) + 기본 음표  + 디지털 악보  ← 다음 실행 (3,000장 생성 후 학습)
Round 2: 단선율(1오선) + 전체 기호  + 디지털 악보  (vocab 1004→1012, data/Round2/ 준비 완료)
Round 3: 2오선(grand) + 전체 기호  + 디지털 악보
Round 4: 2오선        + 전체 기호  + 실사 사진
```

### Round 1 — 기본 음표 3,000장

- SegNet 단독 사전학습 `--epochs 50 --batch 16`
- Encoder+Decoder 학습 `--epochs 100 --batch 8`
- End-to-end fine-tuning `--epochs 30 --batch 4`
- 완료 기준: val TER 안정적 감소 확인

### Round 2 — 새 기호 포함 단선율 누적 추가

- 변경 변수: 기호 종류 확장 (오선 수·이미지 종류 유지)
- Round 1 체크포인트에서 fine-tuning (lr=3e-5)
- `load_checkpoint_with_vocab_expansion()` 사용 (vocab 1004→1012 확장)
- 데이터 폴더: `data/Round2/`
- PNG 검증 선행 필수 (기호가 실제로 PNG에 렌더링되는지 확인)

### Round 3 — 2오선 디지털 악보

- 변경 변수: 오선 수 1→2
- 사전 작업:
  - `generate_dataset.py`에 2-Part 생성 옵션 추가
  - JSON 포맷 변경: `{"staves": [{"tokens":[...]}, {"tokens":[...]}]}`
  - `dataset.py` OMRDataset staves[i] 인덱스 매칭 방식으로 수정
  - `augment.py` 구현 (밝기·노이즈·원근 왜곡 ×8)
- Round 2 체크포인트에서 fine-tuning (lr=1e-5)

### Round 4 — 실사 사진 fine-tuning

- 변경 변수: 이미지 종류 디지털→실사
- 사전 작업:
  - `xml_to_json.py` 구현 (mscz → musicxml → staves[] JSON)
  - 실사 사진 + mscz 파일 준비 (저작권 없는 악보, 최소 200장)
  - `augment.py` 원근 왜곡 강도 상향
- Round 3 체크포인트에서 fine-tuning (lr=5e-6)
- 완료 기준: 실사 val TER 0.10 이하 (90%+ 인식률)

---

## 6. 데이터 요구량

| 모듈 | 최소 | 권장 | 현재 |
|---|---|---|---|
| SegNet (픽셀 분류) | 2,000 악보 | 20,000+ | 학습 코드 완료, 데이터 생성 예정 |
| Encoder-Decoder | 5,000 staff | 50,000+ | 학습 코드 완료, Round 1 실행 예정 |
| 실사 사진 | 200장 | 1,000+ | 0 (Round 4 이후) |

증강 ×8 적용 시: 1,000 악보 → 8,000 유효 이미지

### 권장 학습 순서

```bash
python omr/data_gen/generate_dataset.py -n 5000   # 5,000 악보 생성
# augment.py 실행 → ×8 = 40,000 이미지
# SegNet 단독 사전학습 (~50 epoch, Focal Loss)
# Encoder+Decoder end-to-end (~100 epoch, Teacher Forcing)
# 실사 사진 fine-tuning (~20 epoch)
# PyTorch → ONNX → TFLite INT8 PTQ 변환
```

---

## 7. 정확도 측정 방법

| 지표 | 설명 | 목표 |
|---|---|---|
| **TER** | (삽입+삭제+치환) / 정답 토큰 수 | 0.10 이하 |
| **Note-only TER** | 음표·쉼표 토큰만 추출해 TER 측정 | TER보다 엄격하게 추적 |
| **Sequence Exact Match** | 전체 시퀀스 완벽 일치 악보 비율 | 50%+ (초기), 80%+ (최종) |
| **Pitch Accuracy** | 음높이만 비교 | 95%+ |
| **Duration Accuracy** | 음가만 비교 | 90%+ |

### 측정 시점

| 시점 | 측정 항목 |
|---|---|
| 매 epoch | train loss, val loss (TensorBoard) |
| 매 5 epoch | val TER, note-only TER |
| Round 완료 시 | Exact Match + Pitch/Duration + Confusion Matrix |
| TFLite 변환 후 | FP32 대비 INT8 TER 차이 |

### 시각적 검증

`omr/utils/render_notation.py` 로 예측 토큰 → 커스텀 악보 PNG 렌더링 후 정답 라벨과 나란히 비교.

---

## 8. Round 1 평가 파이프라인

> TFLite 변환 없이 PyTorch 체크포인트 단계에서 바로 정확도 확인.  
> 구현 위치: `omr/utils/pt_predict.py`, `omr/utils/tokens_to_musicxml.py`

```
test.png
  ↓ [pt_predict.py] ← round1_best.pth + tokenizer.json
predicted.json
  ↓ [tokens_to_musicxml.py]
predicted.musicxml ──→ [compare_musicxml.py] ──→ 정확도 리포트
                   ──→ [evaluate.py TER]      ──→ TER / note-only TER
```

### `pt_predict.py` 처리 흐름

```
1. PNG 로드 → 전처리 (autocrop → resize 1920px → CLAHE → bilateral)
2. detect_staffs() → 오선 탐지
3. extract_canvas_tiles() → 256×1280 타일 생성
4. SegNet + Encoder → context 벡터 [1, seq_len, 512]
5. Decoder greedy 디코딩 (EOS 또는 MAX_SEQ=608)
6. ID → 토큰 문자열 변환 (tokenizer.json 역방향)
```

### `tokens_to_musicxml.py` 처리 범위 (Round 1)

음표·쉼표·박자표·조표·도돌이표·세로줄만 변환. Round 2+ 기호는 경고 출력 후 무시.

### 일괄 평가

```bash
for f in data/test/*.png; do
    stem=$(basename $f .png)
    python omr/utils/pt_predict.py \
        --image $f --ckpt checkpoints/round1_best.pth \
        --tokenizer data/tokenizer.json \
        --out_xml output/${stem}_pred.musicxml
done
python omr/utils/evaluate.py \
    --test_dir data/test --pred_dir output \
    --tokenizer data/tokenizer.json
```

---

## 9. 정확도 미달 시 대응 방안

> 기준: val TER > 0.30 이거나 학습 곡선 미수렴.

### 진단 체크리스트

| 증상 | 의심 원인 |
|---|---|
| train loss 감소, val loss 발산 | 과적합 (데이터 부족) |
| train loss도 수렴 안 함 | 학습률 과다 또는 모델 구조 문제 |
| TER 낮은데 음높이만 틀림 | Decoder 토큰 혼동 |
| TER 낮은데 음가만 틀림 | 데이터 음가 분포 불균형 |
| 토큰 아예 없음 | StaffDetector / 전처리 실패 |
| 특정 조성에서만 틀림 | 조표 분포 불균형 |

### 원인별 처방

**A. 데이터 부족**
- 2,000장 → 5,000장 추가 생성
- `augment.py` 조기 도입 (밝기·노이즈 먼저, 원근 왜곡은 Round 3에서)
- Dropout 0.1 → 0.2

**B. 수렴 안 함**
- OneCycleLR max_lr 1e-3 → 5e-4
- SegNet 사전학습 epoch 50 → 80

**C. 음높이 혼동**
- Confusion Matrix로 혼동 쌍 파악
- 해당 음표 범위 데이터 비율 상향

**D. 음가 불균형**
- `_DUR_WEIGHTS` 조정 (온음표·2분음표 비율 소폭 상향)
- 희귀 음가 토큰 loss 가중치 ×2

**E. 오선 탐지 실패**
- `detect_staffs()` `MIN_UNIT` 파라미터 확인 (150 DPI에서 11px 적정 여부)
- PNG 샘플 직접 육안 확인

### 권장 처방 순서

```
1단계: 오선 탐지 정상 동작 확인 + 데이터 품질 수동 검증 5장
2단계: 데이터 2,000 → 5,000장 + 밝기·노이즈 증강 ×4
3단계: 학습률 낮추기 + SegNet 사전학습 epoch 증가
4단계: Dropout 상향 + Label Smoothing + 음가 가중치
```
