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

## Round별 학습 계획

### Round 1 (다음 실행)

```bash
python ml/scripts/train_round.py --round 1
# Phase 1 (SegNet):          epochs=50, batch=16 → ml/models/round1/segnet_best.pt
# Phase 2 (Encoder+Decoder): epochs=100, batch=8 → ml/models/round1/seq2seq_best.pt
# Phase 3 (End-to-End):      epochs=30, batch=4  → ml/models/round1/seq2seq_best.pt 갱신
```

완료 기준: val TER 안정적 감소 확인

### Round 2

```bash
python ml/scripts/train_round.py --round 2 \
    --prev-segnet  ml/models/round1/segnet_best.pt \
    --prev-seq2seq ml/models/round1/seq2seq_best.pt
```

- Round 1 체크포인트에서 fine-tuning (lr=3e-5)
- `load_checkpoint_with_vocab_expansion()` 사용 (vocab 1004→1012 확장)

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

## 정확도 미달 시 대응 방안

| 증상 | 의심 원인 | 처방 |
|---|---|---|
| train loss 감소, val loss 발산 | 과적합 | 데이터 2,000 → 5,000장 + 증강 |
| train loss도 수렴 안 함 | 학습률 과다 | max_lr 1e-3 → 5e-4 |
| TER 낮은데 음높이만 틀림 | Decoder 토큰 혼동 | Confusion Matrix → 해당 범위 데이터 비율 상향 |
| TER 낮은데 음가만 틀림 | 음가 분포 불균형 | `_DUR_WEIGHTS` 조정 + 희귀 음가 loss 가중치 ×2 |
| 토큰 아예 없음 | StaffDetector/전처리 실패 | `detect_staffs()` 파라미터 확인 |
