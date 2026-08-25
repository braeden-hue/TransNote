# TransNote

악보 이미지를 촬영/업로드하면 자체 학습한 OMR(광학 악보 인식) 모델이 음표를 인식해
**사용자 정의 표기법(커스텀 악보)**으로 변환해주는 서비스의 **추론 백엔드** 저장소.

악보를 처음 접하거나 오선보 읽기가 어려운 사람을 위해, 복잡한 조표·옥타브 규칙 없이
"세로 위치 = 음높이, 가로 폭 = 음길이, 테두리 색 = 박자 위치"만으로 읽을 수 있는 표기법을
자체 설계했다. 표기법 규칙 정의는 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote)의
`train/docs/music-notation-rule-designer.md` 참고.

> **2026-08-24**: 이 저장소가 함께 서빙하던 웹 프론트엔드(`webpage/`+`api/`+`server.py`,
> [trans-note.vercel.app](https://trans-note.vercel.app))를 제거했다. 실제 사용자용
> 웹 프론트엔드는 별도 프로젝트(myweb)로 이전됐고, 이 저장소의 RunPod Docker 이미지가
> 그쪽의 인식 요청을 그대로 처리한다. `trans-note.vercel.app`는 마지막 배포본이 남아있을
> 뿐 더 이상 갱신하지 않는다. 앞으로 이 저장소는 **RunPod Docker 이미지 파이프라인 유지 +
> 모바일 온디바이스 양자화**(`train/export_tflite.py`, `train/QUANTIZATION_MOBILE.md`)
> 위주로 다룬다.

---

## 아키텍처

```
(외부 프론트엔드, 예: myweb) → RunPod Serverless 엔드포인트
  → runpod_serverless/handler.py
      → train/inference.py: run_image()
          1. dataset.py: detect_staffs() — OpenCV 고전 알고리즘으로 오선 검출(학습 모델 아님)
          2. dataset.py: extract_staff_canvas()/extract_system_canvas() — 오선 크롭·정규화
          3. model.py: OmrSeq2Seq — CNN 인코더 + Transformer 디코더, autoregressive 토큰 생성
      → token_to_notes.py: tokens_to_score() — 토큰 시퀀스 → 커스텀 표기법 JSON
  ← JSON 응답
```

`.github/workflows/build-runpod-image.yml`이 `runpod_serverless/**`·`train/*.py`·
`train/tokenizer258.json`·`server/token_to_notes.py` 변경을 감지해 Docker 이미지를 자동
빌드하고 Docker Hub에 올린다. RunPod Serverless 엔드포인트는 그 이미지 주소를 "Custom
Image"로 지정해서 만든다(RunPod의 GitHub 연동 자동 빌드는 불안정해서 쓰지 않음).

## 체크포인트 (GitHub Release)

모델 체크포인트(약 184MB)는 용량 문제로 git 저장소에는 포함하지 않고 **GitHub Release**로
따로 배포한다. Docker 빌드 시 `runpod_serverless/Dockerfile`이 자동으로 받아온다.

| 파일 | 위치 | sha256 |
|---|---|---|
| `seq2seq_best.pt` | `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt` | `09c79377636b4e86dcbd4bc9e6744eaef93ad3aa7c0aa8933832eddb0fc0b9a9` |
| `tokenizer258.json` | `train/tokenizer258.json` | `fad052fedb7be8f35d241d7c8943c178b49ca336614ccecc41a57246aa518bcb` |

세그넷(SegNet) 체크포인트는 필요 없다 — 오선 검출은 학습된 모델이 아니라 OpenCV 고전
알고리즘(`detect_staffs()`)으로 수행한다. 모델 아키텍처(레이어 구성)는 위 체크포인트의
텐서 shape에서 자동으로 역산되므로 별도 설정 파일도 필요 없다.

## 디렉토리 구조

| 폴더 | 내용 |
|---|---|
| `runpod_serverless/` | RunPod Serverless GPU 추론 워커(Docker) — `.github/workflows`가 자동 빌드 |
| `train/` | OMR 모델 추론 런타임 코드(PyTorch) + 체크포인트 + 모바일 양자화 작업(`export_tflite.py`, `QUANTIZATION_MOBILE.md`) |
| `server/token_to_notes.py` | Docker 이미지가 복사해가는 런타임 의존 파일(토큰 시퀀스 → 악보 JSON 변환) |
| `realImage/` | 실사 촬영 이미지 데이터셋(로컬 전용, git 미포함) |

## 기술 스택

| 항목 | 내용 |
|---|---|
| 추론 서버 | RunPod Serverless(Docker, GPU) |
| OMR 모델 | PyTorch(CNN 인코더 + Transformer 디코더) — 학습 코드·정확도·개발 히스토리는 별도 저장소 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote) 참고 |
| 모바일 온디바이스 | ONNX export → TFLite(진행 중, 아래 참고) |

## 모바일 온디바이스 추론 (진행 중)

서버(RunPod GPU) 추론과 별개로, **같은 체크포인트를 모바일에서 직접 돌리는 작업**을 진행
중이다(`train/export_tflite.py`, 진행 로그는 `train/QUANTIZATION_MOBILE.md`).

- **자기회귀 디코더의 실제 병목을 찾아 고쳤다**: 서버 추론 경로도 cross-attention(인코더 출력)
  K,V만 캐싱하고 self-attention은 매 스텝 전체를 재계산하고 있었다(O(T³)). 고정 크기 버퍼
  기반 self-attention KV캐시를 구현해 O(T²)로 낮췄고, held-out 실사 10곡에서 토큰 시퀀스
  완전 일치(정확성 검증)·실측 속도 약 1.6~2배 개선을 확인한 뒤 프로덕션(RunPod)에 반영했다.
  더 낮출 수 있는지(O(T log T)급 sparse/linear attention 등)도 검토했으나, 이런 방법들은
  attention 계산 자체를 바꾸는 방식이라 재학습 없인 정확도가 깨진다는 걸 확인 — 재학습 없이
  유일하게 시도 가능했던 슬라이딩 윈도우 attention도 직접 실험해 정확도 하락으로 기각했다.
  지금 O(T²)가 재학습 없이 도달 가능한 합리적 한계.
- **TFLite export를 실제로 끝까지 돌려서 검증**: 인코더+디코더 export 자체는 성공했지만,
  "export 성공"과 "실제 동작"은 다르다는 걸 직접 확인했다 — 디코더가 2번째 디코딩 스텝부터
  reshape 에러로 깨졌다(`past_ids`를 매 스텝 growing 텐서로 리사이즈하는 방식이 TFLite/
  XNNPACK과 안 맞음). self-attention KV캐시를 고정 크기 버퍼(마스킹 기반, growing 텐서
  없음)로 재설계해서 해결 — 10곡 전부 크래시 없이 통과, PyTorch 대비 정확도 93.0%(94.2%
  대비 1.2%p 차이)까지 근접시켰다. 남은 차이는 `InlineTimeCorrector`(마디 중간 박자표
  재추정) 미지원 때문이었는데, 이것도 "고정 길이 일괄 재계산 그래프"(`decoder_bulk_INT8.
  tflite`)로 이식해서 해결 — attention이 매 레이어 모든 앞선 위치를 섞기 때문에 "한
  위치만 패치"는 수학적으로 불가능하다는 걸 확인하고, PyTorch와 동일하게 "교정 후 전체
  재계산" 방식을 택했다.
- **정확도 베이스라인도 재검증**: 처음 잰 82.1%가 held-out이 아니라 학습에 이미 쓰인
  데이터로 잰 수치였다는 걸 발견 — 실제 학습에 안 쓰인 셋(뉴에이지 10곡)으로 재측정하니
  94.2%. 앞으로 양자화 전/후 비교는 이 숫자를 기준으로 한다.

### 벤치마크 리포트 (개발 PC CPU, held-out 10곡 평균 — `train/benchmark.py`)

| 항목 | PyTorch (원본) | TFLite FP32 | TFLite FP16 |
|---|---|---|---|
| 모델 크기 | 184.6MB | 300.2MB(인코더+디코더+일괄캐시 3개 파일) | 150.3MB(FP32 대비 50%) |
| 추론 레이턴시(평균) | **5.3초** | 15.6초 | 측정 불가(런타임 실패) |
| Peak Memory | 1.0GB | 2.4GB | — |
| 정확도(음표 기준) | 94.2% | 93.0%(-1.2%p) | 측정 불가 |

**정직하게 밝히면, 이 desktop CPU 비교에서는 TFLite가 PyTorch보다 3배 느리고 메모리도
2.4배 더 씁니다.** "TFLite=항상 빠름"이라는 통념과 반대되는 결과라 원인을 짚었다:

1. **캐시 텐서 I/O 오버헤드** — 고정 크기 KV캐시가 레이어 8개 × K,V 각각
   `[1,8,300,64]`(약 4.7MB)라, 디코딩 스텝마다 `set_tensor`/`get_tensor`로 파이썬↔TFLite
   인터프리터 경계를 넘나들며 약 19MB(입력+출력)를 복사한다. 시퀀스 길이 ~100스텝이면
   한 이미지당 캐시 I/O만 약 1.9GB — PyTorch는 캐시가 프로세스 안에 계속 상주하는 텐서라
   이 복사 비용이 아예 없다.
2. **cross-attention K,V를 캐싱하지 않음** — TFLite export 그래프를 단순하게 유지하려고
   cross-attention(인코더 출력 대상) K,V를 매 스텝 `memory`에서 다시 projection하도록
   설계했다(PyTorch는 `precompute_memory_kv()`로 1회만 계산). 레이어 8개 × 스텝 수만큼
   반복되는 불필요한 연산.
3. **인터프리터 3개 동시 로드**(encoder/decoder/bulk) — TF 자체의 메모리 관리 오버헤드가
   더해짐.

**FP16은 파일 크기는 정확히 절반(150.3MB)으로 줄었지만, 이 CPU 인터프리터에서
`BATCH_MATMUL` 커널이 fp16 입력을 지원하지 않아 런타임에서 아예 실행이 안 됐다** —
TFLite의 fp16 양자화는 일반적으로 CPU에서 자동으로 fp32로 역양자화해서 계산하는데, 이
프로젝트처럼 attention을 수동으로 구현한 커스텀 그래프에서는 그 역양자화 삽입이 제대로
안 되는 것으로 보인다. 미해결 이슈로 남겨둔다.

**해석**: 이 결과가 "TFLite가 못 쓸 정도로 나쁘다"는 뜻은 아니다 — 이건 개발 PC의 범용
CPU(XNNPACK) 비교이고, 실제 모바일 기기는 NPU/GPU 델리게이트를 쓸 수 있어 다른 그림이
나올 가능성이 크다(③ 항목, 실기 측정 필요). 다만 지금 시점에서 "캐시 I/O·cross-attention
재계산 오버헤드"가 실측으로 확인된 구체적 개선 과제라는 것 자체가, 이 벤치마크가 실제로
뭔가를 증명했다는 뜻이다.
