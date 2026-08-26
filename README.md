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
> 모바일 온디바이스 양자화**(`train/tools/export_tflite.py`, `train/QUANTIZATION_MOBILE.md`)
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
| `train/` | OMR 모델 추론 런타임 코드(PyTorch) + 체크포인트 + 모바일 양자화 작업(`tools/export_tflite.py`, `QUANTIZATION_MOBILE.md`) |
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
중이다(`train/tools/export_tflite.py`, 진행 로그는 `train/QUANTIZATION_MOBILE.md`).

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

### 벤치마크 리포트 (개발 PC CPU, held-out 10곡 평균 — `train/tools/benchmark.py`)

| 항목 | PyTorch (원본) | **TFLite Hybrid**(버킷팅 적용) | TFLite FP16(그래프 전체) |
|---|---|---|---|
| 모델 크기 | 184.6MB | **173.6MB**(작은/큰 버킷 안전판 둘 다 포함, 아래 참고) | 143.3MB |
| 추론 레이턴시(평균) | **3.0초** | **3.5초**(PyTorch 대비 **1.18배**) | 측정 불가(런타임 실패) |
| Peak Memory | 1.0GB | 2.2GB | — |
| 정확도(음표 기준) | 94.2% | **94.3%**(PyTorch와 사실상 동일) | 측정 불가 |

**버킷팅 트레이드오프(2026-08-26, EXPORT_NOTES.md §14)**: self-attention이 매 스텝
`cache_len`(고정 KV캐시 크기)만큼 계산하는 구조라, 캐시를 작은 크기(160)와 큰 크기(300,
안전판)로 나눠서 대부분의 곡을 작은 캐시로 빠르게 처리하게 했다. **속도는 확실히
좋아졌지만(1.3배 → 1.18배) 대신 크기가 116.1MB → 173.6MB로 50% 늘었다** — 작은/큰 버킷
디코더 그래프를 둘 다 담아야 하는데, 큰 버킷+전환용 그래프(~60MB)는 newage01~30 실측에서
단 한 번도 실제로 안 쓰였다(자세한 내용은 QUANTIZATION_MOBILE.md 참고). 실제 앱이라면
이 안전판을 온디맨드로 받게 하는 것도 고려할 만하다.

크기·속도·정확도 세 축 모두 FP32 TFLite보다 나은 배포 형태(Hybrid)를 확보했다. 여기까지
오는 과정에서 여러 번 막혔다가 그때마다 근본 원인을 찾아 풀었다 — 그 경로를 정직하게 남긴다.

**1차 시도(순수 FP16/Dynamic-Range, 실패)**: 처음엔 인코더+디코더 전체를 FP16 또는
dynamic-range(가중치 INT8, 활성값 FP32, 캘리브레이션 불필요)로 통째로 바꾸려 했다.
FP16은 CPU `BATCH_MATMUL` 커널이 fp16 입력을 지원하지 않아 런타임에서 실행이 안 됐고
(onnx2tf의 fp16 변환이 가중치만이 아니라 그래프 전체를 fp16으로 캐스팅 — GPU 델리게이트
전용 설계), dynamic-range는 인코더(CoordConv CNN)에서 압축은 되지만 출력이 수치적으로
발산했다(mean~3×10²⁸, std=inf, 속도도 300초+/장).

**2차 시도(팀원 제안 검증, 디코더 압축 문제 해결)**: 팀원이 "디코더의 attention이 3D 입력을
그대로 `nn.Linear`/`F.linear`에 먹여서 ONNX `MatMul`(→TFLite `BATCH_MATMUL`)로 export되니,
2D로 평탄화해서 `Gemm`(→`FULLY_CONNECTED`)으로 유도하라"고 제안했다 — 실제로 이 프로젝트의
디코더는 `in_proj_weight`를 수동 슬라이싱한 `F.linear` 구현이라 정확히 이 문제를 갖고
있었다. 적용해서 검증한 결과:
- ONNX op 분포가 `MatMul` 위주에서 `Gemm` 80개로 바뀜(남은 `MatMul` 33개는 실제
  attention score 계산 — 가중치가 없는 정상적인 부분)
- Dynamic-range 압축이 실제로 작동: 디코더 130.7MB→**35.3MB**, 일괄캐시 115.0MB→**29.9MB**
- 단, FP16은 op 인식이 개선돼도 여전히 실패(에러 지점만 `BATCH_MATMUL`→`FULLY_CONNECTED`로
  이동) — onnx2tf의 fp16 변환이 활성값까지 캐스팅하는 근본 구조 자체는 안 바뀌기 때문

**3차 시도(자체 발견, 새 버그 진단+수정)**: 압축은 됐는데 실제로 돌려보니 정확도가 0%로
나왔다. 디버깅해서 원인을 찾았다 — `torch.where(mask, a, b)`(self-attention 캐시 쓰기)와
`F.scaled_dot_product_attention(..., is_causal=True)`(causal masking) 둘 다 내부적으로
`Where`/`Select` 노드를 만드는데, onnx2tf의 dynamic-range 양자화가 이 노드의 상수
피연산자를 "가중치"로 잘못 인식해서 양자화 스케일을 `2.68×10³⁶` 같은 값으로 계산 →
그 순간부터 전부 NaN이 됐다. `torch.where`를 산술 블렌드(`mask*a + (1-mask)*b`)로,
`is_causal=True`를 명시적 덧셈 마스크(`attn_mask`)로 바꿔서 `Where` 노드 자체를 없애자
문제가 사라졌다(단일 스텝 logits이 FP32와 사실상 동일해짐, 10곡 검증 94.3%로 확인).

**4차 시도(cross-attention K,V 캐싱, 2026-08-26)**: 여기까지는 모델 크기 얘기였는데,
속도 쪽에는 별개의 원인이 남아있었다 — TFLite 디코더 step 그래프가 cross-attention(인코더
출력 대상) K,V를 캐싱하지 않고 **디코딩 스텝마다** `memory`(S=SEQ_LEN=320)에서 다시
projection하고 있었다(그래프를 간단히 유지하려던 단순화였음, production PyTorch는
`precompute_memory_kv()`로 이미지당 1회만 계산). 곡 하나가 보통 ~100스텝이니 레이어
8개 × 스텝 수만큼 반복되는 불필요한 연산이었다. `_MemoryKVWrapper`라는 별도 그래프
(`decoder_memkv_INT8.tflite`)로 분리해서 이미지당 1회만 호출하고, 그 결과를 디코더
step/일괄캐시 그래프에 입력으로 넘기는 방식으로 바꿨다(PyTorch와 동일한 설계로 이식).
결과: TFLite FP32가 10.6초→4.2초, Hybrid가 8.0초→3.4초로 두 배 넘게 빨라졌고, **PyTorch
대비 배율이 2.5~3.5배 → 1.3배로 줄었다.** 정확도는 수학적으로 동일한 연산 순서 변경이라
불변(94.3%, 10/10 크래시 없음).

**5차 시도(버킷팅, 2026-08-26)**: cross-attention 캐싱 이후에도 self-attention 자체가
`pos`와 무관하게 항상 고정 `cache_len`(=300) 크기로 계산되는 구조는 남아있었다(§3의 고정
shape 설계 대가). newage01~30(소나타류 제외 — 학습에만 쓰고 테스트에선 뺌) 실측 결과
29/30곡이 133 이하라, KV캐시를 **작은 버킷(160)/큰 버킷(300, 안전판)** 둘로 나눠 대부분의
곡을 작은 캐시로 처리하고 넘칠 때만 큰 캐시로 전환("리버킷")하게 했다. 리버킷 자체는
30곡 실측에서 한 번도 발동하지 않았지만, 강제로 트리거시켜 텐서 레벨로 정확성을 검증했다
(오차가 기존 `chunk_len=40` 일괄캐시 그래프와 동일 수준 — 새 버그 아님). 결과: **PyTorch
대비 배율이 1.3배 → 1.18배**로 더 좁혀졌다(정확도 94.3% 불변). 대신 작은/큰 버킷 그래프를
둘 다 담아야 해서 **크기가 116.1MB → 173.6MB로 50% 늘었다** — 속도와 크기를 맞바꾼
트레이드오프임을 정직하게 밝힌다.

**최종 구성**: 인코더는 CoordConv CNN에서 dynamic-range 시 발산이 확인돼 FP32로 고정,
디코더 계열(작은/큰 버킷 step, 일괄캐시, 리버킷)은 Gemm 유도+Where 회피를 반영한
dynamic-range로, cross-attention K,V는 이미지당 1회만 계산하도록 별도 그래프로 분리해서
export한다(`train/tools/export_tflite.py` 기본값 — 버킷팅·Hybrid 둘 다 이제 기본 동작).

**Peak Memory 수치의 측정 한계**: `tools/benchmark.py`가 여러 백엔드를 같은 프로세스 안에서
순차 측정하는데, `psutil`의 `peak_wset`은 프로세스 시작 이후 최고치를 누적 추적하는
값이라(감소하지 않음) 뒤에 실행되는 값이 앞서 실행된 값의 피크를 그대로 물려받을 수
있다 — 정확한 격리 측정을 하려면 백엔드별로 별도 프로세스에서 재야 한다(TODO).

**해석**: 이건 개발 PC의 범용 CPU(XNNPACK) 비교이고, 실제 모바일 기기는 NPU/GPU
델리게이트를 쓸 수 있어 다른 그림이 나올 가능성이 있다(③ 항목, 실기 측정 필요). 다만 이
Hybrid 구성 자체는 크기·속도·정확도 모두에서 실측으로 검증된 실질적 개선이라, ③ 실기
측정도 이 구성으로 진행하는 게 맞다.
