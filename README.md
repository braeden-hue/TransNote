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
- **TFLite export를 실제로 끝까지 돌려서 검증**: 인코더(54.5MB)+디코더(131MB) export 자체는
  성공했지만, "export 성공"과 "실제 동작"은 다르다는 걸 직접 확인했다 — 디코더가 2번째
  디코딩 스텝부터 reshape 에러로 깨졌다(`past_ids`를 매 스텝 growing 텐서로 리사이즈하는
  방식이 TFLite/XNNPACK과 안 맞음). 원인을 정확히 진단했고, 위 self-attention KV캐시(고정
  크기 버퍼 — 애초에 TFLite 호환을 염두에 두고 설계함)를 TFLite export에 그대로 적용하는
  게 다음 단계.
- **정확도 베이스라인도 재검증**: 처음 잰 82.1%가 held-out이 아니라 학습에 이미 쓰인
  데이터로 잰 수치였다는 걸 발견 — 실제 학습에 안 쓰인 셋(뉴에이지 10곡)으로 재측정하니
  94.2%. 앞으로 양자화 전/후 비교는 이 숫자를 기준으로 한다.
