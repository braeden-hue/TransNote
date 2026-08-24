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
| 모바일 양자화 | ONNX export → TFLite(진행 중, `train/QUANTIZATION_MOBILE.md` 참고) |
