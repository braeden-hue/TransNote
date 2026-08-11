# TransNote

악보 이미지를 촬영/업로드하면 자체 학습한 OMR(광학 악보 인식) 모델이 음표를 인식해
**사용자 정의 표기법(커스텀 악보)**으로 변환해주는 웹 앱. 변환된 악보로 화면 위 가상 피아노
또는 연결된 전자 피아노(MIDI)로 바로 연주 연습까지 이어진다.

악보를 처음 접하거나 오선보 읽기가 어려운 사람을 위해, 복잡한 조표·옥타브 규칙 없이
"세로 위치 = 음높이, 가로 폭 = 음길이, 테두리 색 = 박자 위치"만으로 읽을 수 있는 표기법을
자체 설계했다.

**🔗 데모: [trans-note.vercel.app](https://trans-note.vercel.app)**

<a href="https://trans-note.vercel.app"><img src="https://trans-note.vercel.app/api/qr?data=https%3A%2F%2Ftrans-note.vercel.app" alt="데모 QR 코드" width="140" /></a>

QR을 스캔하면 위와 같은 데모 URL로 바로 연결된다(`/api/qr`가 실시간으로 생성 — 정적 이미지
아님, URL이 바뀌어도 갱신 불필요).

---

## 데모 흐름

1. **랜딩 화면** — 3개의 이미지 위 핫스팟 버튼(튜토리얼 / 체험하기 / 프로젝트 소개)
2. **튜토리얼** — 규칙 0(12음 건반 라벨)부터 규칙 1(음높이=세로 위치, 옥타브별 색상)·규칙 2(음길이=폭)·규칙 3(화음)까지 단계별 학습 + 테스트 5문항, 태블릿 가로 화면 기준 스크롤 없이 한 화면에 맞춤(CSS zoom 기반 auto-fit)
3. **체험하기** — 샘플 3곡 중 선택 또는 카메라로 직접 촬영 → 커스텀 악보로 변환 → 오른손만/양손 모드 선택 → 마디 단위로 자동 진행되는 연주 연습(화면 가상 피아노 또는 Web MIDI로 연결한 실물 전자 피아노) → 점수 및 리더보드

---

## 아키텍처

```
webpage/(카메라 촬영·업로드) → POST /api/recognize
  → train/inference.py: run_image()
      1. dataset.py: detect_staffs() — OpenCV 고전 알고리즘으로 오선 검출(학습 모델 아님)
      2. dataset.py: extract_staff_canvas()/extract_system_canvas() — 오선 크롭·정규화
      3. model.py: OmrSeq2Seq — CNN 인코더 + Transformer 디코더, autoregressive 토큰 생성
  → token_to_notes.py: tokens_to_score() — 토큰 시퀀스 → 커스텀 표기법 JSON
  → webpage/js/notation.js — SVG로 커스텀 악보 렌더링
```

`server.py`(FastAPI) 하나가 정적 웹앱(`webpage/`)을 서빙하면서 동시에 위 인식 API도 제공한다 —
별도 백엔드/프론트엔드 레포 분리 없음.

## 실행 (체크포인트 다운로드 필요)

모델 체크포인트(약 184MB)는 용량 문제로 git 저장소에는 포함하지 않고 **GitHub Release**로
따로 배포한다. 아래 2개 파일을 [Releases 페이지](https://github.com/braeden-hue/TransNote/releases/tag/checkpoint-r15)에서
받아 지정된 경로에 넣으면 된다.

| 파일 | 받는 위치 | sha256 |
|---|---|---|
| `seq2seq_best.pt` | `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt` | `09c79377636b4e86dcbd4bc9e6744eaef93ad3aa7c0aa8933832eddb0fc0b9a9` |
| `tokenizer258.json` | `train/tokenizer258.json` | `fad052fedb7be8f35d241d7c8943c178b49ca336614ccecc41a57246aa518bcb` |

```bash
pip install -r server/requirements.txt
python server/server.py
# 기본 0.0.0.0:8080 — 같은 네트워크의 폰/태블릿에서 http://<이 PC의 LAN IP>:8080 으로 접속
```

세그넷(SegNet) 체크포인트는 필요 없다 — 오선 검출은 학습된 모델이 아니라 OpenCV 고전
알고리즘(`detect_staffs()`)으로 수행한다. 모델 아키텍처(레이어 구성)는 위 체크포인트의
텐서 shape에서 자동으로 역산되므로 별도 설정 파일도 필요 없다.

Web MIDI API(전자 피아노 연동)와 카메라(`getUserMedia`)는 보안 컨텍스트(https:// 또는
localhost)에서만 동작한다 — LAN IP로 `http://`만 접속하면 브라우저가 차단한다.

## 디렉토리 구조

| 폴더 | 내용 |
|---|---|
| `server/` | 로컬/LAN 실행용 FastAPI 서버(`server.py`) + `token_to_notes.py` + `requirements.txt` |
| `webpage/` | 정적 웹앱(HTML/CSS/JS), PWA(manifest.json) |
| `api/` | Vercel 서버리스 함수(얇은 프록시) — 실제 추론은 `runpod_serverless/`가 전담 |
| `runpod_serverless/` | RunPod Serverless GPU 추론 워커(Docker) |
| `train/` | OMR 모델 학습 파이프라인(PyTorch) + 체크포인트 |
| `test/` | 학습된 모델 평가/진단 스크립트 |
| `realImage/` | 실사 촬영 이미지 데이터셋(로컬 전용, git 미포함) |

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| 서버 | FastAPI(`server.py`), 정적 파일 서빙 + `/api/recognize`·`/api/status`·`/api/score`·`/api/qr` |
| 프론트엔드 | 바닐라 JS(`webpage/js/`), SVG 커스텀 표기법 렌더링, Web Audio(연주 합성), Web MIDI(전자 피아노 연동) |
| OMR 모델 | PyTorch(CNN 인코더 + Transformer 디코더) — 학습 코드·정확도·개발 히스토리는 별도 저장소 [Model_TransNote](https://github.com/braeden-hue/Model_TransNote) 참고 |
| DB | Firebase(닉네임/점수 저장, 무료 티어) |
