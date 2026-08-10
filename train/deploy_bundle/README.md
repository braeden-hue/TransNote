# RunPod / 배포용 체크포인트 번들

`server.py`가 실제로 로드하는 두 파일만 모아둔 폴더. 그대로 RunPod(또는 다른 서버)의
`train/checkpoints/r15_cropfix_coordconv/`, `train/tokenizer258.json` 경로에 옮기면 된다
(경로를 그대로 재현하고 싶다면 이 폴더 대신 원본 두 경로를 직접 scp해도 동일).

## 포함 파일

| 파일 | 원본 경로 | 크기 | sha256 |
|---|---|---|---|
| `seq2seq_best.pt` | `train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt` | 193,597,556 bytes | `09c79377636b4e86dcbd4bc9e6744eaef93ad3aa7c0aa8933832eddb0fc0b9a` |
| `tokenizer258.json` | `train/tokenizer258.json` | 5,216 bytes | `fad052fedb7be8f35d241d7c8943c178b49ca336614ccecc41a57246aa518b` |

r15 채택 근거·다른 후보(r16/r17) 기각 이유는 [`../docs/TRAINING_REPORT.md`](../docs/TRAINING_REPORT.md)
참고.

## segnet 체크포인트는 왜 없나

`server.py` → `inference.py`의 실제 추론 경로는 학습된 SegNet을 전혀 쓰지 않는다. 오선(악보 줄)
검출은 `dataset.py`의 `detect_staffs()`(OpenCV 고전 알고리즘, 학습 불필요)로만 수행되고,
`inference.py`에는 segnet 관련 코드가 아예 없다(`grep -r segnet train/inference.py` → 0건).
SegNet은 이제 삭제된 옛 Flutter/C++ 모바일 엔진(`ml/omr/engine/`) 전용으로 설계됐던 자산이라
`train/checkpoints_legacy/segnet_best.pt`에만 남아있고, 현재 웹 서버 파이프라인과는 무관하다.
따라서 이 번들에는 포함하지 않았다 — RunPod에 굳이 올릴 필요 없음.

## 아키텍처 설정 파일이 왜 없나

모델 구조(in_ch/backbone 깊이/pool_h 등)는 `model.py`의 `infer_arch_from_state_dict()`가
체크포인트의 state_dict 텐서 shape만 보고 자동으로 역산한다(`server.py`가 로드 시 항상 이
함수를 거침) — 별도 아키텍처 JSON/config 파일이 필요 없다.

## RunPod로 전송 (예시)

```bash
scp -i ~/.ssh/runpod_auto -P <PORT> train/deploy_bundle/seq2seq_best.pt \
    root@<HOST>:/workspace/models/r15_cropfix_coordconv/seq2seq_best.pt
scp -i ~/.ssh/runpod_auto -P <PORT> train/deploy_bundle/tokenizer258.json \
    root@<HOST>:/workspace/tokenizer258.json
```

(POD_TRAINING_CHECKLIST.md 기준 이 프록시에서 `scp`가 거부되는 세션도 있었음 — 그 경우
`PODPLAN.md`류 base64 청크 전송 방식 대신, **push(로컬→pod) 방향은 항상 안정적이었으므로**
`ssh ... "cat > path" < 로컬파일` 형태의 stdin 리다이렉트로 대체 가능.)

## 이 폴더는 git에 커밋되지 않음

`.gitignore`의 `train/**/*.pt` 규칙에 걸려 `seq2seq_best.pt`는 자동 제외된다. 원본
`tokenizer258.json`은 `train/tokenizer258.json` 경로에 이미 git 추적 중이므로, 이 복사본이
없어도 소스는 보존된다 — 이 폴더 전체는 순수히 "전송 편의용 스테이징 디렉토리"다.
