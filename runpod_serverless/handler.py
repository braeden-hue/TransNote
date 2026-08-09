"""
runpod_serverless/handler.py — RunPod Serverless GPU 워커. `server.py`의 /api/recognize와
동일한 인식 로직(train/inference.py: run_image)을 그대로 쓰되, 웹서버가 아니라 RunPod의
서버리스 이벤트 핸들러 형태로 감싼 것.

역할 분담(Vercel + RunPod 하이브리드 배포):
  Vercel(api/recognize.py, 정적 webpage/) — 참가자 브라우저가 직접 접속하는 얇은 프록시.
  RunPod Serverless(이 파일) — GPU 추론만 전담, 요청 없을 땐 0원(콜드 스타트 감수).

입력(event["input"]): {"image_base64": "<base64 문자열>", "filename": "score.jpg"}
출력: server.py의 /api/recognize 응답과 동일한 JSON 스키마
  {"title", "tempo", "timeSignature", "notes"[, "staves"]} 또는 {"error": "..."}

모델/토크나이저 로드는 모듈 최상단에서 1회만 수행 — RunPod 워커가 warm 상태를 유지하는 동안
(콜드 스타트 이후 연속 요청들) 재사용된다. 워커 시작 시점(콜드 스타트)에만 로드 비용이 든다.
"""
import base64
import sys
import tempfile
from pathlib import Path

import runpod
import torch

_ROOT = Path(__file__).resolve().parent  # Dockerfile이 handler.py를 WORKDIR(/app) 바로 밑에
# COPY하고 train/ 하위 파일들도 /app/train/에 넣으므로, 컨테이너 안에서는 handler.py의
# 부모 디렉토리가 곧 /app이다(한 단계만 올라가야 함 — 두 단계 올리면 컨테이너 루트로
# 잘못 빠져서 체크포인트를 못 찾고 매번 시작부터 죽는 버그였음, 2026-08-10 발견).
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "train"))

from model import OmrSeq2Seq, infer_arch_from_state_dict  # noqa: E402
from dataset import load_tokenizer  # noqa: E402
from inference import run_image  # noqa: E402
from token_to_notes import tokens_to_score  # noqa: E402

CHECKPOINT = _ROOT / "train" / "checkpoints" / "r15_cropfix_coordconv" / "seq2seq_best.pt"
TOKENIZER = _ROOT / "train" / "tokenizer258.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tok2id, _id2tok = load_tokenizer(str(TOKENIZER))
_ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
_arch = infer_arch_from_state_dict(_ckpt["model"])
_model = OmrSeq2Seq(vocab_size=len(_tok2id), **_arch).to(DEVICE)
_model.load_state_dict(_ckpt["model"])
_model.eval()
print(f"[runpod handler] 체크포인트 로드 완료: {CHECKPOINT.name} (device={DEVICE}, arch={_arch})")


def handler(event):
    inp = event.get("input") or {}
    b64 = inp.get("image_base64")
    if not b64:
        return {"error": "image_base64 필드가 없습니다"}

    filename = inp.get("filename") or "score.jpg"
    suffix = Path(filename).suffix or ".jpg"

    try:
        img_bytes = base64.b64decode(b64)
    except Exception as e:
        return {"error": f"base64 디코딩 실패: {e}"}

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / f"score{suffix}"
        img_path.write_bytes(img_bytes)
        try:
            tokens = run_image(str(img_path), _model, _tok2id, _id2tok, DEVICE)
        except Exception as e:
            return {"error": f"인식 실패: {e}"}

    if not tokens:
        return {"error": "오선을 찾지 못했습니다 — 다른 사진으로 다시 시도해주세요"}

    score = tokens_to_score(tokens)
    result = {
        "title": Path(filename).stem,
        "tempo": 100,
        "timeSignature": score["timeSignature"],
    }
    if score["bass"]:
        result["staves"] = [
            {"clef": "treble", "notes": score["treble"]},
            {"clef": "bass", "notes": score["bass"]},
        ]
        result["notes"] = score["treble"]
    else:
        result["notes"] = score["treble"]

    return result


runpod.serverless.start({"handler": handler})
