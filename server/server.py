"""
server/server.py — 프로젝트 루트에서 `python server/server.py`로 실행하는 실서비스용 웹 서버.

webpage/ 정적 파일(HTML/CSS/JS)을 서빙하면서 /api/status, /api/recognize를
제공한다. train/의 자체 학습 Round3(대보표) 체크포인트로 추론한다(자체 학습
모델이라 전시/배포에 그대로 쓸 수 있음 — omr_bridge는 2026-08-09 정리 때 삭제됨,
AGPL 기반 로컬 참고용 도구였고 더 이상 안 씀).

실행(프로젝트 루트에서):
    pip install -r server/requirements.txt
    python server/server.py
    (기본 0.0.0.0:8080 — 같은 네트워크의 폰에서 http://<이 PC의 LAN IP>:8080 으로 접속)
"""

import io
import sys
import tempfile
import time
import uuid
from pathlib import Path

import qrcode
import qrcode.image.svg as qrcode_svg
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

_ROOT = Path(__file__).resolve().parent.parent  # server/의 부모 = 저장소 루트
_TRAIN = _ROOT / "train"
sys.path.insert(0, str(_TRAIN))

from model import OmrSeq2Seq, infer_arch_from_state_dict  # noqa: E402
from dataset import load_tokenizer  # noqa: E402
from inference import run_image  # noqa: E402

from token_to_notes import tokens_to_score  # noqa: E402

CHECKPOINT = _TRAIN / "checkpoints" / "r15_cropfix_coordconv" / "seq2seq_best.pt"
TOKENIZER = _TRAIN / "tokenizer258.json"
WEBROOT = _ROOT / "webpage"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tok2id, _id2tok = load_tokenizer(str(TOKENIZER))
_ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
_arch = infer_arch_from_state_dict(_ckpt["model"])
_model = OmrSeq2Seq(vocab_size=len(_tok2id), **_arch).to(DEVICE)
_model.load_state_dict(_ckpt["model"])
_model.eval()
print(f"[server] 체크포인트 로드 완료: {CHECKPOINT.name} (device={DEVICE}, arch={_arch})")

app = FastAPI(title="악보의 재정의 — OMR 서버")


# 정적 파일(특히 js 모듈)이 브라우저에 그대로 캐시되면 서버를 업데이트해도 태블릿/폰이
# 예전 버전을 계속 쓰는 문제가 생긴다(전시 준비 중 실제로 겪음) — 매번 재검증하도록 강제.
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.add_middleware(NoCacheMiddleware)

# 전시 부스 QR "테이크아웃" 흐름용 — 변환 결과를 id로 잠깐 보관해서, 방문자 폰이 QR로
# 같은 결과를 곧바로 열어볼 수 있게 한다. 영속 저장 아님(서버 재시작 시 사라짐, 전시
# 세션 동안만 유지되면 충분).
_SCORES: dict[str, dict] = {}


@app.get("/api/status")
def status():
    return {"andromr": False, "custom": True}


@app.get("/api/score/{score_id}")
def get_score(score_id: str):
    result = _SCORES.get(score_id)
    if result is None:
        raise HTTPException(404, "해당 악보를 찾을 수 없습니다 (서버가 재시작되었을 수 있어요)")
    return JSONResponse(result)


@app.get("/api/qr")
def qr_code(data: str = Query(..., min_length=1, max_length=2000)):
    img = qrcode.make(data, image_factory=qrcode_svg.SvgPathImage, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return Response(content=buf.getvalue(), media_type="image/svg+xml")


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...), model: str = Query("custom")):
    if model != "custom":
        raise HTTPException(400, f"지원하지 않는 모델: {model}")

    suffix = Path(file.filename or "score.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / f"score{suffix}"
        img_path.write_bytes(await file.read())
        try:
            tokens = await run_in_threadpool(
                run_image, str(img_path), _model, _tok2id, _id2tok, DEVICE
            )
        except Exception as e:
            raise HTTPException(500, f"인식 실패: {e}") from e

    if not tokens:
        raise HTTPException(422, "오선을 찾지 못했습니다 — 다른 사진으로 다시 시도해주세요")

    score = tokens_to_score(tokens)
    now_ms = int(time.time() * 1000)
    result = {
        "id": f"n_{now_ms}_{uuid.uuid4().hex[:5]}",
        "title": Path(file.filename or "새 악보").stem,
        "tempo": 100,
        "timeSignature": score["timeSignature"],
        "createdAt": now_ms,
    }
    if score["bass"]:
        clefs = score.get("clefs", ["treble", "bass"])
        result["staves"] = [
            {"clef": clefs[0], "notes": score["treble"]},
            {"clef": clefs[1], "notes": score["bass"]},
        ]
        result["notes"] = score["treble"]
    else:
        result["notes"] = score["treble"]

    _SCORES[result["id"]] = result
    return JSONResponse(result)


# API 라우트를 먼저 등록해야 아래 정적 파일 마운트("/")가 /api/*를 가로채지 않는다.
app.mount("/", StaticFiles(directory=str(WEBROOT), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
