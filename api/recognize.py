"""
api/recognize.py — Vercel 서버리스 함수. GPU 추론은 전혀 하지 않는다(torch도 안 씀,
그래서 함수 배포 크기가 작고 콜드스타트가 빠르다) — 업로드된 이미지를 base64로 인코딩해
RunPod Serverless 엔드포인트(runpod_serverless/handler.py)로 넘기고, 결과 JSON을 그대로
브라우저에 돌려주는 얇은 프록시 역할만 한다.

webpage/js/app.js는 이 경로를 원래 server.py(FastAPI)와 동일한 상대경로/스키마로 호출하므로
프론트엔드 코드 변경이 필요 없다.

필요한 Vercel 환경변수(Project Settings → Environment Variables):
  RUNPOD_API_KEY      — RunPod 계정 API 키
  RUNPOD_ENDPOINT_ID  — RunPod Serverless 엔드포인트 ID

중요 — /runsync 대신 /run + /status 폴링을 쓰는 이유:
RunPod의 /runsync는 서버 쪽에서 최대 90초만 붙잡고 있다가 그 안에 안 끝나면 결과 대신
IN_PROGRESS 상태를 그냥 돌려준다(클라이언트 쪽 타임아웃을 아무리 늘려도 이 90초 자체는
못 늘림). 콜드스타트(7GB대 이미지 pull + 모델 로드)가 90초를 넘기는 경우가 흔해서,
비동기 /run으로 작업만 시작시키고 /status를 짧은 간격으로 반복 조회하는 방식으로 바꿨다.
이 함수 자체의 실행 시간은 vercel.json의 maxDuration(60초)에 맞춰 POLL_BUDGET_SEC를 잡는다.
"""
import base64
import os
import time
import uuid

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

POLL_BUDGET_SEC = 54  # vercel.json maxDuration(60초)보다 여유 있게 짧게
POLL_INTERVAL_SEC = 2


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        return jsonify({"error": "서버 설정 오류: RUNPOD_API_KEY/RUNPOD_ENDPOINT_ID 환경변수 미설정"}), 500

    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "file 필드가 없습니다"}), 400

    image_b64 = base64.b64encode(file.read()).decode("ascii")
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    base_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

    try:
        resp = requests.post(
            f"{base_url}/run",
            headers=headers,
            json={"input": {"image_base64": image_b64, "filename": file.filename or "score.jpg"}},
            timeout=15,
        )
    except requests.RequestException as e:
        return jsonify({"error": f"RunPod 요청 실패: {e}"}), 502

    if resp.status_code != 200:
        return jsonify({"error": f"RunPod 오류 (HTTP {resp.status_code}): {resp.text[:300]}"}), 502

    job_id = resp.json().get("id")
    if not job_id:
        return jsonify({"error": "RunPod 응답에 작업 id가 없습니다"}), 502

    deadline = time.monotonic() + POLL_BUDGET_SEC
    data = None
    while time.monotonic() < deadline:
        try:
            status_resp = requests.get(f"{base_url}/status/{job_id}", headers=headers, timeout=10)
        except requests.RequestException:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        data = status_resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            break
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            return jsonify({"error": f"RunPod 작업 실패: {status} - {data.get('error', '')}"}), 502
        time.sleep(POLL_INTERVAL_SEC)  # IN_QUEUE / IN_PROGRESS — 계속 대기
    else:
        # 콜드스타트가 유난히 오래 걸리는 첫 요청 등 — 명확한 재시도 안내로 응답
        return jsonify({
            "error": "인식 서버가 아직 준비 중이에요(첫 요청은 최대 1분 이상 걸릴 수 있어요) — 잠시 후 다시 시도해주세요"
        }), 504

    if not data or data.get("status") != "COMPLETED":
        return jsonify({"error": f"RunPod 작업 실패: {data.get('status') if data else 'unknown'}"}), 502

    output = data.get("output") or {}
    if "error" in output:
        return jsonify({"error": output["error"]}), 422

    now_ms = int(time.time() * 1000)
    output["id"] = f"n_{now_ms}_{uuid.uuid4().hex[:5]}"
    output["createdAt"] = now_ms
    return jsonify(output)
