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


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        return jsonify({"error": "서버 설정 오류: RUNPOD_API_KEY/RUNPOD_ENDPOINT_ID 환경변수 미설정"}), 500

    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "file 필드가 없습니다"}), 400

    image_b64 = base64.b64encode(file.read()).decode("ascii")

    try:
        resp = requests.post(
            f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync",
            headers={
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"input": {"image_base64": image_b64, "filename": file.filename or "score.jpg"}},
            # RunPod 콜드스타트(컨테이너 기동+모델 로드)를 감안한 넉넉한 타임아웃.
            # 웜 상태 추론 자체는 GPU에서 수 초 이내.
            timeout=90,
        )
    except requests.RequestException as e:
        return jsonify({"error": f"RunPod 요청 실패: {e}"}), 502

    if resp.status_code != 200:
        return jsonify({"error": f"RunPod 오류 (HTTP {resp.status_code}): {resp.text[:300]}"}), 502

    data = resp.json()
    if data.get("status") != "COMPLETED":
        return jsonify({
            "error": f"RunPod 작업 실패: {data.get('status')} - {data.get('error', '')}"
        }), 502

    output = data.get("output") or {}
    if "error" in output:
        return jsonify({"error": output["error"]}), 422

    now_ms = int(time.time() * 1000)
    output["id"] = f"n_{now_ms}_{uuid.uuid4().hex[:5]}"
    output["createdAt"] = now_ms
    return jsonify(output)
