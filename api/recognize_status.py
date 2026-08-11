"""
api/recognize_status.py — api/recognize.py가 제출한 RunPod job의 진행 상태를 1회 조회하는
가벼운 프록시. webpage/js/recognize.js가 이 엔드포인트를 짧은 간격(1초)으로 반복 호출해서
완료를 기다린다. 매 호출은 RunPod /status를 한 번 찔러보고 바로 반환하므로 몇백ms 안에
끝나고, 아무리 콜드스타트가 오래 걸려도 이 함수의 실행시간/Vercel 타임아웃과는 무관하다.

api/recognize.py와 인증 헤더 조립 등 로직이 일부 겹치지만 의도적으로 중복시켰다 — sibling
모듈(api/_runpod_client.py)로 공유하는 구조를 먼저 시도했다가 Vercel 배포에서 HTTP 500이
났던 적이 있어(recognize.py 주석 참고), 각 함수를 완전히 독립시키는 쪽이 안전했다.
"""
import os
import time
import uuid

import requests
from flask import Flask, jsonify, request

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

app = Flask(__name__)


@app.route("/api/recognize/status", methods=["GET"])
def recognize_status():
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        return jsonify({"error": "서버 설정 오류: RUNPOD_API_KEY/RUNPOD_ENDPOINT_ID 환경변수 미설정"}), 500

    job_id = request.args.get("id")
    if not job_id:
        return jsonify({"error": "id 쿼리 파라미터가 없습니다"}), 400

    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    base_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

    try:
        resp = requests.get(f"{base_url}/status/{job_id}", headers=headers, timeout=10)
    except requests.RequestException as e:
        return jsonify({"error": f"RunPod 요청 실패: {e}"}), 502

    data = resp.json()
    status = data.get("status")

    if status == "COMPLETED":
        output = data.get("output") or {}
        if "error" in output:
            return jsonify({"status": "FAILED", "error": output["error"]})
        now_ms = int(time.time() * 1000)
        output["id"] = f"n_{now_ms}_{uuid.uuid4().hex[:5]}"
        output["createdAt"] = now_ms
        output["_timing"] = {"delayTimeMs": data.get("delayTime"), "executionTimeMs": data.get("executionTime")}
        return jsonify({"status": "COMPLETED", "result": output})

    if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        return jsonify({"status": "FAILED", "error": f"{status} - {data.get('error', '')}"})

    # IN_QUEUE / IN_PROGRESS — 아직 진행 중, 브라우저가 다시 물어볼 것
    return jsonify({"status": status or "UNKNOWN", "delayTimeMs": data.get("delayTime")})
