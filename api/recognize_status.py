"""
api/recognize_status.py — api/recognize.py가 제출한 RunPod job의 진행 상태를 1회 조회하는
가벼운 프록시. webpage/js/recognize.js가 이 엔드포인트를 짧은 간격(1초)으로 반복 호출해서
완료를 기다린다. 매 호출은 RunPod /status를 한 번 찔러보고 바로 반환하므로 몇백ms 안에
끝나고, 아무리 콜드스타트가 오래 걸려도 이 함수의 실행시간/Vercel 타임아웃과는 무관하다.
"""
from flask import Flask, jsonify, request

import requests

from _runpod_client import auth_headers, base_url, finalize_output, is_configured

app = Flask(__name__)


@app.route("/api/recognize/status", methods=["GET"])
def recognize_status():
    if not is_configured():
        return jsonify({"error": "서버 설정 오류: RUNPOD_API_KEY/RUNPOD_ENDPOINT_ID 환경변수 미설정"}), 500

    job_id = request.args.get("id")
    if not job_id:
        return jsonify({"error": "id 쿼리 파라미터가 없습니다"}), 400

    try:
        resp = requests.get(f"{base_url()}/status/{job_id}", headers=auth_headers(), timeout=10)
    except requests.RequestException as e:
        return jsonify({"error": f"RunPod 요청 실패: {e}"}), 502

    data = resp.json()
    status = data.get("status")

    if status == "COMPLETED":
        output = data.get("output") or {}
        if "error" in output:
            return jsonify({"status": "FAILED", "error": output["error"]})
        return jsonify({"status": "COMPLETED", "result": finalize_output(output, data)})

    if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        return jsonify({"status": "FAILED", "error": f"{status} - {data.get('error', '')}"})

    # IN_QUEUE / IN_PROGRESS — 아직 진행 중, 브라우저가 다시 물어볼 것
    return jsonify({"status": status or "UNKNOWN", "delayTimeMs": data.get("delayTime")})
