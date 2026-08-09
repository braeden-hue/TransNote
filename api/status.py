"""api/status.py — RunPod 엔드포인트 환경변수가 설정돼 있는지만 알려주는 헬스체크."""
import os

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/api/status", methods=["GET"])
def status():
    configured = bool(os.environ.get("RUNPOD_API_KEY")) and bool(os.environ.get("RUNPOD_ENDPOINT_ID"))
    return jsonify({"andromr": False, "custom": configured})
