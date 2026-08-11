"""
api/recognize.py — Vercel 서버리스 함수. GPU 추론은 전혀 하지 않는다(torch도 안 씀,
그래서 함수 배포 크기가 작고 콜드스타트가 빠르다) — 업로드된 이미지를 base64로 인코딩해
RunPod Serverless 엔드포인트(runpod_serverless/handler.py)에 작업을 "제출"만 하고,
job_id를 즉시 브라우저에 돌려준다.

webpage/js/app.js는 이 경로를 원래 server.py(FastAPI)와 동일한 상대경로/스키마로 호출하므로
프론트엔드 코드 변경이 필요 없다.

필요한 Vercel 환경변수(Project Settings → Environment Variables):
  RUNPOD_API_KEY      — RunPod 계정 API 키
  RUNPOD_ENDPOINT_ID  — RunPod Serverless 엔드포인트 ID

중요 — 이 함수 안에서 완료까지 기다리지 않는 이유:
예전엔 여기서 /run 제출 후 /status를 최대 54초(POLL_BUDGET_SEC)까지 반복 조회하는 블로킹
폴링을 했다. 그런데 RunPod 콜드스타트(7GB대 이미지 pull + 모델 로드)가 그보다 자주 더
오래 걸려(90초+가 드물지 않음) 실제로는 콜드스타트마다 "아직 준비 중이에요" 타임아웃이
자주 발생했다. 그래서 이 함수는 /run 제출까지만 하고(1초 이내로 끝남) 바로 반환하며,
완료 여부 확인은 api/recognize_status.py를 브라우저(webpage/js/recognize.js)가 직접
반복 호출해서 처리한다.

api/_runpod_client.py 같은 sibling 모듈로 이 로직을 recognize_status.py와 공유하는 구조를
처음 시도했으나, 배포 후 실제로 HTTP 500이 발생했다(로컬 py_compile은 통과했지만 Vercel의
legacy @vercel/python 빌더가 같은 api/ 디렉터리의 sibling 모듈을 항상 함수 번들에 포함해
준다는 보장이 없었던 것으로 보임 — api/status.py, api/qr.py 등 기존에 확실히 동작하던
함수들도 전부 다른 파일을 import하지 않는 완전 독립 파일이었다). 그래서 각 함수가 완전히
독립적으로 동작하도록 되돌렸다 — 공용 로직이 두 파일에 조금 중복되지만 배포 안정성이 우선.
"""
import base64
import os

import requests
from flask import Flask, jsonify, request

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

app = Flask(__name__)


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
        # 디버그용 — 실제로 어떤 URL/ID로 요청을 보냈는지까지 에러 메시지에 그대로 노출
        # (ENDPOINT_ID는 API 키와 달리 비밀값이 아니라 노출돼도 안전 - 공백/오타 확인용).
        return jsonify({
            "error": (
                f"RunPod 오류 (HTTP {resp.status_code}) url={base_url}/run "
                f"id_repr={RUNPOD_ENDPOINT_ID!r} body={resp.text[:200]}"
            )
        }), 502

    job_id = resp.json().get("id")
    if not job_id:
        return jsonify({"error": "RunPod 응답에 작업 id가 없습니다"}), 502

    return jsonify({"jobId": job_id})
