"""api/_runpod_client.py — RunPod Serverless REST 호출 공통 로직.
recognize.py(작업 제출)와 recognize_status.py(진행 상태 조회)가 이 모듈을 함께 import해서
쓴다(Vercel Python 함수는 같은 api/ 디렉터리의 sibling 모듈을 그대로 import할 수 있음).
엔드포인트 URL 조립/인증 헤더/완료 응답 후처리(id·createdAt·_timing 부여)를 한 곳에
모아 두 함수 사이에서 중복되거나 어긋나지 않게 한다.
"""
import os
import time
import uuid

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")


def is_configured():
    return bool(RUNPOD_API_KEY) and bool(RUNPOD_ENDPOINT_ID)


def base_url():
    return f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"


def auth_headers():
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}


def finalize_output(output, status_data):
    """COMPLETED 상태의 RunPod job에서, output에 id/createdAt/진단용 타이밍을 채워 반환."""
    now_ms = int(time.time() * 1000)
    output["id"] = f"n_{now_ms}_{uuid.uuid4().hex[:5]}"
    output["createdAt"] = now_ms
    output["_timing"] = {
        "delayTimeMs": status_data.get("delayTime"),
        "executionTimeMs": status_data.get("executionTime"),
    }
    return output
