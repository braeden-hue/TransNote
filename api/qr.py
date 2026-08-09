"""api/qr.py — server.py의 /api/qr과 동일. 상태 저장이 필요 없는 순수 변환이라
RunPod 없이 Vercel 함수 안에서 바로 처리한다."""
import io

import qrcode
import qrcode.image.svg as qrcode_svg
from flask import Flask, Response, request

app = Flask(__name__)


@app.route("/api/qr", methods=["GET"])
def qr_code():
    data = request.args.get("data", "")
    if not data:
        return Response("data 쿼리 파라미터가 필요합니다", status=400)
    img = qrcode.make(data, image_factory=qrcode_svg.SvgPathImage, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), mimetype="image/svg+xml")
