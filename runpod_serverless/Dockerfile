# RunPod Serverless GPU 워커 이미지. 빌드 컨텍스트는 반드시 저장소 루트로 지정할 것
# (예: docker build -f runpod_serverless/Dockerfile -t transnote-omr .)
# RunPod 콘솔에서 "GitHub repo에서 빌드" 옵션을 쓰면 Dockerfile path만
# runpod_serverless/Dockerfile로 지정하면 됨(컨텍스트는 자동으로 repo root).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# train/ 최상위 활성 스크립트 전부 복사(선택적으로 일부만 골라오면 dataset.py의
# real_texture_augment.py 같은 sibling import가 누락될 위험이 있어 통째로 복사).
COPY train/*.py train/tokenizer258.json ./train/
COPY token_to_notes.py .
COPY runpod_serverless/handler.py .
COPY runpod_serverless/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 체크포인트(184MB)는 용량 문제로 git에 없음 — GitHub Release에서 빌드 시점에 받아온다.
# Release가 바뀌면(체크포인트 교체) 이 레이어부터 다시 빌드하면 됨.
RUN mkdir -p train/checkpoints/r15_cropfix_coordconv && \
    wget -q -O train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt \
    https://github.com/braeden-hue/TransNote/releases/download/checkpoint-r15/seq2seq_best.pt && \
    test -s train/checkpoints/r15_cropfix_coordconv/seq2seq_best.pt

CMD ["python3", "-u", "handler.py"]
