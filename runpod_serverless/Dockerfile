# RunPod Serverless GPU 워커 이미지. 빌드 컨텍스트는 반드시 저장소 루트로 지정할 것
# (예: docker build -f runpod_serverless/Dockerfile -t transnote-omr .)
#
# 배포 경로: RunPod의 "GitHub repo에서 자동 빌드" 기능이 반복적으로 오작동해서
# (linking flow 검증 실패, Dockerfile/handler 미인식) 이 방식은 쓰지 않는다.
# 대신 .github/workflows/build-runpod-image.yml이 이 Dockerfile을 빌드해
# Docker Hub(kyutae424/transnote-omr:latest)에 올리고, RunPod Serverless
# 엔드포인트는 "Custom Image" 소스로 그 이미지 주소를 직접 지정해서 만든다.
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
