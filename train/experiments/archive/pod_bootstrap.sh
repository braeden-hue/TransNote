#!/bin/bash
# round3train/pod_bootstrap.sh
#
# 포드 세션을 시작할 때마다(재접속/재시작 직후) 맨 먼저 실행. PODPLAN.md에 이미 기록된
# 문제: 포드 재시작마다 컨테이너 디스크가 초기화돼서 cv2(opencv-python-headless),
# music21, xvfb가 매번 사라지고 재설치가 필요했음(안 하면 ModuleNotFoundError로 학습이
# 즉시 죽음).
#
# 2026-07-30 venv 방식 폐기: /workspace가 네트워크 볼륨(MooseFS)이라 venv 생성(작은 파일
# 다수 생성)이 실측으로 몇 분 넘게 멈춘 것처럼 느려짐(실제로는 진행 중이었지만 사실상
# 못 쓸 수준). 대신 PODPLAN.md에 이미 검증됐던 원래 방식(시스템 파이썬에 직접
# --break-system-packages로 설치)으로 되돌림 -- 컨테이너 로컬 디스크라 재시작마다
# 재설치는 필요하지만, 설치 자체는 수십 초면 끝남.
#
# 사용법: bash round3train/pod_bootstrap.sh

APT_CACHE=/workspace/apt_cache

echo "[bootstrap] 시작..."

# ── 1) cv2/music21 (시스템 파이썬에 직접) ───────────────────────────────────
if python3 -c "import cv2, music21" 2>/dev/null; then
  echo "[bootstrap] cv2/music21 이미 설치돼있음"
else
  echo "[bootstrap] cv2/music21 설치 중..."
  pip install --break-system-packages --quiet opencv-python-headless music21
fi
python3 -c "
import cv2, music21
print(f'[bootstrap] cv2={cv2.__version__} music21={getattr(music21, \"__version__\", \"?\")}')"

# ── 2) xvfb (apt, 오프라인 캐시) ─────────────────────────────────────────────
if command -v Xvfb >/dev/null 2>&1; then
  echo "[bootstrap] xvfb 이미 설치돼있음"
elif [ -d "$APT_CACHE" ] && ls "$APT_CACHE"/*.deb >/dev/null 2>&1; then
  echo "[bootstrap] 캐시된 .deb로 xvfb 오프라인 설치 시도..."
  dpkg -i "$APT_CACHE"/*.deb 2>&1 | tail -5
  if ! command -v Xvfb >/dev/null 2>&1; then
    echo "[bootstrap] 오프라인 설치 실패(캐시 불완전 가능) -- 온라인 재시도"
    apt-get update -qq && apt-get install -y xvfb
  fi
else
  echo "[bootstrap] xvfb 캐시 없음 -- 온라인 설치하며 .deb 캐시 생성"
  mkdir -p "$APT_CACHE"
  apt-get update -qq
  apt-get install -y --download-only xvfb
  cp /var/cache/apt/archives/*.deb "$APT_CACHE"/ 2>/dev/null
  apt-get install -y xvfb
fi
if command -v Xvfb >/dev/null 2>&1; then
  echo "[bootstrap] xvfb OK"
else
  echo "[bootstrap] 경고: xvfb 설치 확인 실패 -- 렌더링 단계 진행 전 수동 확인 필요"
fi

# ── 3) MuseScore 4 (AppImage, /workspace에 캐시) ────────────────────────────
# AppImage가 컨테이너 환경(FUSE 없는 경우 많음)에서 바로 실행 안 되는 경우가 흔해서
# --appimage-extract로 풀어서 squashfs-root/AppRun을 직접 실행하는 방식을 씀(FUSE 불필요).
MS_DIR=/workspace/musescore
MS_BIN="$MS_DIR/squashfs-root/AppRun"
if [ -x "$MS_BIN" ]; then
  echo "[bootstrap] MuseScore 이미 설치돼있음 ($MS_BIN)"
else
  echo "[bootstrap] MuseScore 없음 -- GitHub 최신 릴리스 AppImage 다운로드"
  mkdir -p "$MS_DIR"
  apt-get install -y -qq libfuse2 libgl1 libegl1 libnss3 libxcomposite1 libxrandr2 \
    libxdamage1 libxkbcommon0 libasound2t64 libpipewire-0.3-0 fontconfig 2>&1 | tail -5
  DL_URL=$(curl -s https://api.github.com/repos/musescore/MuseScore/releases/latest \
    | grep -o '"browser_download_url": *"[^"]*x86_64\.AppImage"' | head -1 | cut -d'"' -f4)
  if [ -z "$DL_URL" ]; then
    echo "[bootstrap] 경고: 최신 릴리스 AppImage URL을 못 찾음 -- 수동 확인 필요"
  else
    echo "[bootstrap] 다운로드: $DL_URL"
    curl -L -o "$MS_DIR/MuseScore.AppImage" "$DL_URL"
    chmod +x "$MS_DIR/MuseScore.AppImage"
    ( cd "$MS_DIR" && ./MuseScore.AppImage --appimage-extract >/dev/null )
  fi
fi
if [ -x "$MS_BIN" ]; then
  xvfb-run -a "$MS_BIN" --version 2>&1 | tail -3
  echo "[bootstrap] MuseScore OK -- 경로: $MS_BIN"
else
  echo "[bootstrap] 경고: MuseScore 설치/추출 실패 -- 렌더링 전 수동 확인 필요"
fi

echo "[bootstrap] 완료"
echo "[bootstrap] MuseScore 경로: $MS_BIN"
