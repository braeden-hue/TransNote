#!/bin/bash
ORT_LIB="/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/_deps/onnxruntime_prebuilt-src/lib"
EXE="/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/omr_flutter_test"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 출력 파일명 = 입력 파일명(확장자 제거) + .musicxml
INPUT_BASE="$(basename "${1%.*}")"
OUTPUT_XML="$SCRIPT_DIR/${INPUT_BASE}.musicxml"

# OMR 실행
env LD_LIBRARY_PATH="$ORT_LIB" "$EXE" "$1" "$OUTPUT_XML"

# 성공 시 MuseScore3로 자동 열기
if [ -f "$OUTPUT_XML" ]; then
    WIN_PATH="$(wslpath -w "$OUTPUT_XML")"
    powershell.exe -command "Start-Process 'shell:AppsFolder\MuseScoreBVBA.MuseScoreNotationSoftware_pz631wrhsw9tj!MuseScore3' -ArgumentList '$WIN_PATH'" 2>/dev/null
fi
