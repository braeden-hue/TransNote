#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_flutter_desktop.sh  —  musicscore_flutter 데스크탑 테스트 빌드 + 실행
#
# Flutter 앱의 nativeProcessImageBytes 와 동일한 C++ 경로를 WSL에서 검증
#
# 사용법:
#   cd /mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop
#   bash build_flutter_desktop.sh [build|run <image> [output.xml]|clean|help]
#
# 의존성 (최초 1회):
#   sudo apt update && sudo apt install -y cmake ninja-build libopencv-dev g++ git
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$HOME/flutter_desktop_build"
ASSETS_DIR="/mnt/c/Users/kyutae/AndroidStudioProjects/MusicScore/app/src/main/assets"
EXE="$BUILD_DIR/omr_flutter_test"

# MusicScore 빌드에서 이미 다운로드된 TFLite/ONNX 소스 재사용 (시간 절약)
PREV_BUILD="$HOME/omr_desktop_build"

green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }

check_deps() {
    local missing=()
    for cmd in cmake ninja g++ git; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
        missing+=("libopencv-dev")
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        red "[error] 설치 필요: ${missing[*]}"
        yellow "  → sudo apt update && sudo apt install -y cmake ninja-build libopencv-dev g++ git"
        exit 1
    fi
}

cmd_build() {
    check_deps

    if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        yellow "[cmake] Configure 시작..."
        mkdir -p "$BUILD_DIR"

        local extra_args=()

        # omr_desktop_build에 TFLite 소스가 있으면 재사용 (재다운로드 생략)
        local tfsrc="$PREV_BUILD/_deps/tensorflow_src-src"
        if [[ -d "$tfsrc" ]]; then
            yellow "[info] 기존 TFLite 소스 재사용: $tfsrc"
            extra_args+=("-DFETCHCONTENT_SOURCE_DIR_TENSORFLOW_SRC=$tfsrc")
        fi

        # ONNX Runtime prebuilt 재사용
        local ort_src="$PREV_BUILD/_deps/onnxruntime_prebuilt-src"
        if [[ -d "$ort_src" ]]; then
            yellow "[info] 기존 ONNX Runtime 재사용: $ort_src"
            extra_args+=("-DFETCHCONTENT_SOURCE_DIR_ONNXRUNTIME_PREBUILT=$ort_src")
        fi

        cmake "$SCRIPT_DIR" \
            -B "$BUILD_DIR" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_FLAGS="-O2" \
            "${extra_args[@]}"
    else
        green "[cmake] 기존 캐시 재사용"
    fi

    yellow "[ninja] 빌드 중..."
    ninja -C "$BUILD_DIR" omr_flutter_test -j"$(nproc)"

    green "[ok] 빌드 완료: $EXE"
    echo ""
    echo "실행 예시:"
    echo "  bash build_flutter_desktop.sh run test.jpg"
    echo "  bash build_flutter_desktop.sh run test.jpg result.xml"
}

cmd_run() {
    local image_path output_path
    image_path="$(realpath "$1" 2>/dev/null || echo "$1")"
    output_path="$(realpath "${2:-result.xml}" 2>/dev/null || echo "${2:-result.xml}")"

    if [[ -z "$1" ]]; then
        red "[error] 이미지 경로를 입력하세요."
        echo "  사용법: bash build_flutter_desktop.sh run <image.jpg> [output.xml]"
        exit 1
    fi

    cmd_build

    if [[ ! -f "$image_path" ]]; then
        red "[error] 이미지 파일 없음: $image_path"
        exit 1
    fi

    if [[ ! -d "$ASSETS_DIR" ]]; then
        red "[error] 모델 파일 디렉토리 없음: $ASSETS_DIR"
        exit 1
    fi

    echo ""
    yellow "[omr] Flutter 코드 경로로 인식 시작: $(basename "$image_path")"
    echo "────────────────────────────────────────"

    local ort_lib_dir
    ort_lib_dir="$(find "$BUILD_DIR/_deps" -name "libonnxruntime.so*" -exec dirname {} \; 2>/dev/null | head -1)"
    if [[ -n "$ort_lib_dir" ]]; then
        export LD_LIBRARY_PATH="$ort_lib_dir:${LD_LIBRARY_PATH:-}"
    fi

    "$EXE" "$ASSETS_DIR" "$image_path" "$output_path"
    local exit_code=$?

    echo "────────────────────────────────────────"
    if [[ $exit_code -eq 0 ]]; then
        green "[ok] 결과 저장: $output_path"
        echo ""
        echo "MusicXML 미리보기 (앞 30줄):"
        head -30 "$output_path"
    else
        red "[error] 인식 실패 (exit code: $exit_code)"
    fi
    return $exit_code
}

cmd_clean() {
    yellow "[clean] 빌드 디렉토리 삭제: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    green "[ok] 완료"
}

cmd_help() {
    cat <<EOF
사용법: bash build_flutter_desktop.sh [명령] [옵션]

명령:
  build                      빌드 (기본값)
  run <image> [output.xml]   이미지 인식 실행
  clean                      빌드 디렉토리 삭제
  help                       이 도움말 출력

예시:
  bash build_flutter_desktop.sh
  bash build_flutter_desktop.sh run test.jpg
  bash build_flutter_desktop.sh run test.png
  bash build_flutter_desktop.sh run /mnt/c/Users/kyutae/Pictures/score.jpg result.xml

빌드 위치: $BUILD_DIR
모델 위치: $ASSETS_DIR
EOF
}

case "${1:-build}" in
    build)          cmd_build ;;
    run)            cmd_run "$2" "$3" ;;
    clean)          cmd_clean ;;
    help|--help|-h) cmd_help ;;
    *)
        red "[error] 알 수 없는 명령: $1"
        cmd_help
        exit 1
        ;;
esac
