cmake_minimum_required(VERSION 3.22.1)
project("musicscore_flutter")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── MusicScore C++ 소스 루트 (상대 경로) ────────────────────────────────────
# CMakeLists.txt 위치: musicscore_flutter/android/app/src/main/cpp/
# 6단계 위            → AndroidStudioProjects/
set(HOMR_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../MusicScore/app/src/main/cpp")

# ─── 외부 라이브러리: tinyxml2 (FetchContent) ────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG        10.0.0
)
FetchContent_MakeAvailable(tinyxml2)

# ─── 외부 라이브러리: OpenCV (prefab) ────────────────────────────────────────
find_package(OpenCV REQUIRED CONFIG)

# ─── 외부 라이브러리: TFLite (prefab) ────────────────────────────────────────
find_package(tensorflowlite REQUIRED CONFIG)

# ─── 외부 라이브러리: ONNX Runtime (prefab) ──────────────────────────────────
find_package(onnxruntime REQUIRED CONFIG)

# ─── Android 시스템 라이브러리 ────────────────────────────────────────────────
find_library(log-lib    log)
find_library(android-lib android)

# ─── 소스 파일 목록 ───────────────────────────────────────────────────────────
add_library(${CMAKE_PROJECT_NAME} SHARED
    flutter_jni.cpp

    # 전처리
    ${HOMR_CPP_ROOT}/homr/autocrop.cpp
    ${HOMR_CPP_ROOT}/homr/resize.cpp
    ${HOMR_CPP_ROOT}/homr/color_adjust.cpp
    ${HOMR_CPP_ROOT}/homr/noise_filtering.cpp

    # 기하 구조
    ${HOMR_CPP_ROOT}/homr/bounding_boxes.cpp
    ${HOMR_CPP_ROOT}/homr/find_peaks.cpp

    # 오선·심볼 감지
    ${HOMR_CPP_ROOT}/homr/staff_detection.cpp
    ${HOMR_CPP_ROOT}/homr/staff_dewarping.cpp
    ${HOMR_CPP_ROOT}/homr/note_detection.cpp
    ${HOMR_CPP_ROOT}/homr/bar_line_detection.cpp
    ${HOMR_CPP_ROOT}/homr/staff_parsing.cpp

    # 세그멘테이션 추론
    ${HOMR_CPP_ROOT}/homr/segmentation/inference_segnet.cpp

    # 트랜스포머 추론
    ${HOMR_CPP_ROOT}/homr/transformer/encoder_inference.cpp
    ${HOMR_CPP_ROOT}/homr/transformer/decoder_inference.cpp
    ${HOMR_CPP_ROOT}/homr/transformer/staff2score.cpp

    # 음악 후처리
    ${HOMR_CPP_ROOT}/homr/tr_omr_parser.cpp
    ${HOMR_CPP_ROOT}/homr/rhythm_rules.cpp
    ${HOMR_CPP_ROOT}/homr/accidental_rules.cpp
    ${HOMR_CPP_ROOT}/homr/music_xml_generator.cpp

    # 파이프라인
    ${HOMR_CPP_ROOT}/homr/pipeline.cpp
)

# ─── include 경로 ─────────────────────────────────────────────────────────────
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${HOMR_CPP_ROOT}
)

# ─── 링크 ─────────────────────────────────────────────────────────────────────
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${android-lib}
    ${log-lib}
    OpenCV::OpenCV
    tensorflowlite::tensorflowlite
    onnxruntime::onnxruntime
    tinyxml2
)
