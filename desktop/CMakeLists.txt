cmake_minimum_required(VERSION 3.22)
project(omr_flutter_test CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ─── homr C++ 소스 루트 ───────────────────────────────────────────────────────
# musicscore_flutter/desktop/ → ../../ → AndroidStudioProjects/
set(HOMR_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../MusicScore/app/src/main/cpp")

# ─── 기본 assets / 테스트 이미지 경로 (빌드 시 고정, 실행 시 인자 생략 가능) ──
set(DEFAULT_ASSETS_DIR "/mnt/c/Users/kyutae/AndroidStudioProjects/MusicScore/app/src/main/assets"
    CACHE STRING "기본 모델 파일 디렉토리")
set(DEFAULT_IMAGE_DIR "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop"
    CACHE STRING "기본 이미지 디렉토리 (파일명만 입력하면 이 경로에서 찾음)")
set(DEFAULT_IMAGE_NAME "num2.png"
    CACHE STRING "기본 테스트 이미지 파일명")

# ─── OpenCV (시스템 설치: sudo apt install libopencv-dev) ─────────────────────
find_package(OpenCV REQUIRED)

# ─── ONNX Runtime (decoder 전용, prebuilt 바이너리) ──────────────────────────
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-arm64-1.18.0.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-win-x64-1.18.0.zip")
endif()

FetchContent_Declare(onnxruntime_prebuilt URL ${ORT_URL})
FetchContent_MakeAvailable(onnxruntime_prebuilt)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION  "${onnxruntime_prebuilt_SOURCE_DIR}/lib/onnxruntime.dll"
        IMPORTED_IMPLIB    "${onnxruntime_prebuilt_SOURCE_DIR}/lib/onnxruntime.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_prebuilt_SOURCE_DIR}/include")
else()
    file(GLOB ORT_SO "${onnxruntime_prebuilt_SOURCE_DIR}/lib/libonnxruntime.so*"
                     "${onnxruntime_prebuilt_SOURCE_DIR}/lib/libonnxruntime.dylib*")
    list(GET ORT_SO 0 ORT_SO_PATH)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION  "${ORT_SO_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_prebuilt_SOURCE_DIR}/include")
endif()

# ─── TensorFlow Lite (segnet + encoder 전용) ──────────────────────────────────
# MusicScore 데스크톱 빌드에서 이미 다운로드된 TF 소스를 재사용 (재다운로드 없음)
set(ABSL_PROPAGATE_CXX_STD ON)

set(TFLITE_SRC_HINT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../MusicScore/app/src/main/cpp/desktop/build/_deps/tensorflow_src-src")

if(EXISTS "${TFLITE_SRC_HINT}/tensorflow/lite/CMakeLists.txt")
    message(STATUS "TFLite: 기존 소스 재사용 → ${TFLITE_SRC_HINT}")
    set(TFLITE_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    # TSL/XLA 헤더를 Python TF 패키지에서 제공 (shallow clone에서 누락된 파일 보완)
    set(TF_PY_INCLUDE "/home/braeden/.local/lib/python3.10/site-packages/tensorflow/include")
    if(EXISTS "${TF_PY_INCLUDE}")
        include_directories(SYSTEM "${TF_PY_INCLUDE}")
        message(STATUS "TFLite: TSL/XLA 헤더 추가 → ${TF_PY_INCLUDE}")
    endif()
    add_subdirectory(
        "${TFLITE_SRC_HINT}/tensorflow/lite"
        "${CMAKE_BINARY_DIR}/tflite_build"
        EXCLUDE_FROM_ALL)
else()
    message(STATUS "TFLite: 소스 새로 다운로드 (수 분 소요)")
    FetchContent_Declare(tensorflow_src
        GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
        GIT_TAG        v2.16.1
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE)
    FetchContent_Populate(tensorflow_src)
    add_subdirectory(
        "${tensorflow_src_SOURCE_DIR}/tensorflow/lite"
        "${CMAKE_BINARY_DIR}/tflite_build"
        EXCLUDE_FROM_ALL)
endif()

# ─── tinyxml2 ─────────────────────────────────────────────────────────────────
FetchContent_Declare(tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG        10.0.0)
FetchContent_MakeAvailable(tinyxml2)

# ─── 소스 파일 목록 ───────────────────────────────────────────────────────────
add_executable(omr_flutter_test
    test_main.cpp
    asset_manager_impl.cpp

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
# TFLite 헤더: TF 소스 루트에서 탐색
if(EXISTS "${TFLITE_SRC_HINT}/tensorflow/lite/CMakeLists.txt")
    set(TFLITE_INCLUDE_ROOT "${TFLITE_SRC_HINT}")
else()
    set(TFLITE_INCLUDE_ROOT "${tensorflow_src_SOURCE_DIR}")
endif()

target_include_directories(omr_flutter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # Android mock 헤더
    ${HOMR_CPP_ROOT}
    ${TFLITE_INCLUDE_ROOT}                # tensorflow/lite/*.h
    "${CMAKE_BINARY_DIR}/tflite_build/flatbuffers/include"  # flatbuffers (TFLite 의존)
)

# ─── 기본 경로 컴파일 정의 ────────────────────────────────────────────────────
target_compile_definitions(omr_flutter_test PRIVATE
    DEFAULT_ASSETS_DIR="${DEFAULT_ASSETS_DIR}"
    DEFAULT_IMAGE_DIR="${DEFAULT_IMAGE_DIR}"
    DEFAULT_IMAGE_NAME="${DEFAULT_IMAGE_NAME}"
)

# ─── 링크 ─────────────────────────────────────────────────────────────────────
target_link_libraries(omr_flutter_test PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    tensorflow-lite
    tinyxml2
)

# ─── make run 타겟 ────────────────────────────────────────────────────────────
add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:onnxruntime>:$ENV{LD_LIBRARY_PATH}"
        $<TARGET_FILE:omr_flutter_test>
    DEPENDS omr_flutter_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "OMR 실행 중..."
    USES_TERMINAL
)
