cmake_minimum_required(VERSION 3.22)
project(omr_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Compiler flags ───────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

# ─── Source files ─────────────────────────────────────────────────────────────
set(ENGINE_SOURCES
    src/omr_engine.cpp
    src/preprocessor.cpp
    src/perspective_corrector.cpp
    src/noise_filter.cpp
    src/segnet_runner.cpp
    src/staff_detector.cpp
    src/staff_canvas.cpp
    src/page_dewarper.cpp
    src/encoder_runner.cpp
    src/decoder_runner.cpp
    src/token_parser.cpp
)

# ─── Shared library (Dart FFI target) ────────────────────────────────────────
add_library(omr_engine SHARED ${ENGINE_SOURCES})
target_include_directories(omr_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ─── OpenCV ───────────────────────────────────────────────────────────────────
# Android: set OPENCV_ANDROID_SDK to the extracted OpenCV-android-sdk directory.
# Desktop: install opencv4 via apt / brew / vcpkg.
if(ANDROID)
    set(OPENCV_ANDROID_SDK "" CACHE PATH "Path to OpenCV Android SDK")
    set(OpenCV_DIR "${OPENCV_ANDROID_SDK}/sdk/native/jni")
    find_package(OpenCV REQUIRED core imgproc imgcodecs)
else()
    find_package(OpenCV REQUIRED core imgproc imgcodecs)
endif()
target_link_libraries(omr_engine PRIVATE ${OpenCV_LIBS})
target_include_directories(omr_engine PRIVATE ${OpenCV_INCLUDE_DIRS})

# ─── TensorFlow Lite ──────────────────────────────────────────────────────────
# Set TFLITE_ROOT to the directory containing:
#   include/tensorflow/lite/*.h
#   lib/libtensorflowlite.so  (Android)  or  lib/libtensorflow-lite.a  (desktop)
set(TFLITE_ROOT "" CACHE PATH "Path to TFLite installation")

target_include_directories(omr_engine PRIVATE "${TFLITE_ROOT}/include")

if(ANDROID)
    find_library(TFLITE_LIB tensorflowlite
                 PATHS "${TFLITE_ROOT}/lib/${ANDROID_ABI}"
                 NO_DEFAULT_PATH)
else()
    find_library(TFLITE_LIB tensorflow-lite
                 PATHS "${TFLITE_ROOT}/lib"
                 NO_DEFAULT_PATH)
endif()

if(NOT TFLITE_LIB)
    message(FATAL_ERROR
        "TFLite library not found. Set -DTFLITE_ROOT=<path> when invoking CMake.")
endif()
target_link_libraries(omr_engine PRIVATE ${TFLITE_LIB})

# ─── Android-specific ────────────────────────────────────────────────────────
if(ANDROID)
    target_link_libraries(omr_engine PRIVATE log android)
    set_target_properties(omr_engine PROPERTIES
        ANDROID_STL c++_shared)
endif()

# ─── Desktop binaries ────────────────────────────────────────────────────────
if(NOT ANDROID)
    # Smoke-test: run inference on a single image and print tokens.
    add_executable(omr_test test/test_engine.cpp)
    target_link_libraries(omr_test PRIVATE omr_engine)
    target_include_directories(omr_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Batch evaluator: compute TER / accuracy over a labelled test set.
    add_executable(omr_eval test/eval_engine.cpp)
    target_link_libraries(omr_eval PRIVATE omr_engine)
    target_include_directories(omr_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# ─── Build summary ────────────────────────────────────────────────────────────
message(STATUS "omr_engine build:")
message(STATUS "  OpenCV    : ${OpenCV_VERSION}")
message(STATUS "  TFLite    : ${TFLITE_LIB}")
message(STATUS "  C++ std   : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
