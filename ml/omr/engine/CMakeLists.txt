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
# Android: consumed via prefab (android/app/build.gradle.kts declares
# `implementation("org.opencv:opencv:4.9.0")` with buildFeatures.prefab = true).
# AGP auto-generates an OpenCVConfig.cmake for this build and points
# CMAKE_FIND_ROOT_PATH at it, so a plain find_package(... CONFIG) resolves it --
# no manual OPENCV_ANDROID_SDK path needed. The AAR exposes a single monolithic
# module (opencv_java4, includes core+imgproc+imgcodecs+... already linked in),
# unlike the desktop OpenCVConfig.cmake's per-component COMPONENTS API.
# Desktop: install opencv4 via apt / brew / vcpkg (classic find_package).
if(ANDROID)
    find_package(OpenCV CONFIG REQUIRED)
    set(OMR_OPENCV_LIB OpenCV::opencv_java4)
else()
    find_package(OpenCV REQUIRED core imgproc imgcodecs)
    set(OMR_OPENCV_LIB ${OpenCV_LIBS})
    target_include_directories(omr_engine PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(omr_engine PRIVATE ${OMR_OPENCV_LIB})

# ─── TensorFlow Lite ──────────────────────────────────────────────────────────
# Android: org.tensorflow:tensorflow-lite's AAR is NOT a prefab package -- it
# only ships the TFLite *C API* headers/symbols (tensorflow/lite/c/c_api.h +
# libtensorflowlite_jni.so), not the full C++ Interpreter/FlatBufferModel API
# nor a prefab module.cmake. That's why this engine's TFLite runner classes
# (segnet/encoder/decoder_runner) are written against the C API only.
# android/app/build.gradle.kts extracts that AAR's headers + renames
# libtensorflowlite_jni.so -> libtensorflowlite.so per-ABI into
# <buildDir>/tflite_extracted/ (at Gradle configuration time, see the top of
# build.gradle.kts) and passes -DTFLITE_ROOT=<that dir> here. Renaming the
# copy used only for linking is safe: the .so's embedded SONAME is still
# "libtensorflowlite_jni.so", so libomr_engine.so's DT_NEEDED entry still
# correctly points at the untouched copy Gradle bundles into the APK via the
# normal jniLibs merge of the same AAR dependency.
# Desktop: point TFLITE_ROOT at a TFLite C API build (include/ + lib/).
set(TFLITE_ROOT "" CACHE PATH "Path to TFLite installation (C API headers + lib)")

target_include_directories(omr_engine PRIVATE "${TFLITE_ROOT}/include")
# See third_party/tflite_c_api_shim/README.md -- the AAR's header subset is
# missing two headers transitively included by tensorflow/lite/c/c_api.h.
target_include_directories(omr_engine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite_c_api_shim")

if(ANDROID)
    # NO_CMAKE_FIND_ROOT_PATH is required here: the NDK toolchain sets
    # CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY (pointed at AGP's prefab dir for
    # OpenCV above), which silently rejects any PATHS outside that root --
    # including an explicit TFLITE_ROOT -- unless root-path filtering is
    # disabled for this call.
    find_library(TFLITE_LIB tensorflowlite
                 PATHS "${TFLITE_ROOT}/lib/${ANDROID_ABI}"
                 NO_DEFAULT_PATH
                 NO_CMAKE_FIND_ROOT_PATH)
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
