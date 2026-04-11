# Install script for directory: /mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/contrib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/eight_bit_int_gemm" TYPE FILE FILES "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/eight_bit_int_gemm/eight_bit_int_gemm.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/meta" TYPE FILE FILES
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/base.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/legacy_multi_thread_common.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/legacy_multi_thread_gemm.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/legacy_multi_thread_gemv.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/legacy_operations_common.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/legacy_single_thread_gemm.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/multi_thread_common.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/multi_thread_gemm.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/multi_thread_transform.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/quantized_mul_kernels.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/quantized_mul_kernels_arm_32.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/quantized_mul_kernels_arm_64.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/single_thread_gemm.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/single_thread_transform.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/streams.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/streams_arm_32.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/streams_arm_64.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/transform_kernels.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/transform_kernels_arm_32.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/meta/transform_kernels_arm_64.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/public" TYPE FILE FILES
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/public/bit_depth.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/public/gemmlowp.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/public/map.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/public/output_stages.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/profiling" TYPE FILE FILES
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/profiling/instrumentation.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/profiling/profiler.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/profiling/pthread_everywhere.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/internal" TYPE FILE FILES
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/allocator.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/block_params.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/common.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/compute.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/detect_platform.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/dispatch_gemm_shape.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/kernel.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/kernel_avx.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/kernel_default.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/kernel_msa.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/kernel_neon.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/kernel_reference.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/kernel_sse.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/multi_thread_gemm.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/output.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/output_avx.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/output_msa.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/output_neon.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/output_sse.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/pack.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/pack_avx.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/pack_msa.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/pack_neon.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/pack_sse.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/platform.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/simd_wrappers.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/simd_wrappers_common_neon_sse.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/simd_wrappers_msa.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/simd_wrappers_neon.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/simd_wrappers_sse.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/single_thread_gemm.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/internal/unpack.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/fixedpoint" TYPE FILE FILES
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/fixedpoint/fixedpoint.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/fixedpoint/fixedpoint_avx.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/fixedpoint/fixedpoint_msa.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/fixedpoint/fixedpoint_neon.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/fixedpoint/fixedpoint_sse.h"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/gemmlowp/fixedpoint/fixedpoint_wasmsimd.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/_deps/gemmlowp-build/libeight_bit_int_gemm.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp/gemmlowp-config.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp/gemmlowp-config.cmake"
         "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/_deps/gemmlowp-build/CMakeFiles/Export/lib/cmake/gemmlowp/gemmlowp-config.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp/gemmlowp-config-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp/gemmlowp-config.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp" TYPE FILE FILES "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/_deps/gemmlowp-build/CMakeFiles/Export/lib/cmake/gemmlowp/gemmlowp-config.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp" TYPE FILE FILES "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/_deps/gemmlowp-build/CMakeFiles/Export/lib/cmake/gemmlowp/gemmlowp-config-release.cmake")
  endif()
endif()

