# Install script for directory: /mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/AdolcForward"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/AlignedVector3"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/ArpackSupport"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/AutoDiff"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/BVH"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/EulerAngles"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/FFT"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/IterativeSolvers"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/KroneckerProduct"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/MatrixFunctions"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/MoreVectorization"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/MPRealSupport"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/NNLS"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/NonLinearOptimization"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/NumericalDiff"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/OpenGLSupport"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/Polynomials"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/Skyline"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/SparseExtra"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/SpecialFunctions"
    "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/_deps/eigen-build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

