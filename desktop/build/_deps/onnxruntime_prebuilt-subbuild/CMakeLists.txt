# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(onnxruntime_prebuilt-populate NONE)



include(ExternalProject)
ExternalProject_Add(onnxruntime_prebuilt-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz"
                    SOURCE_DIR          "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/_deps/onnxruntime_prebuilt-src"
                    BINARY_DIR          "/mnt/c/Users/kyutae/AndroidStudioProjects/musicscore_flutter/desktop/build/_deps/onnxruntime_prebuilt-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


