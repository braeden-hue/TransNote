# tflite_c_api_shim

`org.tensorflow:tensorflow-lite`'s AAR ships a *pruned* subset of the TFLite C
API headers under `headers/` (see `android/app/build.gradle.kts`, which
extracts that subset for the CMake build via `-DTFLITE_ROOT=`). That subset is
missing two headers that `tensorflow/lite/core/c/c_api.h` transitively
`#include`s:

- `tensorflow/lite/core/async/c/types.h`
- `tensorflow/lite/core/c/registration_external.h`

Both are small, self-contained (only depend on headers already present in the
AAR's subset) and only declare types/APIs for TFLite's *async* delegate
execution path, which this engine never uses. They're vendored here verbatim
(Apache License 2.0, from the `tensorflow/tensorflow` repo, tag `v2.16.1` --
matching the `tensorflow-lite` version pinned in
`android/app/build.gradle.kts`) purely to satisfy the `#include` chain.

`ml/omr/engine/CMakeLists.txt` adds this directory to the include path
alongside `${TFLITE_ROOT}/include` for both Android and desktop builds.

If a future TFLite AAR bump ships these headers directly, this shim directory
can be deleted along with the corresponding CMakeLists.txt include line.
