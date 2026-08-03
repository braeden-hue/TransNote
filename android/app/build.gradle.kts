import java.util.zip.ZipFile

plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

// ─── TFLite C API extraction for the CMake native build ───────────────────────
// org.tensorflow:tensorflow-lite's AAR is NOT a prefab package (unlike OpenCV
// below) -- it only ships the TFLite C API headers (tensorflow/lite/c/*.h)
// plus libtensorflowlite_jni.so per ABI in classic jni/<abi>/ layout, so CMake
// can't find_package() it. We extract those files ourselves, once per
// configuration, into a stable location and rename the .so to
// libtensorflowlite.so so ml/omr/engine/CMakeLists.txt's
// find_library(TFLITE_LIB tensorflowlite ...) picks it up
// (see -DTFLITE_ROOT= passed to externalNativeBuild.cmake.arguments below).
val tfliteExtractedDir = layout.buildDirectory.dir("tflite_extracted").get().asFile
run {
    val tfliteConfig = configurations.detachedConfiguration(
        dependencies.create("org.tensorflow:tensorflow-lite:2.16.1")
    )
    val aarFile = tfliteConfig.resolve().first { it.name.endsWith(".aar") }

    val includeDir = File(tfliteExtractedDir, "include")
    val markerFile = File(tfliteExtractedDir, ".extracted-${aarFile.name}")
    if (!markerFile.exists()) {
        tfliteExtractedDir.deleteRecursively()
        includeDir.mkdirs()

        ZipFile(aarFile).use { zip ->
            val abiMap = mapOf(
                "arm64-v8a"   to "arm64-v8a",
                "armeabi-v7a" to "armeabi-v7a",
                "x86"         to "x86",
                "x86_64"      to "x86_64",
            )
            for (entry in zip.entries()) {
                when {
                    entry.name.startsWith("headers/") && !entry.isDirectory -> {
                        val dest = File(includeDir, entry.name.removePrefix("headers/"))
                        dest.parentFile.mkdirs()
                        zip.getInputStream(entry).use { input -> dest.outputStream().use { input.copyTo(it) } }
                    }
                    entry.name.startsWith("jni/") && entry.name.endsWith("libtensorflowlite_jni.so") -> {
                        val abi = entry.name.removePrefix("jni/").substringBefore('/')
                        val abiDirName = abiMap[abi] ?: abi
                        val dest = File(tfliteExtractedDir, "lib/$abiDirName/libtensorflowlite.so")
                        dest.parentFile.mkdirs()
                        zip.getInputStream(entry).use { input -> dest.outputStream().use { input.copyTo(it) } }
                    }
                }
            }
        }
        markerFile.writeText("ok")
    }
}

android {
    namespace = "com.example.musicscore_flutter"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.musicscore_flutter"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++20"
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DTFLITE_ROOT=${tfliteExtractedDir.absolutePath}",
                )
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            // ml/omr/engine — 자체 구현 C++ OMR 엔진 (Dart FFI 대상, libomr_engine.so).
            // 예전 MusicScore/ 형제 저장소를 링크하던 android/app/src/main/cpp/CMakeLists.txt는 제거됨.
            path = file("../../ml/omr/engine/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        prefab = true
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // OpenCV (prefab)
    implementation("org.opencv:opencv:4.9.0")

    // TensorFlow Lite — ml/omr/engine은 segnet/encoder/decoder 전부 TFLite로 통일되어
    // 있으므로(homr 기반 구엔진과 달리) ONNX Runtime 의존성은 더 이상 필요 없다.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
}
