#include <jni.h>
#include <memory>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <opencv2/opencv.hpp>

#include "homr/pipeline.hpp"

// ─── 전역 파이프라인 인스턴스 ─────────────────────────────────────────────────
static std::unique_ptr<homr::OmrPipeline> g_pipeline;

extern "C" {

// ─── JNI: 파이프라인 초기화 ──────────────────────────────────────────────────
// Kotlin: OmrPipeline.nativeInit(assetManager)
JNIEXPORT jboolean JNICALL
Java_com_example_musicscore_1flutter_OmrPipeline_nativeInit(
        JNIEnv* env, jobject /* this */, jobject asset_manager_obj) {
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager_obj);
    if (!mgr) return JNI_FALSE;

    g_pipeline = std::make_unique<homr::OmrPipeline>(mgr);
    return g_pipeline->is_ready() ? JNI_TRUE : JNI_FALSE;
}

// ─── JNI: JPEG/PNG 바이트 배열 → MusicXML 문자열 ────────────────────────────
// Kotlin: OmrPipeline.nativeProcessImageBytes(byte[])
JNIEXPORT jstring JNICALL
Java_com_example_musicscore_1flutter_OmrPipeline_nativeProcessImageBytes(
        JNIEnv* env, jobject /* this */, jbyteArray image_bytes) {
    if (!g_pipeline || !g_pipeline->is_ready()) return nullptr;

    jsize len = env->GetArrayLength(image_bytes);
    jbyte* data = env->GetByteArrayElements(image_bytes, nullptr);

    std::vector<uchar> buf(data, data + len);
    env->ReleaseByteArrayElements(image_bytes, data, JNI_ABORT);

    cv::Mat bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (bgr.empty()) return nullptr;

    auto result = g_pipeline->process_image(bgr);
    if (!result.success) return nullptr;

    return env->NewStringUTF(result.music_xml.c_str());
}

} // extern "C"
