// Flutter OMR desktop test entry point
//
// Flutter 앱의 nativeProcessImageBytes 와 동일한 코드 경로를 검증:
//   1. AAssetManager mock 초기화  →  OmrPipeline 생성
//   2. 이미지 파일을 바이트로 읽기 (Flutter MethodChannel이 전달하는 방식 그대로)
//   3. cv::imdecode → pipeline.process_image → MusicXML 저장
//
// Usage: omr_flutter_test <assets_dir> <image.jpg> [output.xml]
//
// Example:
//   omr_flutter_test ../MusicScore/app/src/main/assets sheet.jpg result.xml
//
// assets_dir must contain:
//   segnet_308_int8.tflite
//   encoder_331_int8.tflite
//   decoder_331_int8.onnx
//   tokenizer_rhythm.json
//   tokenizer_pitch.json
//   tokenizer_lift.json
//   tokenizer_note.json

#include "android/asset_manager.h"
#include "homr/pipeline.hpp"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// 파일 경로에서 확장자를 제거한 베이스명 반환
// e.g. "/path/to/num2.png" → "/path/to/num2"
static std::string strip_extension(const std::string& path) {
    auto dot = path.rfind('.');
    auto sep = path.find_last_of("/\\");
    if (dot != std::string::npos && (sep == std::string::npos || dot > sep))
        return path.substr(0, dot);
    return path;
}

// 파일명에 경로 구분자가 없으면 DEFAULT_IMAGE_DIR 를 앞에 붙여 반환
static std::string resolve_image_path(const std::string& name) {
    if (name.find('/') == std::string::npos &&
        name.find('\\') == std::string::npos)
        return std::string(DEFAULT_IMAGE_DIR) + "/" + name;
    return name;
}

int main(int argc, char* argv[]) {
    // 사용법:
    //   ./omr_flutter_test                    → 기본 이미지(DEFAULT_IMAGE_DIR/DEFAULT_IMAGE_NAME)
    //   ./omr_flutter_test num3.jpg           → desktop 폴더의 num3.jpg
    //   ./omr_flutter_test num3.jpg out.xml   → 출력 경로 지정
    const char* assets_dir = DEFAULT_ASSETS_DIR;
    std::string image_str  = (argc > 1)
        ? resolve_image_path(argv[1])
        : std::string(DEFAULT_IMAGE_DIR) + "/" + DEFAULT_IMAGE_NAME;
    // 출력 경로: 명시적으로 지정하지 않으면 입력 파일명(확장자 제거) + ".musicxml"
    std::string output_str = (argc > 2)
        ? std::string(argv[2])
        : strip_extension(image_str) + ".musicxml";
    const char* output_path = output_str.c_str();
    const char* image_path  = image_str.c_str();

    // ── 1. 파이프라인 초기화 (Flutter: nativeInit 호출과 동일) ──────────────────
    AAssetManager* mgr = desktop_create_asset_manager(assets_dir);
    homr::OmrPipeline pipeline(mgr);x

    if (!pipeline.is_ready()) {
        fprintf(stderr, "[error] 파이프라인 초기화 실패 — 모델 파일 확인: %s\n", assets_dir);
        desktop_destroy_asset_manager(mgr);
        return 1;
    }
    fprintf(stdout, "[ok] 파이프라인 준비됨\n");

    // ── 2. 이미지를 바이트로 읽기 (Flutter: Uint8List.fromList(bytes) 와 동일) ──
    std::ifstream img_file(image_path, std::ios::binary);
    if (!img_file) {
        fprintf(stderr, "[error] 이미지 파일 없음: %s\n", image_path);
        desktop_destroy_asset_manager(mgr);
        return 1;
    }
    std::vector<uchar> buf(
        (std::istreambuf_iterator<char>(img_file)),
        std::istreambuf_iterator<char>{});
    fprintf(stdout, "[ok] 이미지 로드: %s (%zu bytes)\n", image_path, buf.size());

    // ── 3. 인식 실행 (Flutter: nativeProcessImageBytes 와 동일한 코드 경로) ──────
    cv::Mat bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        fprintf(stderr, "[error] 이미지 디코딩 실패: %s\n", image_path);
        desktop_destroy_asset_manager(mgr);
        return 1;
    }
    fprintf(stdout, "[ok] 디코딩 완료: %dx%d\n", bgr.cols, bgr.rows);

    fprintf(stdout, "[..] OMR 파이프라인 실행 중...\n");
    fflush(stdout);

    auto result = pipeline.process_image(bgr);

    if (!result.success) {
        fprintf(stderr, "[error] 인식 실패: %s\n", result.error_message.c_str());
        desktop_destroy_asset_manager(mgr);
        return 1;
    }

    // ── 4. 결과 저장 ─────────────────────────────────────────────────────────────
    {
        std::ofstream out(output_path);
        if (!out) {
            fprintf(stderr, "[error] 파일 쓰기 실패: %s\n", output_path);
            desktop_destroy_asset_manager(mgr);
            return 1;
        }
        out << result.music_xml;
    }
    fprintf(stdout, "[ok] MusicXML 저장: %s (%zu bytes)\n",
            output_path, result.music_xml.size());

    desktop_destroy_asset_manager(mgr);
    return 0;
}
