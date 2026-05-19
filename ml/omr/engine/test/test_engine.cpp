/**
 * test_engine.cpp  –  Standalone desktop smoke-test for the OMR engine.
 *
 * Build (Linux / WSL):
 *   cd omr/engine
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release \
 *         -DTFLITE_ROOT=/path/to/tflite \
 *         -DOpenCV_DIR=/usr/include/opencv4
 *   cmake --build build -j$(nproc)
 *   ./build/omr_test <image.jpg> <segnet.tflite> <encoder.tflite> \
 *                    <decoder.tflite> <tokenizer.json>
 */

#include "omr_engine.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); std::exit(1); }
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Usage: omr_test <image> <segnet.tflite> <encoder.tflite>"
            " <decoder.tflite> <tokenizer.json>\n");
        return 1;
    }

    const char* img_path  = argv[1];
    const char* seg_path  = argv[2];
    const char* enc_path  = argv[3];
    const char* dec_path  = argv[4];
    const char* tok_path  = argv[5];

    printf("Loading engine...\n");
    OmrEngineHandle engine = omr_create(seg_path, enc_path, dec_path, tok_path);
    if (!engine) { fprintf(stderr, "omr_create failed\n"); return 1; }
    printf("Engine ready.\n");

    printf("Reading image: %s\n", img_path);
    auto img_bytes = read_file(img_path);

    printf("Running inference...\n");
    OmrResultC* result = omr_process(engine,
                                      img_bytes.data(),
                                      (int32_t)img_bytes.size());

    if (!result || !result->success) {
        fprintf(stderr, "Inference failed: %s\n",
                result ? result->error : "(null result)");
        omr_free_result(result);
        omr_destroy(engine);
        return 1;
    }

    printf("Decoded %d tokens:\n", result->token_count);
    for (int i = 0; i < result->token_count; ++i)
        printf("  [%4d] %5d  %s\n", i, result->tokens[i].id, result->tokens[i].text);

    omr_free_result(result);
    omr_destroy(engine);
    return 0;
}
