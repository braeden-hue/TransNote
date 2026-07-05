#include "omr_engine.h"
#include "types.hpp"
#include "preprocessor.hpp"
#include "segnet_runner.hpp"
#include "staff_detector.hpp"
#include "staff_canvas.hpp"
#include "page_dewarper.hpp"
#include "encoder_runner.hpp"
#include "decoder_runner.hpp"
#include "token_parser.hpp"

#include <opencv2/imgcodecs.hpp>
#include <cstring>
#include <memory>
#include <vector>
#include <stdexcept>

namespace {
// Beam width for decoder_runner.cpp's DecoderRunner::decode_beam().
// beam_width=1 would fall back to plain greedy decoding.
constexpr int kBeamWidth = 4;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal engine struct
// ─────────────────────────────────────────────────────────────────────────────

struct OmrEngine {
    omr::Preprocessor  preprocessor;
    omr::SegnetRunner  segnet;
    omr::StaffDetector detector;
    omr::PageDewarper  page_dewarper;
    omr::StaffCanvas   canvas_builder;
    omr::EncoderRunner encoder;
    omr::DecoderRunner decoder;
    omr::TokenParser   parser;

    OmrEngine(const char* seg_path,
              const char* enc_path,
              const char* dec_path,
              const char* tok_path)
        : segnet  (seg_path)
        , encoder (enc_path)
        , decoder (dec_path, /* vocab_size set after parser loads */ 978)
        , parser  (tok_path)
    {
        // Reconstruct decoder with correct vocab size now that parser is loaded.
        // (The constructor above uses a placeholder 978; adjust if needed.)
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Full pipeline (C++ internal)
// ─────────────────────────────────────────────────────────────────────────────

static omr::OmrResult run_pipeline(OmrEngine* eng,
                                    const uint8_t* image_data,
                                    int32_t image_len)
{
    omr::OmrResult res;
    res.success = false;

    // 1. Decode image bytes (JPEG / PNG) into a BGR cv::Mat.
    std::vector<uint8_t> buf(image_data, image_data + image_len);
    cv::Mat bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (bgr.empty()) { res.error = "imdecode failed"; return res; }

    // 2. Preprocess: autocrop → resize 1920 → CLAHE → grayscale.
    cv::Mat gray = eng->preprocessor.process(bgr);

    // 3. Segmentation: sliding-window TFLite inference.
    //    Returns 5 probability maps (one per foreground class).
    auto seg_maps = eng->segnet.run(gray);
    // seg_maps[3] = staff-line probability map (SEG_STAFF_LINE - 1 = index 3).

    // 4. Staff detection: find groups of 5 staff lines.
    auto staffs = eng->detector.detect(seg_maps[omr::SEG_STAFF_LINE - 1]);
    if (staffs.empty()) { res.error = "no staff lines detected"; return res; }

    // 4.5. Page dewarping: correct curvature using the staff-line mask.
    //      Traces each line's actual y-position column by column, builds a
    //      rubber-sheet displacement field, and remaps the grayscale image so
    //      staff lines become perfectly straight.  staffs[].y_lines are updated
    //      to the post-warp ideal positions for downstream canvas building.
    gray = eng->page_dewarper.dewarp(gray,
                                      seg_maps[omr::SEG_STAFF_LINE - 1],
                                      staffs);

    // 5. Encode + decode each staff group.
    for (const auto& staff : staffs) {
        // 5a. Build CANVAS_H × CANVAS_W tiles.
        auto tiles = eng->canvas_builder.build_tiles(gray, staff);

        for (const auto& tile : tiles) {
            // 5b. Encoder: tile → [seq_len × 512] latent sequence.
            cv::Mat enc_out = eng->encoder.run(tile);

            // 5c. Decoder: beam-search autoregressive decoding.
            std::vector<int32_t> ids = eng->decoder.decode_beam(enc_out, kBeamWidth);

            // 5d. Convert IDs → OmrTokens and append.
            auto tokens = eng->parser.decode_ids(ids);
            for (auto& t : tokens)
                res.tokens.push_back(std::move(t));
        }

        // Insert a staff-separator barline between different staff groups.
        // (Only if more than one staff is present.)
        if (&staff != &staffs.back())
            res.tokens.push_back({eng->parser.token_to_id("barline"), "barline"});
    }

    res.success = true;
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
//  C API implementation (Dart FFI entry points)
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

OmrEngineHandle omr_create(const char* segnet_path,
                            const char* encoder_path,
                            const char* decoder_path,
                            const char* tokenizer_path)
{
    try {
        return new OmrEngine(segnet_path, encoder_path, decoder_path, tokenizer_path);
    } catch (const std::exception& e) {
        return nullptr;
    }
}

OmrResultC* omr_process(OmrEngineHandle handle,
                          const uint8_t*  image_data,
                          int32_t         image_len)
{
    auto* result = new OmrResultC;
    std::memset(result, 0, sizeof(OmrResultC));

    if (!handle) {
        result->success = 0;
        std::strncpy(result->error, "null engine handle", sizeof(result->error) - 1);
        return result;
    }

    try {
        auto* eng = reinterpret_cast<OmrEngine*>(handle);
        omr::OmrResult r = run_pipeline(eng, image_data, image_len);

        result->success     = r.success ? 1 : 0;
        result->token_count = static_cast<int32_t>(r.tokens.size());

        if (!r.success) {
            std::strncpy(result->error, r.error.c_str(), sizeof(result->error) - 1);
            result->tokens = nullptr;
            return result;
        }

        // Allocate token array.
        result->tokens = new OmrTokenC[result->token_count];
        for (int i = 0; i < result->token_count; ++i) {
            result->tokens[i].id = r.tokens[i].id;
            std::strncpy(result->tokens[i].text,
                         r.tokens[i].text.c_str(),
                         sizeof(OmrTokenC::text) - 1);
            result->tokens[i].text[sizeof(OmrTokenC::text) - 1] = '\0';
        }
    } catch (const std::exception& e) {
        result->success = 0;
        std::strncpy(result->error, e.what(), sizeof(result->error) - 1);
        result->tokens      = nullptr;
        result->token_count = 0;
    }
    return result;
}

void omr_free_result(OmrResultC* result) {
    if (!result) return;
    delete[] result->tokens;
    delete result;
}

void omr_destroy(OmrEngineHandle handle) {
    delete reinterpret_cast<OmrEngine*>(handle);
}

} // extern "C"
