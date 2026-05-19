#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <array>

namespace omr {

// ─── Segmentation class indices ──────────────────────────────────────────────
// Matches the 6-class output of the segnet model.
// Class 0 is background; classes 1-5 are music symbol types.
enum SegClass : int {
    SEG_BACKGROUND = 0,
    SEG_STEM_REST  = 1,   // stems and rests
    SEG_NOTEHEAD   = 2,   // filled and open noteheads
    SEG_CLEF_KEY   = 3,   // clef glyphs and key-signature accidentals
    SEG_STAFF_LINE = 4,   // the five staff lines themselves
    SEG_SYMBOL     = 5,   // bar lines, time signatures, other symbols
    SEG_NUM_CLASSES = 6
};

// ─── One staff group (5 lines) detected in the image ─────────────────────────
struct StaffGroup {
    float y_lines[5];   // pixel y of each of the 5 staff lines (top→bottom)
    float unit_size;    // average gap between adjacent lines (px)
    float x_start;      // leftmost x of valid staff content
    float x_end;        // rightmost x of valid staff content
};

// ─── A tile ready for the encoder ────────────────────────────────────────────
// Fixed canvas size that the encoder expects.
static constexpr int CANVAS_H = 256;
static constexpr int CANVAS_W = 1280;

struct StaffCanvas {
    uint8_t pixels[CANVAS_H][CANVAS_W];  // grayscale, range [0,255]
};

// ─── One decoded token from the unified DeepScore vocabulary ─────────────────
struct OmrToken {
    int32_t     id;     // index in tokenizer.json  (0 = PAD, 1 = SOS, 2 = EOS)
    std::string text;   // human-readable, e.g. "note-C4-1/4", "rest-1/8"
};

// ─── Final pipeline result ────────────────────────────────────────────────────
struct OmrResult {
    bool                    success;
    std::string             error;
    std::vector<OmrToken>   tokens; // full decoded token sequence (excl. SOS/EOS/PAD)
};

// ─── Model paths bundle ───────────────────────────────────────────────────────
struct ModelPaths {
    std::string segnet;      // e.g. "assets/segnet.tflite"
    std::string encoder;     // e.g. "assets/encoder.tflite"
    std::string decoder;     // e.g. "assets/decoder.tflite"
    std::string tokenizer;   // e.g. "assets/tokenizer.json"
};

} // namespace omr
