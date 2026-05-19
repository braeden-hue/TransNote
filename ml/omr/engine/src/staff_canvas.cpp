#include "staff_canvas.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

std::vector<cv::Mat> StaffCanvas::build_tiles(const cv::Mat&    gray,
                                               const StaffGroup& staff) const {
    cv::Mat region  = crop_staff_region(gray, staff);
    cv::Mat canvas  = scale_to_canvas_height(region);
    auto    tiles   = tile_horizontal(canvas);
    return tiles;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 1: crop staff region (including vertical margin)
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat StaffCanvas::crop_staff_region(const cv::Mat&    gray,
                                        const StaffGroup& staff) const {
    // Extend the crop region by MARGIN_UNITS above and below.
    float margin = staff.unit_size * MARGIN_UNITS;

    int y_top = std::max(0, (int)std::floor(staff.y_lines[0] - margin));
    int y_bot = std::min(gray.rows - 1,
                         (int)std::ceil(staff.y_lines[4] + margin));

    int x_left  = std::max(0,              (int)staff.x_start);
    int x_right = std::min(gray.cols - 1,  (int)staff.x_end);

    cv::Rect roi(x_left, y_top, x_right - x_left + 1, y_bot - y_top + 1);
    return gray(roi).clone();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 2: scale height to exactly CANVAS_H, keep aspect ratio
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat StaffCanvas::scale_to_canvas_height(const cv::Mat& strip) const {
    if (strip.rows == CANVAS_H) return strip;

    double scale = (double)CANVAS_H / strip.rows;
    int new_w    = (int)std::round(strip.cols * scale);

    cv::Mat out;
    cv::resize(strip, out, {new_w, CANVAS_H}, 0, 0,
               scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 3: tile into CANVAS_W-wide strips with TILE_OVERLAP overlap
//
//  Each tile is exactly CANVAS_H × CANVAS_W (white-padded on the right
//  when the strip is shorter than CANVAS_W).
// ─────────────────────────────────────────────────────────────────────────────

std::vector<cv::Mat> StaffCanvas::tile_horizontal(const cv::Mat& strip) const {
    std::vector<cv::Mat> tiles;
    const int W = strip.cols;

    const int step = CANVAS_W - TILE_OVERLAP;  // advance per tile (1216 px)

    for (int x = 0; x < W; x += step) {
        // Create a white canvas.
        cv::Mat tile(CANVAS_H, CANVAS_W, CV_8UC1, cv::Scalar(255));

        // Copy whatever portion of the strip fits.
        cv::Rect src_roi(x, 0, std::min(CANVAS_W, W - x), CANVAS_H);
        cv::Rect dst_roi(0, 0, src_roi.width,              CANVAS_H);
        strip(src_roi).copyTo(tile(dst_roi));

        tiles.push_back(tile);
        if (x + CANVAS_W >= W) break;  // last tile consumed all remaining content
    }

    // Always produce at least one tile.
    if (tiles.empty()) {
        cv::Mat tile(CANVAS_H, CANVAS_W, CV_8UC1, cv::Scalar(255));
        cv::Rect src_roi(0, 0, std::min(CANVAS_W, W), CANVAS_H);
        strip(src_roi).copyTo(tile(cv::Rect(0, 0, src_roi.width, CANVAS_H)));
        tiles.push_back(tile);
    }
    return tiles;
}

} // namespace omr
