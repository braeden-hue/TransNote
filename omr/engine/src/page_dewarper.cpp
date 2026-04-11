#include "page_dewarper.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat PageDewarper::dewarp(const cv::Mat&           gray,
                              const cv::Mat&           staff_mask,
                              std::vector<StaffGroup>& staffs) const
{
    if (staffs.empty()) return gray;

    // Initialise remap fields as identity (source pixel = destination pixel).
    // map_y[y][x] = y  →  no vertical displacement.
    // map_x is not modified (horizontal lines only bow vertically).
    cv::Mat map_x(gray.size(), CV_32F);
    cv::Mat map_y(gray.size(), CV_32F);
    for (int y = 0; y < gray.rows; ++y) {
        float* rx = map_x.ptr<float>(y);
        float* ry = map_y.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            rx[x] = (float)x;
            ry[x] = (float)y;
        }
    }

    bool any_corrected = false;

    for (auto& sg : staffs) {
        // Trace actual line positions column by column.
        auto samples = trace_lines(staff_mask, sg);
        if ((int)samples.size() < MIN_SAMPLES) continue;

        // Update the remap field in this staff's vertical region.
        update_remap_region(gray, samples, sg, map_y);

        // Update the StaffGroup so downstream canvas building uses flat coords.
        flatten_staff(sg);
        any_corrected = true;
    }

    if (!any_corrected) return gray;

    // Apply the combined displacement field to the grayscale image.
    cv::Mat dewarped;
    cv::remap(gray, dewarped, map_x, map_y,
              cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
    return dewarped;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 1: trace actual y-positions of the 5 staff lines column by column
//
//  For each sample column x, and for each of the 5 lines:
//    - Define a search band of ±SEARCH_HALF × unit_size around the global y.
//    - Find the row with the maximum staff-line mask value in that band.
//    - Accept the sample only if the peak exceeds MIN_MASK_VAL (evidence of
//      a real staff line pixel — not just noise).
//  If any of the 5 lines cannot be traced at a given x, that column is skipped.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<PageDewarper::ColumnSample>
PageDewarper::trace_lines(const cv::Mat&    staff_mask,
                           const StaffGroup& sg) const
{
    std::vector<ColumnSample> samples;
    const int W = staff_mask.cols;
    const int H = staff_mask.rows;
    const float half = sg.unit_size * SEARCH_HALF;

    for (int x = 0; x < W; x += TRACE_STEP_PX) {
        ColumnSample cs;
        cs.x = x;
        bool valid = true;

        for (int line = 0; line < 5; ++line) {
            const int y_min = std::max(0,   (int)(sg.y_lines[line] - half));
            const int y_max = std::min(H-1, (int)(sg.y_lines[line] + half));

            float best_val = -1.f;
            float best_y   = sg.y_lines[line];

            // Single-column peak search — O(search_band) per sample.
            for (int y = y_min; y <= y_max; ++y) {
                const float val = staff_mask.at<float>(y, x);
                if (val > best_val) { best_val = val; best_y = (float)y; }
            }

            if (best_val < MIN_MASK_VAL) { valid = false; break; }
            cs.y[line] = best_y;
        }

        if (valid) samples.push_back(cs);
    }
    return samples;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 2: build vertical displacement in map_y for one staff's region
//
//  For each destination column x in [samples.front().x, samples.back().x]:
//    - Linearly interpolate the 5 traced y-positions from the two nearest
//      column samples (bracketing interpolation).
//    - For each destination row y_dst in the correction region:
//        * If y_dst is between ideal lines i and i+1: piecewise-linear map.
//        * If y_dst is above line 0 or below line 4: linear extrapolation.
//
//  The "ideal" y positions are: y_ideal[i] = y_lines[0] + i × unit_size.
//  After remapping, each staff line will appear at exactly its ideal y.
//
//  Ref: column-wise rubber-sheet warp — Liang et al. (2008) §III-B.
// ─────────────────────────────────────────────────────────────────────────────

void PageDewarper::update_remap_region(const cv::Mat&                   gray,
                                        const std::vector<ColumnSample>& samples,
                                        const StaffGroup&                sg,
                                        cv::Mat&                         map_y) const
{
    if (samples.empty()) return;

    // Ideal (flat) y positions for this staff.
    float y_ideal[5];
    y_ideal[0] = sg.y_lines[0];
    for (int i = 1; i < 5; ++i)
        y_ideal[i] = y_ideal[0] + i * sg.unit_size;

    // Vertical correction region (includes a margin above and below the staff).
    const float margin = sg.unit_size * MARGIN_UNITS;
    const int y_top = std::max(0,             (int)(y_ideal[0] - margin));
    const int y_bot = std::min(gray.rows - 1, (int)(y_ideal[4] + margin));

    // Horizontal range covered by the traced samples.
    const int x_lo = samples.front().x;
    const int x_hi = samples.back().x;

    // We use a moving pointer into the samples array to find the bracket for
    // each column without restarting the search from the beginning each time.
    int lo_idx = 0;

    for (int x = x_lo; x <= x_hi; ++x) {
        // Advance lo_idx so that samples[lo_idx].x <= x < samples[lo_idx+1].x.
        while (lo_idx + 1 < (int)samples.size() - 1 &&
               samples[lo_idx + 1].x <= x)
        {
            ++lo_idx;
        }
        const int hi_idx = lo_idx + 1;
        if (hi_idx >= (int)samples.size()) break;

        // Interpolation weight t ∈ [0,1].
        const int   dx = samples[hi_idx].x - samples[lo_idx].x;
        const float t  = (dx <= 0) ? 0.f
                                   : (float)(x - samples[lo_idx].x) / (float)dx;

        // Interpolated traced y positions at column x.
        float yt[5];
        for (int line = 0; line < 5; ++line)
            yt[line] = samples[lo_idx].y[line] * (1.f - t)
                     + samples[hi_idx].y[line] * t;

        // For each destination row in the correction region, compute source row.
        float* row_map = nullptr;
        for (int y_dst = y_top; y_dst <= y_bot; ++y_dst) {
            const float dy = (float)y_dst;
            float src_y;

            if (dy <= y_ideal[0]) {
                // Above the first line: linear extrapolation from lines 0-1.
                const float denom = std::max(1.f, y_ideal[1] - y_ideal[0]);
                const float slope = (yt[1] - yt[0]) / denom;
                src_y = yt[0] + slope * (dy - y_ideal[0]);

            } else if (dy >= y_ideal[4]) {
                // Below the last line: linear extrapolation from lines 3-4.
                const float denom = std::max(1.f, y_ideal[4] - y_ideal[3]);
                const float slope = (yt[4] - yt[3]) / denom;
                src_y = yt[4] + slope * (dy - y_ideal[4]);

            } else {
                // Between lines: find enclosing pair i, i+1.
                int i = 0;
                while (i < 3 && y_ideal[i + 1] <= dy) ++i;
                const float denom = std::max(1.f, y_ideal[i+1] - y_ideal[i]);
                const float alpha = (dy - y_ideal[i]) / denom;
                src_y = yt[i] + alpha * (yt[i+1] - yt[i]);
            }

            src_y = std::max(0.f, std::min((float)(gray.rows - 1), src_y));
            map_y.at<float>(y_dst, x) = src_y;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 3: update StaffGroup to reflect post-dewarp (flat) line positions
// ─────────────────────────────────────────────────────────────────────────────

void PageDewarper::flatten_staff(StaffGroup& sg) {
    // After dewarping, each line i sits at y_lines[0] + i × unit_size.
    // y_lines[0] and unit_size are unchanged; only lines 1-4 are corrected.
    for (int i = 1; i < 5; ++i)
        sg.y_lines[i] = sg.y_lines[0] + i * sg.unit_size;
}

} // namespace omr
