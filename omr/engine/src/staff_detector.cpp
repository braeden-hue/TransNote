#include "staff_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

std::vector<StaffGroup> StaffDetector::detect(const cv::Mat& staff_mask) const {
    // Step 1: project the staff-line probability map onto the vertical axis.
    auto proj = horizontal_projection(staff_mask);

    // Step 2: find local maxima (candidate staff-line rows).
    // A staff line is at least MIN_UNIT pixels from the next one.
    auto peaks = find_peaks(proj, MIN_UNIT / 2.f, /*threshold=*/0.05f);
    if (peaks.size() < 5) return {};

    // Step 3: group 5 consecutive peaks separated by MIN_UNIT..MAX_UNIT.
    auto groups = group_peaks(peaks, MIN_UNIT, MAX_UNIT);

    // Step 4: build StaffGroup structs and refine with sliding window.
    std::vector<StaffGroup> result;
    result.reserve(groups.size());

    for (const auto& g : groups) {
        StaffGroup sg;
        sg.x_start   = 0.f;
        sg.x_end     = static_cast<float>(staff_mask.cols - 1);
        sg.unit_size = 0.f;

        for (int i = 0; i < 5; ++i)
            sg.y_lines[i] = static_cast<float>(g[i]);

        // Compute mean gap (unit_size).
        float gap_sum = 0.f;
        for (int i = 0; i < 4; ++i)
            gap_sum += sg.y_lines[i + 1] - sg.y_lines[i];
        sg.unit_size = gap_sum / 4.f;

        // Sanity check: reject if unit_size is outside expected range.
        if (sg.unit_size < MIN_UNIT || sg.unit_size > MAX_UNIT) continue;

        // Reject if gap std-dev is too large (non-uniform spacing = false positive).
        float mean_gap = sg.unit_size;
        float var = 0.f;
        for (int i = 0; i < 4; ++i) {
            float d = (sg.y_lines[i + 1] - sg.y_lines[i]) - mean_gap;
            var += d * d;
        }
        float stddev = std::sqrt(var / 4.f);
        if (stddev > MAX_DEV_RATIO * mean_gap) continue;

        // Local refinement.
        refine_staff(sg, staff_mask);

        result.push_back(sg);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 1: horizontal projection
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> StaffDetector::horizontal_projection(const cv::Mat& mask) const {
    // Sum each row; normalise by width so values are in [0,1].
    std::vector<float> proj(mask.rows, 0.f);
    const float inv_w = 1.f / mask.cols;

    for (int r = 0; r < mask.rows; ++r) {
        const float* row = mask.ptr<float>(r);
        float sum = 0.f;
        for (int c = 0; c < mask.cols; ++c) sum += row[c];
        proj[r] = sum * inv_w;
    }
    return proj;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 2: local-maxima peak finding
//
//  Algorithm: scan the projection, collect all rows that are local maxima
//  above `threshold`, then apply non-maximum suppression with spacing `min_dist`.
//  This is a straightforward signal-processing technique with no IP issues.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int> StaffDetector::find_peaks(const std::vector<float>& proj,
                                            float min_dist,
                                            float threshold) const {
    const int N = static_cast<int>(proj.size());
    std::vector<std::pair<float, int>> candidates; // (value, row)

    for (int i = 1; i < N - 1; ++i) {
        if (proj[i] > threshold &&
            proj[i] >= proj[i - 1] &&
            proj[i] >= proj[i + 1])
        {
            candidates.emplace_back(proj[i], i);
        }
    }

    // Sort by value descending, then apply NMS.
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    std::vector<int> peaks;
    std::vector<bool> suppressed(N, false);

    for (auto& [val, row] : candidates) {
        if (suppressed[row]) continue;
        peaks.push_back(row);
        // Suppress neighbours within min_dist.
        for (int j = std::max(0, (int)(row - min_dist));
             j <= std::min(N - 1, (int)(row + min_dist)); ++j)
        {
            suppressed[j] = true;
        }
    }

    std::sort(peaks.begin(), peaks.end()); // restore top→bottom order
    return peaks;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 3: group peaks into sets of 5
//
//  Sliding window: for each starting peak, check if the next 4 peaks are
//  spaced within [min_unit, max_unit] pixels apart.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<int>> StaffDetector::group_peaks(
        const std::vector<int>& peaks,
        float min_unit, float max_unit) const
{
    std::vector<std::vector<int>> groups;

    for (int i = 0; i + 4 < static_cast<int>(peaks.size()); ++i) {
        bool valid = true;
        for (int k = 0; k < 4; ++k) {
            float gap = static_cast<float>(peaks[i + k + 1] - peaks[i + k]);
            if (gap < min_unit || gap > max_unit) { valid = false; break; }
        }
        if (valid) {
            groups.push_back({peaks[i], peaks[i+1], peaks[i+2], peaks[i+3], peaks[i+4]});
            i += 4; // skip ahead to avoid overlapping groups
        }
    }
    return groups;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 4: sliding-window refinement
//
//  For each staff line, scan a narrow horizontal band (±0.5 unit_size) and
//  find the row with the maximum projection value.
// ─────────────────────────────────────────────────────────────────────────────

void StaffDetector::refine_staff(StaffGroup& sg, const cv::Mat& mask) const {
    const float search_r = sg.unit_size * 0.5f;
    const int W = mask.cols;
    const int win = std::max(WIN_MIN, W / WIN_DIV);

    for (int line = 0; line < 5; ++line) {
        float best_y = sg.y_lines[line];
        float best_score = -1.f;

        int y_min = std::max(0, (int)(sg.y_lines[line] - search_r));
        int y_max = std::min(mask.rows - 1, (int)(sg.y_lines[line] + search_r));

        for (int y = y_min; y <= y_max; ++y) {
            // Average mask value in a central horizontal window.
            int x0 = std::max(0, W / 2 - win / 2);
            int x1 = std::min(W - 1, x0 + win);
            const float* row = mask.ptr<float>(y);
            float sum = 0.f;
            for (int x = x0; x <= x1; ++x) sum += row[x];
            float score = sum / (x1 - x0 + 1);
            if (score > best_score) { best_score = score; best_y = (float)y; }
        }
        sg.y_lines[line] = best_y;
    }

    // Recompute unit_size after refinement.
    float gap_sum = 0.f;
    for (int i = 0; i < 4; ++i)
        gap_sum += sg.y_lines[i + 1] - sg.y_lines[i];
    sg.unit_size = gap_sum / 4.f;
}

} // namespace omr
