#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace omr {

/**
 * StaffDetector – locates 5-line staff groups in a segmentation mask.
 *
 * Algorithm (all standard signal-processing techniques):
 *   1. Horizontal projection: sum each row of the staff-line mask.
 *      Rows containing staff lines produce high values.
 *   2. Local-maxima peak finding with minimum-distance constraint.
 *      Ref: local_maxima / find_peaks (scipy-style, implemented from scratch)
 *   3. Group adjacent peaks into sets of 5 (one staff group).
 *   4. Sliding-window refinement: re-fit each line position locally.
 *
 * Tuning constants derived from typical engraved sheet music at 1920 px width.
 */
class StaffDetector {
public:
    // Minimum and maximum staff line spacing in pixels (at 1920 px width).
    static constexpr float MIN_UNIT = 11.0f;
    static constexpr float MAX_UNIT = 60.0f;

    // Width of the sliding window used for local line-position refinement.
    // Auto-set to max(40, image_width / 25) during detect().
    static constexpr int WIN_DIV = 25;
    static constexpr int WIN_MIN = 40;

    // Reject a staff group if the std-dev of line gaps exceeds this fraction
    // of the mean gap.
    static constexpr float MAX_DEV_RATIO = 0.6f;

    /**
     * Detect all staff groups in the image.
     * @param staff_mask  Float map in [0,1], same size as the original image,
     *                    corresponding to SEG_STAFF_LINE output.
     * @return            Detected staff groups, ordered top-to-bottom.
     *                    Empty if no valid group found.
     */
    std::vector<StaffGroup> detect(const cv::Mat& staff_mask) const;

private:
    // Step 1: horizontal projection of the staff-line probability map.
    std::vector<float> horizontal_projection(const cv::Mat& mask) const;

    // Step 2: find local maxima with minimum spacing = min_unit.
    std::vector<int> find_peaks(const std::vector<float>& proj,
                                float min_dist, float threshold = 0.1f) const;

    // Step 3: group a sorted peak list into groups of 5.
    std::vector<std::vector<int>> group_peaks(const std::vector<int>& peaks,
                                               float min_unit,
                                               float max_unit) const;

    // Step 4: refine y positions using a sliding window on the local mask.
    void refine_staff(StaffGroup& sg, const cv::Mat& mask) const;
};

} // namespace omr
