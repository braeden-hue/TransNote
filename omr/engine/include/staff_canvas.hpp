#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace omr {

/**
 * StaffCanvas – crops and tiles one staff group into fixed-size encoder inputs.
 *
 * Each StaffGroup is:
 *   1. Cropped from the full grayscale image with a vertical margin.
 *   2. Rescaled so that the staff height maps to CANVAS_H pixels.
 *   3. Tiled horizontally into CANVAS_W-wide strips with TILE_OVERLAP overlap.
 *
 * The encoder expects exactly CANVAS_H × CANVAS_W grayscale uint8 inputs.
 * The tile size and overlap are the same values used during model training.
 */
class StaffCanvas {
public:
    // Vertical margin added above/below the outermost staff lines (in unit_size units).
    static constexpr float MARGIN_UNITS = 2.0f;

    // Horizontal overlap between adjacent tiles (pixels, in canvas space).
    static constexpr int TILE_OVERLAP = 64;

    /**
     * Build all canvas tiles for one staff group.
     * @param gray   Full preprocessed grayscale image (same size as segmentation input).
     * @param staff  Detected staff group.
     * @return       Vector of CANVAS_H × CANVAS_W grayscale tiles (CV_8UC1).
     */
    std::vector<cv::Mat> build_tiles(const cv::Mat& gray,
                                      const StaffGroup& staff) const;

private:
    // Crop the staff region from the full image (including vertical margin).
    cv::Mat crop_staff_region(const cv::Mat& gray,
                               const StaffGroup& staff) const;

    // Scale so staff fits exactly into CANVAS_H pixels height.
    cv::Mat scale_to_canvas_height(const cv::Mat& strip) const;

    // Slice a full-width strip into CANVAS_W tiles with TILE_OVERLAP overlap.
    std::vector<cv::Mat> tile_horizontal(const cv::Mat& strip) const;
};

} // namespace omr
