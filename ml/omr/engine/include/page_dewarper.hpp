#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace omr {

/**
 * PageDewarper – corrects page curvature (book-spine bow, hand-held flex) using
 * the staff-line segmentation mask as ground truth for actual line positions.
 *
 * Algorithm (column-by-column rubber-sheet warp):
 *   1. For each detected staff group, trace the actual y-position of each of the
 *      5 staff lines at regular column intervals by finding the peak of the
 *      staff-line probability mask within a search band.
 *   2. Compute "ideal" y-positions: where the lines should be if the page were
 *      perfectly flat (y_ideal[i] = y_lines[0] + i × unit_size).
 *   3. Build a dense vertical displacement map (map_y) by piecewise-linear
 *      interpolation between the traced and ideal y-positions at each column.
 *      Above and below the staff, the mapping is linearly extrapolated.
 *   4. Apply cv::remap() to produce a dewarped image where all staff lines are
 *      straight and equally spaced.
 *
 * Ref: Liang J. et al. (2008) "Geometric Rectification of Camera-Captured
 *      Document Images." IEEE TPAMI 30(4):591–605. (column-wise warp, §III-B)
 * Implementation: cv::remap (OpenCV Apache-2.0). Pure linear algebra — no IP.
 *
 * When multiple staff groups are present, each updates its own region of the
 * shared displacement map; areas between staffs use identity (no correction).
 */
class PageDewarper {
public:
    // Column sampling interval for tracing staff lines (pixels).
    // Smaller = more accurate curve fit, slightly slower.
    static constexpr int   TRACE_STEP_PX  = 16;

    // Half-width of the y-search window around each expected line position,
    // expressed as a multiple of the inter-line spacing (unit_size).
    static constexpr float SEARCH_HALF    = 0.65f;

    // Minimum mask value at a column to accept that sample as valid.
    static constexpr float MIN_MASK_VAL   = 0.05f;

    // Skip a staff group if fewer than this many valid column samples were found.
    static constexpr int   MIN_SAMPLES    = 8;

    // Vertical correction region: this many unit_sizes above / below the staff.
    static constexpr float MARGIN_UNITS   = 2.0f;

    /**
     * Dewarp the image using all detected staff groups as control anchors.
     *
     * @param gray       Preprocessed grayscale image (CV_8UC1).
     * @param staff_mask Float [0,1] probability map, SEG_STAFF_LINE channel,
     *                   same spatial size as gray.
     * @param staffs     Staff groups detected on the (not-yet-dewarped) image.
     *                   y_lines are updated in-place to ideal flat positions.
     * @return           Dewarped CV_8UC1 image.  Returns gray unchanged if
     *                   no staff group has enough samples.
     */
    cv::Mat dewarp(const cv::Mat&           gray,
                   const cv::Mat&           staff_mask,
                   std::vector<StaffGroup>& staffs) const;

private:
    // Per-column traced y positions for one staff's 5 lines.
    struct ColumnSample {
        int   x;
        float y[5];
    };

    // Trace actual line y-positions column by column using the mask.
    std::vector<ColumnSample> trace_lines(const cv::Mat&    staff_mask,
                                          const StaffGroup& sg) const;

    // Write per-pixel vertical displacement into map_y for one staff's region.
    void update_remap_region(const cv::Mat&                   gray,
                             const std::vector<ColumnSample>& samples,
                             const StaffGroup&                sg,
                             cv::Mat&                         map_y) const;

    // After dewarping, set y_lines[1..4] to ideal equally-spaced positions.
    static void flatten_staff(StaffGroup& sg);
};

} // namespace omr
