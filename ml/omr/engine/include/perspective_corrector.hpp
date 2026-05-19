#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace omr {

/**
 * PerspectiveCorrector – two-stage geometric correction for smartphone photos.
 *
 * Stage 1 (preferred): detect the four corners of the sheet music page and apply
 *   a projective (homography) warp to produce a fronto-parallel view.
 *   Algorithm: Otsu threshold → largest contour → convex hull → approxPolyDP.
 *   Ref: Forsyth & Ponce (2003) "Computer Vision: A Modern Approach" §2.1.
 *   Implementation: cv::getPerspectiveTransform / cv::warpPerspective (Apache-2.0).
 *
 * Stage 2 (fallback): when no clean quadrilateral is found, estimate the dominant
 *   skew angle from near-horizontal Hough lines and rotate to level.
 *   Ref: Matas J. et al. (2000) "Robust Detection of Lines Using the Progressive
 *        Probabilistic Hough Transform." CVIU 78(1):119–137.
 *   Implementation: cv::HoughLinesP (OpenCV Apache-2.0).
 *
 * Both stages use only OpenCV (Apache-2.0). No third-party library required.
 */
class PerspectiveCorrector {
public:
    // Minimum fraction of image area a candidate page quad must cover.
    // Images where the page fills <20 % of the frame skip Stage 1.
    static constexpr double MIN_PAGE_AREA_RATIO  = 0.20;

    // Maximum skew angle corrected by the Stage 2 deskew.
    // Larger rotations are unlikely to be pure camera tilt.
    static constexpr double MAX_DESKEW_ANGLE_DEG = 15.0;

    // Minimum normalised line length accepted by HoughLinesP (Stage 2).
    static constexpr double MIN_LINE_FRAC = 0.30;

    /**
     * Correct perspective / skew.
     * @param gray  Single-channel grayscale input (any size).
     * @return      Corrected grayscale image (may differ in size from input).
     *              Returns input unchanged if no meaningful correction is found.
     */
    cv::Mat correct(const cv::Mat& gray) const;

private:
    // Detect the sheet boundary as a quadrilateral.
    // Returns 4 corners in order [TL, TR, BR, BL], or empty on failure.
    std::vector<cv::Point2f> detect_page_quad(const cv::Mat& gray) const;

    // Warp a 4-corner quad to an upright rectangle.
    cv::Mat warp_quad(const cv::Mat& gray,
                      const std::vector<cv::Point2f>& corners) const;

    // Estimate and correct rotation-only skew using Probabilistic Hough.
    cv::Mat deskew(const cv::Mat& gray) const;

    // Re-order 4 arbitrary points to [TL, TR, BR, BL].
    static std::vector<cv::Point2f> order_points(std::vector<cv::Point2f> pts);
};

} // namespace omr
