#pragma once
#include "perspective_corrector.hpp"
#include "noise_filter.hpp"
#include <opencv2/core.hpp>

namespace omr {

/**
 * Preprocessor – five-stage image conditioning pipeline for smartphone photos.
 *
 * Stages (all standard algorithms, OpenCV Apache-2.0):
 *  1. perspective_correct – deskew / four-corner homography warp.
 *     Ref: Matas et al. (2000) CVIU; Forsyth & Ponce (2003).
 *  2. autocrop            – largest-contour bounding-box crop.
 *     Ref: Otsu N. (1979) IEEE Trans. SMC.
 *  3. resize              – bilinear rescaling to TARGET_WIDTH.
 *  4. apply_clahe         – local contrast normalisation.
 *     Ref: Zuiderveld K. (1994) Graphics Gems IV.
 *  5. noise_filter        – illumination normalisation + bilateral denoising.
 *     Ref: Tomasi & Manduchi (1998) ICCV.
 */
class Preprocessor {
public:
    // Target width after resize (pixels).
    static constexpr int TARGET_WIDTH = 1920;

    // CLAHE parameters (tuned for printed/scanned sheet music).
    static constexpr double CLAHE_CLIP = 1.0;
    static constexpr int    CLAHE_TILE = 8;

    /**
     * Full pipeline: BGR image → preprocessed grayscale.
     * Safe to call from any thread (all sub-components are stateless).
     */
    cv::Mat process(const cv::Mat& bgr_in) const;

    // Individual stages (public for unit testing).
    cv::Mat autocrop(const cv::Mat& gray) const;
    cv::Mat resize_to_width(const cv::Mat& src, int target = TARGET_WIDTH) const;
    cv::Mat apply_clahe(const cv::Mat& gray) const;

private:
    PerspectiveCorrector perspective_corrector_;
    NoiseFilter          noise_filter_;
};

} // namespace omr
