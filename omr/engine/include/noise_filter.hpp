#pragma once
#include <opencv2/core.hpp>

namespace omr {

/**
 * NoiseFilter – two-stage noise reduction for smartphone-captured sheet music.
 *
 * Stage 1 – illumination normalisation:
 *   Estimates the broad illumination background by downsampling the image,
 *   applying a Gaussian blur, then upsampling.  Dividing the original by this
 *   background removes uneven lighting (shadows, flash fall-off) without
 *   affecting local contrast.
 *   Ref: flat-field correction, standard imaging technique (no IP restriction).
 *   Implementation: cv::GaussianBlur + cv::divide (OpenCV Apache-2.0).
 *
 * Stage 2 – edge-preserving denoising:
 *   Bilateral filter smooths Gaussian / JPEG-compression noise while preserving
 *   the hard edges of staff lines and noteheads.
 *   Ref: Tomasi C. & Manduchi R. (1998) "Bilateral Filtering for Gray and Color
 *        Images." Proc. ICCV pp. 839–846.
 *   Implementation: cv::bilateralFilter (OpenCV Apache-2.0).
 *
 * Applied after CLAHE so that the filter operates on contrast-normalised input.
 */
class NoiseFilter {
public:
    // Background estimation: downsampled width used for illumination estimation.
    static constexpr int    BG_DS_WIDTH     = 192;   // pixels
    static constexpr int    BG_BLUR_KSIZE   = 31;    // kernel size on downsampled image
    static constexpr double BG_BLUR_SIGMA   = 10.0;

    // Bilateral filter parameters.
    static constexpr int    BILATERAL_D      = 9;    // neighbourhood diameter
    static constexpr double BILATERAL_SIG_C  = 20.0; // intensity sigma
    static constexpr double BILATERAL_SIG_S  = 7.0;  // spatial sigma

    /**
     * Apply illumination normalisation followed by bilateral denoising.
     * @param gray  CV_8UC1 grayscale, already CLAHE-processed.
     * @return      Filtered CV_8UC1 image (same size as input).
     */
    cv::Mat filter(const cv::Mat& gray) const;

private:
    cv::Mat normalize_illumination(const cv::Mat& gray) const;
    cv::Mat bilateral_smooth(const cv::Mat& gray) const;
};

} // namespace omr
