#include "noise_filter.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat NoiseFilter::filter(const cv::Mat& gray) const {
    cv::Mat result = normalize_illumination(gray);
    result = bilateral_smooth(result);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stage 1: illumination normalisation
//
//  Sheet music photos taken with a smartphone often have a brightness gradient:
//  the flash or window light brightens one region while the rest is dimmer.
//  CLAHE corrects this locally, but a global division by the background removes
//  the large-scale gradient BEFORE local equalisation, improving the result.
//
//  Algorithm (flat-field correction):
//    1. Downsample the image to BG_DS_WIDTH pixels wide.
//    2. Apply a large Gaussian blur – at this resolution the blur completely
//       averages out ink strokes, leaving only the illumination envelope.
//    3. Upsample back → background estimate B(x,y).
//    4. Normalised pixel = clip( I(x,y) / B(x,y) × μ_B, 0, 255 )
//       where μ_B is the mean background brightness (preserves overall tone).
//
//  Ref: flat-field / shading correction, standard imaging technique (no IP).
//  Implementation: cv::GaussianBlur + cv::divide (OpenCV Apache-2.0).
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat NoiseFilter::normalize_illumination(const cv::Mat& gray) const {
    // 1. Downsample.
    const double scale = (double)BG_DS_WIDTH / gray.cols;
    cv::Mat small;
    cv::resize(gray, small, {}, scale, scale, cv::INTER_AREA);

    // 2. Gaussian blur on small image (approximates very large sigma on original).
    cv::Mat blurred;
    cv::GaussianBlur(small, blurred, {BG_BLUR_KSIZE, BG_BLUR_KSIZE},
                     BG_BLUR_SIGMA, BG_BLUR_SIGMA);

    // 3. Upsample background estimate back to original resolution.
    cv::Mat background;
    cv::resize(blurred, background, gray.size(), 0, 0, cv::INTER_LINEAR);

    // 4. Divide original by background; scale so mean background maps to 200.
    //    Using float arithmetic to prevent integer overflow.
    cv::Mat gray_f, bg_f;
    gray.convertTo(gray_f, CV_32F);
    background.convertTo(bg_f, CV_32F);

    const float mean_bg = static_cast<float>(cv::mean(background)[0]);
    const float target  = 200.f;  // output mean brightness for white paper

    cv::Mat result_f;
    // result = (gray / background) × target
    cv::divide(gray_f, bg_f, result_f, target);

    cv::Mat result;
    result_f.convertTo(result, CV_8U);  // automatically clamps [0,255]
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stage 2: edge-preserving bilateral filter
//
//  Bilateral filter: each output pixel is a weighted average of neighbours.
//  Weights decrease with spatial distance AND with intensity difference.
//  → Homogeneous regions (paper) are smoothed; edges (ink strokes) are kept.
//
//  Parameters (tuned for 1920 px wide sheet music at ~150 DPI):
//    d=9:    Examine a 9×9 neighbourhood (moderately large for detail retention).
//    σ_c=20: Pixels differing by >20 grey levels are treated as separate regions.
//    σ_s=7:  Spatial Gaussian half-width; keeps influence local.
//
//  Ref: Tomasi & Manduchi (1998) ICCV pp. 839–846.
//  Implementation: cv::bilateralFilter (OpenCV Apache-2.0).
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat NoiseFilter::bilateral_smooth(const cv::Mat& gray) const {
    cv::Mat out;
    cv::bilateralFilter(gray, out, BILATERAL_D, BILATERAL_SIG_C, BILATERAL_SIG_S);
    return out;
}

} // namespace omr
