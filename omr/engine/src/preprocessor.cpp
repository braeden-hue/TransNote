#include "preprocessor.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Full pipeline
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat Preprocessor::process(const cv::Mat& bgr_in) const {
    // 0. Convert to single-channel grayscale.
    cv::Mat gray;
    if (bgr_in.channels() == 3)
        cv::cvtColor(bgr_in, gray, cv::COLOR_BGR2GRAY);
    else
        gray = bgr_in.clone();

    // 1. Correct camera tilt and perspective before cropping.
    //    Works best on the full (uncropped) image where page corners are visible.
    gray = perspective_corrector_.correct(gray);

    // 2. Remove surrounding white border (desk, hand, etc.).
    gray = autocrop(gray);

    // 3. Rescale to fixed width so all downstream models see consistent resolution.
    gray = resize_to_width(gray);

    // 4. Local contrast normalisation (handles remaining lighting variation).
    gray = apply_clahe(gray);

    // 5. Remove residual noise and paper texture while preserving ink edges.
    gray = noise_filter_.filter(gray);

    return gray;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 1: autocrop
//
//  Approach (standard contour-based crop):
//    1. Otsu binarisation to separate foreground content from background.
//    2. Morphological closing to bridge small gaps within notation.
//    3. Find the largest external contour – this bounds the printed content.
//    4. Return the bounding rectangle, padded by a small margin.
//
//  Edge-case guard: if the bounding rect covers >80 % of the image area,
//  the image is already cropped and we return it unchanged.
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat Preprocessor::autocrop(const cv::Mat& gray) const {
    // Invert so that dark ink becomes white (foreground = high values).
    cv::Mat inv;
    cv::bitwise_not(gray, inv);

    // Binarise with Otsu threshold.
    cv::Mat binary;
    cv::threshold(inv, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Close small holes to make the content region contiguous.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {7, 7});
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    // Find external contours.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return gray;

    // Select the largest contour by area.
    auto it = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b){ return cv::contourArea(a) < cv::contourArea(b); });

    cv::Rect bbox = cv::boundingRect(*it);

    // Guard: if bbox is almost the full image, skip crop.
    double area_ratio = (double)(bbox.width * bbox.height)
                      / (double)(gray.cols * gray.rows);
    if (area_ratio > 0.80) return gray;

    // Small padding (1 % of image dimension) to avoid clipping edge pixels.
    int pad_x = std::max(2, gray.cols / 100);
    int pad_y = std::max(2, gray.rows / 100);
    bbox.x      = std::max(0,           bbox.x      - pad_x);
    bbox.y      = std::max(0,           bbox.y      - pad_y);
    bbox.width  = std::min(gray.cols - bbox.x, bbox.width  + 2 * pad_x);
    bbox.height = std::min(gray.rows - bbox.y, bbox.height + 2 * pad_y);

    return gray(bbox).clone();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 2: resize to fixed width
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat Preprocessor::resize_to_width(const cv::Mat& src, int target) const {
    if (src.cols == target) return src;

    double scale = (double)target / src.cols;
    int    new_h = (int)std::round(src.rows * scale);

    cv::Mat dst;
    cv::resize(src, dst, {target, new_h}, 0, 0,
               scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
//
//  Reference: Zuiderveld K. (1994). "Contrast Limited Adaptive Histogram
//  Equalization." Graphics Gems IV.
//  Implemented via OpenCV's cv::createCLAHE (Apache 2.0 licence).
//
//  Sheet music images often have uneven illumination from smartphone cameras.
//  CLAHE normalises local contrast without over-amplifying noise, making
//  staff lines and noteheads equally visible across all regions.
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat Preprocessor::apply_clahe(const cv::Mat& gray) const {
    auto clahe = cv::createCLAHE(CLAHE_CLIP, {CLAHE_TILE, CLAHE_TILE});
    cv::Mat out;
    clahe->apply(gray, out);
    return out;
}

} // namespace omr
