#include "perspective_corrector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat PerspectiveCorrector::correct(const cv::Mat& gray) const {
    // Attempt full perspective correction (Stage 1).
    auto corners = detect_page_quad(gray);
    if (!corners.empty())
        return warp_quad(gray, corners);

    // Fallback: rotation-only deskew (Stage 2).
    return deskew(gray);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stage 1: quadrilateral page detection
//
//  We work on a downsampled copy (≤640 px wide) for speed, then scale the
//  detected corners back to the original resolution.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<cv::Point2f>
PerspectiveCorrector::detect_page_quad(const cv::Mat& gray) const {
    // 1. Downsample for speed.
    const double scale = std::min(1.0, 640.0 / gray.cols);
    cv::Mat small;
    cv::resize(gray, small, {}, scale, scale, cv::INTER_AREA);

    // 2. Soft blur then Otsu threshold on the inverted image.
    //    (Ink is dark → inverted ink is bright; paper background becomes dark.)
    cv::Mat inv;
    cv::GaussianBlur(small, inv, {5, 5}, 0);
    cv::bitwise_not(inv, inv);
    cv::Mat binary;
    cv::threshold(inv, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 3. Large morphological close to merge the content into one solid region.
    //    Kernel size ≈ 3 % of image width.
    int kside = std::max(11, (int)(small.cols * 0.03)) | 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {kside, kside});
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    // 4. Find the largest external contour.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {};

    auto it = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b){ return cv::contourArea(a) < cv::contourArea(b); });

    const double area     = cv::contourArea(*it);
    const double img_area = small.cols * small.rows;
    if (area / img_area < MIN_PAGE_AREA_RATIO) return {};

    // 5. Convex hull of the largest contour, then approximate as polygon.
    //    Sweep epsilon until we get exactly 4 corners.
    std::vector<cv::Point> hull;
    cv::convexHull(*it, hull);

    const double peri = cv::arcLength(hull, true);
    std::vector<cv::Point> approx;
    for (double eps_f : {0.02, 0.04, 0.06, 0.08, 0.10, 0.12}) {
        cv::approxPolyDP(hull, approx, eps_f * peri, true);
        if (approx.size() == 4) break;
    }
    if (approx.size() != 4) return {};

    // 6. Scale corners back to original resolution and order them.
    std::vector<cv::Point2f> pts;
    pts.reserve(4);
    for (const auto& p : approx)
        pts.push_back({(float)(p.x / scale), (float)(p.y / scale)});

    return order_points(pts);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stage 1b: warp quadrilateral to upright rectangle
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat PerspectiveCorrector::warp_quad(const cv::Mat&                  gray,
                                         const std::vector<cv::Point2f>& c) const {
    // c = [TL, TR, BR, BL].  Compute output rectangle dimensions.
    float w_top  = cv::norm(c[1] - c[0]);
    float w_bot  = cv::norm(c[2] - c[3]);
    float h_left = cv::norm(c[3] - c[0]);
    float h_right= cv::norm(c[2] - c[1]);

    int dst_w = (int)std::round(std::max(w_top, w_bot));
    int dst_h = (int)std::round(std::max(h_left, h_right));
    if (dst_w < 16 || dst_h < 16) return gray;

    const std::vector<cv::Point2f> dst = {
        {0.f,              0.f},
        {(float)(dst_w-1), 0.f},
        {(float)(dst_w-1), (float)(dst_h-1)},
        {0.f,              (float)(dst_h-1)}
    };

    cv::Mat M = cv::getPerspectiveTransform(c, dst);
    cv::Mat out;
    cv::warpPerspective(gray, out, M, {dst_w, dst_h},
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stage 2: Hough-based deskew (rotation only)
//
//  Algorithm:
//    1. Canny edge detection on a downsampled copy.
//    2. Probabilistic Hough Transform to find line segments.
//    3. Keep near-horizontal lines (|angle| ≤ MAX_DESKEW_ANGLE_DEG).
//    4. Take the median angle → rotate the full-resolution image to compensate.
//
//  Ref: Matas J. et al. (2000) CVIU 78(1):119–137.
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat PerspectiveCorrector::deskew(const cv::Mat& gray) const {
    // Work at ≤1024 px for speed.
    const double scale = std::min(1.0, 1024.0 / gray.cols);
    cv::Mat small;
    cv::resize(gray, small, {}, scale, scale, cv::INTER_AREA);

    cv::Mat edges;
    cv::Canny(small, edges, 50, 150);

    // Probabilistic Hough: require lines ≥ 30 % of image width.
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines,
                    /*rho=*/1.0, /*theta=*/CV_PI / 180.0,
                    /*threshold=*/60,
                    /*minLineLength=*/small.cols * MIN_LINE_FRAC,
                    /*maxLineGap=*/small.cols * 0.05);

    if (lines.empty()) return gray;

    // Collect angles of near-horizontal segments.
    std::vector<double> angles;
    angles.reserve(lines.size());
    for (const auto& l : lines) {
        const double dx = l[2] - l[0];
        const double dy = l[3] - l[1];
        if (std::abs(dx) < 1.0) continue;
        const double ang = std::atan2(dy, dx) * 180.0 / CV_PI;
        if (std::abs(ang) <= MAX_DESKEW_ANGLE_DEG)
            angles.push_back(ang);
    }
    if (angles.empty()) return gray;

    // Robust angle estimate: median.
    std::sort(angles.begin(), angles.end());
    const double median_angle = angles[angles.size() / 2];
    if (std::abs(median_angle) < 0.1) return gray;  // negligible

    // Rotate around image centre; fill new border with white (paper colour).
    const cv::Point2f centre(gray.cols / 2.f, gray.rows / 2.f);
    cv::Mat R = cv::getRotationMatrix2D(centre, median_angle, 1.0);

    cv::Mat out;
    cv::warpAffine(gray, out, R, gray.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Utility: reorder 4 points to [TL, TR, BR, BL]
//
//  Method: TL has the smallest (x+y), BR the largest.  Of the remaining two,
//  the one with the larger (x−y) is TR (right+top), the other is BL.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<cv::Point2f>
PerspectiveCorrector::order_points(std::vector<cv::Point2f> pts) {
    // Sort ascending by x+y: index 0 = TL, index 3 = BR.
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b){
                  return (a.x + a.y) < (b.x + b.y); });

    const cv::Point2f tl = pts[0], br = pts[3];

    // Of the middle two: larger (x−y) → TR; smaller → BL.
    if ((pts[1].x - pts[1].y) > (pts[2].x - pts[2].y))
        return {tl, pts[1], br, pts[2]};  // [TL, TR, BR, BL]
    else
        return {tl, pts[2], br, pts[1]};
}

} // namespace omr
