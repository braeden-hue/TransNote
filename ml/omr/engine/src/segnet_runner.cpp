#include "segnet_runner.hpp"
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / model loading
// ─────────────────────────────────────────────────────────────────────────────

SegnetRunner::SegnetRunner(const std::string& model_path) {
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_)
        throw std::runtime_error("SegnetRunner: failed to load model: " + model_path);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interp_);
    if (!interp_)
        throw std::runtime_error("SegnetRunner: failed to build interpreter");

    interp_->AllocateTensors();
}

SegnetRunner::~SegnetRunner() = default;

// ─────────────────────────────────────────────────────────────────────────────
//  Public interface
// ─────────────────────────────────────────────────────────────────────────────

std::vector<cv::Mat> SegnetRunner::run(const cv::Mat& gray) const {
    const int H = gray.rows;
    const int W = gray.cols;

    // Accumulators: one float map per foreground class + a weight map for blending.
    // We skip class 0 (background), so we accumulate classes 1..5.
    const int N_FG = SEG_NUM_CLASSES - 1;
    std::vector<cv::Mat> accum(N_FG, cv::Mat::zeros(H, W, CV_32FC1));
    cv::Mat weight_map = cv::Mat::zeros(H, W, CV_32FC1);

    // Slide the window across the image.
    for (int y = 0; y < H; y += STRIDE) {
        for (int x = 0; x < W; x += STRIDE) {
            cv::Mat patch    = extract_patch(gray, x, y);
            auto    patch_out = infer_patch(patch);
            blend_patch(accum, weight_map, patch_out, x, y);
        }
    }

    // Normalise by weight map to handle variable overlap.
    std::vector<cv::Mat> result(N_FG);
    for (int c = 0; c < N_FG; ++c) {
        cv::divide(accum[c], weight_map, result[c]);
        // Clamp to [0,1].
        cv::threshold(result[c], result[c], 0.0, 0.0, cv::THRESH_TOZERO);
        cv::min(result[c], 1.0f, result[c]);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Private helpers
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat SegnetRunner::extract_patch(const cv::Mat& src, int x, int y) const {
    // Region of interest, possibly extending beyond image bounds.
    cv::Rect roi(x, y, PATCH_SIZE, PATCH_SIZE);

    // Create a white-background canvas.
    cv::Mat patch(PATCH_SIZE, PATCH_SIZE, CV_8UC1, cv::Scalar(255));

    // Clamp the ROI to image boundaries.
    cv::Rect img_rect(0, 0, src.cols, src.rows);
    cv::Rect clipped = roi & img_rect;
    if (!clipped.empty()) {
        // Offset inside the patch canvas.
        cv::Rect dst_roi(clipped.x - roi.x, clipped.y - roi.y,
                         clipped.width,      clipped.height);
        src(clipped).copyTo(patch(dst_roi));
    }
    return patch;
}

std::vector<cv::Mat> SegnetRunner::infer_patch(const cv::Mat& patch) const {
    // Fill TFLite input tensor: [1, 3, PATCH_SIZE, PATCH_SIZE] float32 NCHW.
    // We replicate the single grayscale channel across 3 channels, values [0,255].
    float* input = interp_->typed_input_tensor<float>(0);
    const int plane = PATCH_SIZE * PATCH_SIZE;
    for (int i = 0; i < PATCH_SIZE; ++i) {
        const uint8_t* row = patch.ptr<uint8_t>(i);
        for (int j = 0; j < PATCH_SIZE; ++j) {
            float v = static_cast<float>(row[j]);
            // NCHW: channel 0,1,2 at (i*PATCH_SIZE+j)
            input[0 * plane + i * PATCH_SIZE + j] = v;
            input[1 * plane + i * PATCH_SIZE + j] = v;
            input[2 * plane + i * PATCH_SIZE + j] = v;
        }
    }

    interp_->Invoke();

    // Read output: [1, SEG_NUM_CLASSES, PATCH_SIZE, PATCH_SIZE] float32 NCHW.
    const float* output = interp_->typed_output_tensor<float>(0);

    // Return N_FG = 5 soft maps (classes 1..5), using softmax per pixel.
    // We use a simple exp-normalisation; argmax accuracy is sufficient.
    const int N_FG = SEG_NUM_CLASSES - 1;
    std::vector<cv::Mat> maps(N_FG, cv::Mat(PATCH_SIZE, PATCH_SIZE, CV_32FC1));

    for (int r = 0; r < PATCH_SIZE; ++r) {
        for (int c = 0; c < PATCH_SIZE; ++c) {
            int px = r * PATCH_SIZE + c;
            // Collect logits for all classes.
            float logits[SEG_NUM_CLASSES];
            float max_l = -1e9f;
            for (int cls = 0; cls < SEG_NUM_CLASSES; ++cls) {
                logits[cls] = output[cls * plane + px];
                if (logits[cls] > max_l) max_l = logits[cls];
            }
            // Softmax (numerically stable).
            float sum = 0.f;
            float exp_l[SEG_NUM_CLASSES];
            for (int cls = 0; cls < SEG_NUM_CLASSES; ++cls) {
                exp_l[cls] = std::exp(logits[cls] - max_l);
                sum += exp_l[cls];
            }
            for (int fg = 0; fg < N_FG; ++fg)
                maps[fg].at<float>(r, c) = exp_l[fg + 1] / sum;
        }
    }
    return maps;
}

void SegnetRunner::blend_patch(std::vector<cv::Mat>&        accum,
                                cv::Mat&                      weight_map,
                                const std::vector<cv::Mat>&  patch_out,
                                int ox, int oy) {
    const int H = weight_map.rows;
    const int W = weight_map.cols;
    const int P = PATCH_SIZE;
    const int N_FG = static_cast<int>(accum.size());

    for (int r = 0; r < P; ++r) {
        int img_r = oy + r;
        if (img_r < 0 || img_r >= H) continue;

        for (int c = 0; c < P; ++c) {
            int img_c = ox + c;
            if (img_c < 0 || img_c >= W) continue;

            // Linear weight: highest at patch centre, zero at corners.
            float wr = 1.f - std::abs(r - P * 0.5f) / (P * 0.5f);
            float wc = 1.f - std::abs(c - P * 0.5f) / (P * 0.5f);
            float w  = (wr * wc) + 1e-6f;

            weight_map.at<float>(img_r, img_c) += w;
            for (int fg = 0; fg < N_FG; ++fg)
                accum[fg].at<float>(img_r, img_c) += patch_out[fg].at<float>(r, c) * w;
        }
    }
}

} // namespace omr
