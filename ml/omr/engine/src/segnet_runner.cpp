#include "segnet_runner.hpp"
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / model loading
// ─────────────────────────────────────────────────────────────────────────────

SegnetRunner::SegnetRunner(const std::string& model_path) {
    model_ = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model_)
        throw std::runtime_error("SegnetRunner: failed to load model: " + model_path);

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    interp_ = TfLiteInterpreterCreate(model_, options);
    TfLiteInterpreterOptionsDelete(options);
    if (!interp_)
        throw std::runtime_error("SegnetRunner: failed to build interpreter");

    TfLiteInterpreterAllocateTensors(interp_);
}

SegnetRunner::~SegnetRunner() {
    if (interp_) TfLiteInterpreterDelete(interp_);
    if (model_)  TfLiteModelDelete(model_);
}

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
    // Fill TFLite input tensor: [1, PATCH_SIZE, PATCH_SIZE, 1] float32 NHWC
    // (single channel -- see round3train/export_tflite.py, onnx2tf converts
    // the PyTorch NCHW export to NHWC since channel count is 1 there's no
    // memory-layout difference from a plain HxW grid).
    // Normalisation matches round3train/export_tflite.py::_build_segnet_calib:
    //   f = pixel/127.5 - 1.0
    TfLiteTensor* in_tensor = TfLiteInterpreterGetInputTensor(interp_, 0);
    float* input = reinterpret_cast<float*>(TfLiteTensorData(in_tensor));
    const int plane = PATCH_SIZE * PATCH_SIZE;
    for (int i = 0; i < PATCH_SIZE; ++i) {
        const uint8_t* row = patch.ptr<uint8_t>(i);
        float* out_row = input + i * PATCH_SIZE;
        for (int j = 0; j < PATCH_SIZE; ++j) {
            out_row[j] = static_cast<float>(row[j]) / 127.5f - 1.0f;
        }
    }

    TfLiteInterpreterInvoke(interp_);

    // Read output: [1, PATCH_SIZE, PATCH_SIZE, SEG_NUM_CLASSES] float32 NHWC.
    const TfLiteTensor* out_tensor = TfLiteInterpreterGetOutputTensor(interp_, 0);
    const float* output = reinterpret_cast<const float*>(TfLiteTensorData(out_tensor));

    // Return N_FG = 5 soft maps (classes 1..5), using softmax per pixel.
    // We use a simple exp-normalisation; argmax accuracy is sufficient.
    const int N_FG = SEG_NUM_CLASSES - 1;
    std::vector<cv::Mat> maps(N_FG, cv::Mat(PATCH_SIZE, PATCH_SIZE, CV_32FC1));
    (void)plane;

    for (int r = 0; r < PATCH_SIZE; ++r) {
        for (int c = 0; c < PATCH_SIZE; ++c) {
            int px = r * PATCH_SIZE + c;
            const float* logits = output + px * SEG_NUM_CLASSES; // NHWC: channels contiguous
            float max_l = -1e9f;
            for (int cls = 0; cls < SEG_NUM_CLASSES; ++cls)
                if (logits[cls] > max_l) max_l = logits[cls];
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
