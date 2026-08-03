#include "encoder_runner.hpp"
#include <tensorflow/lite/c/c_api.h>
#include <stdexcept>
#include <cstring>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / model loading
// ─────────────────────────────────────────────────────────────────────────────

EncoderRunner::EncoderRunner(const std::string& model_path) {
    model_ = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model_)
        throw std::runtime_error("EncoderRunner: failed to load model: " + model_path);

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    interp_ = TfLiteInterpreterCreate(model_, options);
    TfLiteInterpreterOptionsDelete(options);
    if (!interp_)
        throw std::runtime_error("EncoderRunner: failed to build interpreter");

    TfLiteInterpreterAllocateTensors(interp_);
}

EncoderRunner::~EncoderRunner() {
    if (interp_) TfLiteInterpreterDelete(interp_);
    if (model_)  TfLiteModelDelete(model_);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Inference
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat EncoderRunner::run(const cv::Mat& canvas_u8) const {
    // Verify dimensions.
    if (canvas_u8.rows != CANVAS_H || canvas_u8.cols != CANVAS_W)
        throw std::runtime_error("EncoderRunner: canvas must be "
                                  + std::to_string(CANVAS_H) + "x"
                                  + std::to_string(CANVAS_W));

    fill_input_tensor(canvas_u8);
    TfLiteInterpreterInvoke(interp_);

    // Output tensor: [1, seq_len, EMBED_DIM] float32.
    const TfLiteTensor* out_tensor = TfLiteInterpreterGetOutputTensor(interp_, 0);
    const int seq_len = TfLiteTensorDim(out_tensor, 1);

    // Copy to an OpenCV mat (seq_len rows × EMBED_DIM cols).
    cv::Mat result(seq_len, EMBED_DIM, CV_32FC1);
    const float* src = reinterpret_cast<const float*>(TfLiteTensorData(out_tensor));
    std::memcpy(result.data, src, seq_len * EMBED_DIM * sizeof(float));
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Input normalisation
//
//  Formula: f = (pixel / 255.0 − IMG_MEAN) / IMG_STD
//  Stored as NCHW: [1, 1, CANVAS_H, CANVAS_W] (single channel).
//
//  The constants IMG_MEAN and IMG_STD are the mean and standard deviation of
//  grayscale pixel values (in [0,1] range) across the sheet-music training set.
//  They must match the values used during model training.
// ─────────────────────────────────────────────────────────────────────────────

void EncoderRunner::fill_input_tensor(const cv::Mat& canvas_u8) const {
    TfLiteTensor* in_tensor = TfLiteInterpreterGetInputTensor(interp_, 0);
    float* input = reinterpret_cast<float*>(TfLiteTensorData(in_tensor));
    const float scale  = 1.f / 255.f;

    for (int r = 0; r < CANVAS_H; ++r) {
        const uint8_t* row = canvas_u8.ptr<uint8_t>(r);
        for (int c = 0; c < CANVAS_W; ++c) {
            float f = row[c] * scale;
            input[r * CANVAS_W + c] = (f - IMG_MEAN) / IMG_STD;
        }
    }
}

} // namespace omr
