#include "encoder_runner.hpp"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <stdexcept>
#include <cstring>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / model loading
// ─────────────────────────────────────────────────────────────────────────────

EncoderRunner::EncoderRunner(const std::string& model_path) {
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_)
        throw std::runtime_error("EncoderRunner: failed to load model: " + model_path);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interp_);
    if (!interp_)
        throw std::runtime_error("EncoderRunner: failed to build interpreter");

    interp_->AllocateTensors();
}

EncoderRunner::~EncoderRunner() = default;

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
    interp_->Invoke();

    // Output tensor: [1, seq_len, EMBED_DIM] float32.
    const TfLiteTensor* out_tensor = interp_->output_tensor(0);
    const int seq_len = out_tensor->dims->data[1];

    // Copy to an OpenCV mat (seq_len rows × EMBED_DIM cols).
    cv::Mat result(seq_len, EMBED_DIM, CV_32FC1);
    const float* src = out_tensor->data.f;
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
    float* input = interp_->typed_input_tensor<float>(0);
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
