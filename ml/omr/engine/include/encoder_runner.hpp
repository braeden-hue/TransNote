#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

struct TfLiteModel;
struct TfLiteInterpreter;

namespace omr {

/**
 * EncoderRunner – maps a staff canvas to a latent sequence via TFLite.
 *
 * Expected model I/O (ConvNeXt-based CNN encoder, trained from scratch):
 *   Input  : [1, 1, CANVAS_H, CANVAS_W]  float32  NCHW
 *            normalised:  f = (pixel/255 − IMG_MEAN) / IMG_STD
 *   Output : [1, seq_len, EMBED_DIM]     float32
 *            where seq_len ≈ CANVAS_W / 4 = 320 for a 1280-wide canvas.
 *
 * Normalisation constants:
 *   IMG_MEAN = 0.7931   (mean pixel value of sheet-music background at uint8/255)
 *   IMG_STD  = 0.1738   (std of the same population)
 * These must match the constants used during model training.
 */
class EncoderRunner {
public:
    static constexpr float IMG_MEAN  = 0.7931f;
    static constexpr float IMG_STD   = 0.1738f;
    static constexpr int   EMBED_DIM = 512;

    explicit EncoderRunner(const std::string& model_path);
    ~EncoderRunner();

    /**
     * Encode one canvas tile.
     * @param canvas_u8  CV_8UC1, exactly CANVAS_H × CANVAS_W.
     * @return           Float matrix of shape [seq_len × EMBED_DIM] (CV_32FC1, row=position).
     */
    cv::Mat run(const cv::Mat& canvas_u8) const;

private:
    // Normalise uint8 canvas → float32 NCHW tensor in-place into interpreter input.
    void fill_input_tensor(const cv::Mat& canvas_u8) const;

    TfLiteModel*       model_  = nullptr;
    TfLiteInterpreter* interp_ = nullptr;
};

} // namespace omr
