#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>
#include <memory>
#include <string>

// Forward-declare the TFLite C API opaque types to avoid header pollution.
// (Android's org.tensorflow:tensorflow-lite AAR only ships the C API headers/
// symbols -- see ml/omr/engine/CMakeLists.txt Android branch -- so this engine
// is built against tflite's C API, not the C++ Interpreter/FlatBufferModel API.)
struct TfLiteModel;
struct TfLiteInterpreter;

namespace omr {

/**
 * SegnetRunner – patch-based semantic segmentation via TFLite.
 *
 * The model classifies each pixel into one of SEG_NUM_CLASSES (6) categories.
 * Because the model has a fixed input size (PATCH_SIZE × PATCH_SIZE), large
 * images are processed in overlapping tiles; blended overlaps reduce artefacts
 * at tile borders.
 *
 * Model I/O (matches round3train/export_tflite.py's actual ONNX->TFLite export
 * for segnet -- verified via ml/omr/utils/inspect_tflite.py against
 * round3train/tflite_export/segnet_INT8.tflite):
 *   Input  : [1, PATCH_SIZE, PATCH_SIZE, 1]  float32  NHWC (single channel)
 *            grayscale, normalised as f = pixel/127.5 - 1.0 (see
 *            round3train/export_tflite.py::_build_segnet_calib)
 *   Output : [1, PATCH_SIZE, PATCH_SIZE, SEG_NUM_CLASSES]  float32  NHWC
 *            raw logits; argmax → predicted class per pixel
 *
 * Algorithm: sliding-window inference with 50 % overlap and linear-blend
 * stitching.  Standard technique; no proprietary IP.
 */
class SegnetRunner {
public:
    static constexpr int PATCH_SIZE  = 320;  // model input side (px)
    static constexpr int STRIDE      = 160;  // 50 % overlap
    static constexpr float BIN_THRESH = 0.4f; // logit → binary mask threshold

    explicit SegnetRunner(const std::string& model_path);
    ~SegnetRunner();

    /**
     * Run segmentation on a full preprocessed grayscale image.
     * @param gray  CV_8UC1, any size (will be padded internally)
     * @return      Vector of SEG_NUM_CLASSES-1 float maps in [0,1],
     *              same spatial size as `gray`, one per foreground class.
     *              Index 0 = SEG_STEM_REST, ..., index 4 = SEG_SYMBOL.
     */
    std::vector<cv::Mat> run(const cv::Mat& gray) const;

private:
    // Extract one patch (white-padded if out of bounds)
    cv::Mat extract_patch(const cv::Mat& src, int x, int y) const;

    // Run one patch through the TFLite model, return [SEG_NUM_CLASSES][PATCH][PATCH]
    std::vector<cv::Mat> infer_patch(const cv::Mat& patch) const;

    // Stitch overlapping patch results back into a full-image accumulator
    static void blend_patch(std::vector<cv::Mat>& accum,
                             cv::Mat&              weight_map,
                             const std::vector<cv::Mat>& patch_out,
                             int ox, int oy);

    TfLiteModel*       model_  = nullptr;
    TfLiteInterpreter* interp_ = nullptr;
};

} // namespace omr
