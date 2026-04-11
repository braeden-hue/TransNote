#pragma once
#include "types.hpp"
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace tflite { class FlatBufferModel; class Interpreter; }

namespace omr {

/**
 * DecoderRunner – autoregressive greedy decoding via TFLite.
 *
 * Algorithm: standard Transformer decoder inference with Key-Value cache.
 * Reference: Vaswani et al. (2017) "Attention is All You Need" (public domain).
 *
 * Expected model I/O (one forward step):
 *   Inputs:
 *     token_in   : [1, 1]           int32   current token id
 *     context    : [1, seq_len, 512] float32 full encoder output (step 0)
 *                  or [1, 1, 512]           first encoder token (step > 0)
 *     cache_len  : [1]              int32   number of past steps (= current step)
 *     kv_in[i]   : [1, NUM_HEADS, cache_len, HEAD_DIM] float32  (i = 0..NUM_KV-1)
 *   Outputs:
 *     logits_out : [1, 1, VOCAB_SIZE] float32  unnormalised scores
 *     kv_out[i]  : [1, NUM_HEADS, cache_len+1, HEAD_DIM] float32
 *
 * Constants must match the model trained in Phase 2:
 *   NUM_LAYERS = 8    (decoder transformer blocks)
 *   NUM_HEADS  = 8    (multi-head attention heads)
 *   HEAD_DIM   = 64   (dimension per head)
 *   NUM_KV     = 32   (NUM_LAYERS × 2 for key + value per layer)
 *   MAX_SEQ    = 608  (maximum generated tokens before forced stop)
 *   EOS_ID     = 2    (token ID for End-Of-Sequence)
 *   SOS_ID     = 1    (token ID for Start-Of-Sequence, seed token)
 */
class DecoderRunner {
public:
    static constexpr int NUM_LAYERS  = 8;
    static constexpr int NUM_HEADS   = 8;
    static constexpr int HEAD_DIM    = 64;
    static constexpr int NUM_KV      = NUM_LAYERS * 2;  // key + value per layer
    static constexpr int MAX_SEQ     = 608;
    static constexpr int32_t EOS_ID  = 2;
    static constexpr int32_t SOS_ID  = 1;

    /**
     * @param model_path   Path to decoder TFLite model.
     * @param vocab_size   Number of tokens in the unified vocabulary.
     */
    explicit DecoderRunner(const std::string& model_path, int vocab_size);
    ~DecoderRunner();

    /**
     * Greedy decode a full token sequence for one staff canvas.
     *
     * @param encoder_out  [seq_len × EMBED_DIM] float32 mat (from EncoderRunner).
     * @return             Token IDs in order, without SOS/EOS/PAD.
     */
    std::vector<int32_t> decode(const cv::Mat& encoder_out) const;

private:
    // One step of the autoregressive loop.
    // Updates kv_cache in-place and returns argmax token ID.
    int32_t step(int32_t      prev_token,
                 const cv::Mat& context,     // [1 or seq_len, 512]
                 int            step_idx,
                 std::vector<cv::Mat>& kv_cache) const;

    // Initialise kv_cache to NUM_KV empty mats (zero rows).
    std::vector<cv::Mat> init_kv_cache() const;

    int vocab_size_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter>     interp_;
};

} // namespace omr
