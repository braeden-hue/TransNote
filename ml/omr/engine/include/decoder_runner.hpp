#pragma once
#include "types.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct TfLiteModel;
struct TfLiteInterpreter;

namespace omr {

/**
 * DecoderRunner – autoregressive greedy/beam decoding via TFLite.
 *
 * Model I/O (matches round3train/export_tflite.py::_DecoderStepWrapper --
 * verified via ml/omr/utils/inspect_tflite.py against
 * round3train/tflite_export/decoder_INT8.tflite):
 *   Inputs:
 *     past_ids : [1, T]      int64   tokens generated so far, incl. leading SOS
 *                                     (T grows by 1 every step)
 *     memory   : [1, S, 512] float32 full encoder output (constant every step,
 *                                     S = EncoderRunner::run()'s seq_len)
 *   Output:
 *     next_logits : [1, vocab_size] float32  logits for the token at position T
 *                                             (i.e. the next token to generate)
 *
 * There is no explicit KV-cache tensor -- the exported model recomputes
 * self-attention over the full past_ids sequence every step (see
 * round3train/export_tflite.py docstring: "매 스텝 O(T) 재계산 -- 608 토큰
 * 이하에서는 감당 가능"). Reference: Vaswani et al. (2017) "Attention is All
 * You Need" (public domain) for the underlying Transformer decoder algorithm.
 *
 * Constants must match round3train/dataset.py / model.py:
 *   MAX_SEQ    = 608  (maximum generated tokens before forced stop)
 *   EOS_ID     = 2    (token ID for End-Of-Sequence)
 *   SOS_ID     = 1    (token ID for Start-Of-Sequence, seed token)
 */
class DecoderRunner {
public:
    static constexpr int MAX_SEQ     = 608;
    static constexpr int32_t EOS_ID  = 2;
    static constexpr int32_t SOS_ID  = 1;
    static constexpr int32_t PAD_ID  = 0;

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

    /**
     * Beam-search decode a full token sequence for one staff canvas.
     *
     * Standard length-normalised beam search. Since the exported model has no
     * external KV-cache, each beam only needs to remember its own generated
     * token list (past_ids is rebuilt from SOS + that list every step) --
     * there is no per-beam cache to clone.
     *
     * @param encoder_out   [seq_len × EMBED_DIM] float32 mat (from EncoderRunner).
     * @param beam_width    Number of parallel hypotheses to track (e.g. 4).
     * @param length_penalty Divides final score by (token_count ^ length_penalty)
     *                       when ranking finished beams (0 = no penalty).
     * @return              Token IDs of the best beam, without SOS/EOS/PAD.
     */
    std::vector<int32_t> decode_beam(const cv::Mat& encoder_out,
                                      int   beam_width     = 4,
                                      float length_penalty  = 0.7f) const;

private:
    // One forward step: past_ids (incl. leading SOS) + constant encoder memory
    // -> logits over the vocabulary for the next token.
    void step_logits(const std::vector<int64_t>& past_ids,
                      const cv::Mat&              memory,
                      std::vector<float>&         out_logits) const;

    int vocab_size_;
    TfLiteModel*       model_  = nullptr;
    TfLiteInterpreter* interp_ = nullptr;
};

} // namespace omr
