#include "decoder_runner.hpp"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <limits>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────

DecoderRunner::DecoderRunner(const std::string& model_path, int vocab_size)
    : vocab_size_(vocab_size)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_)
        throw std::runtime_error("DecoderRunner: failed to load model: " + model_path);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interp_);
    if (!interp_)
        throw std::runtime_error("DecoderRunner: failed to build interpreter");

    // Note: AllocateTensors() is called per-step because the cache size grows.
}

DecoderRunner::~DecoderRunner() = default;

// ─────────────────────────────────────────────────────────────────────────────
//  Greedy autoregressive decoding
//
//  Algorithm: standard Transformer greedy decoding with KV cache.
//
//  At each step t:
//    - Feed the previous token id and the encoder context.
//    - The model returns logits over the vocabulary.
//    - Greedy selection: argmax(logits).
//    - If the selected token is EOS, stop.
//    - Accumulate the updated KV cache for the next step.
//
//  KV cache avoids recomputing attention over past positions at every step.
//  This is a standard inference optimisation described in the original
//  Transformer paper (Vaswani et al. 2017) and widely used in production.
//
//  For step 0: we pass the full encoder output as context.
//  For steps 1+: we pass only the first encoder position as context token
//                (the KV cache provides access to all past information).
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int32_t> DecoderRunner::decode(const cv::Mat& encoder_out) const {
    std::vector<cv::Mat> kv_cache = init_kv_cache();
    std::vector<int32_t> result;

    int32_t prev_token = SOS_ID;

    for (int t = 0; t < MAX_SEQ; ++t) {
        // On step 0 pass the full encoder output; after that pass the first token only.
        cv::Mat context;
        if (t == 0) {
            context = encoder_out;                                // [seq_len, 512]
        } else {
            context = encoder_out.rowRange(0, 1);                 // [1, 512]
        }

        int32_t next_token = step(prev_token, context, t, kv_cache);

        if (next_token == EOS_ID) break;

        // Skip PAD tokens (safety guard).
        if (next_token == 0) continue;

        result.push_back(next_token);
        prev_token = next_token;
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Single decoder step
//
//  Model input tensors (must match the trained TFLite model):
//    [0] token_in    : [1, 1]                int32
//    [1] context     : [1, seq_len, 512]     float32  (seq_len = 1 after step 0)
//    [2] cache_len   : [1]                   int32
//    [3..34] kv_in_i : [1, NUM_HEADS, t, HEAD_DIM]  float32   (NUM_KV = 32 tensors)
//
//  Model output tensors:
//    [0] logits_out  : [1, 1, vocab_size]    float32
//    [1..32] kv_out_i: [1, NUM_HEADS, t+1, HEAD_DIM]  float32
// ─────────────────────────────────────────────────────────────────────────────

int32_t DecoderRunner::step(int32_t            prev_token,
                             const cv::Mat&     context,
                             int                step_idx,
                             std::vector<cv::Mat>& kv_cache) const {
    // Re-allocate tensors because cache shape changes each step.
    interp_->AllocateTensors();

    // ── Input 0: token_id ────────────────────────────────────────────────────
    {
        int32_t* t_in = interp_->typed_input_tensor<int32_t>(0);
        t_in[0] = prev_token;
    }

    // ── Input 1: encoder context ─────────────────────────────────────────────
    {
        float* ctx_in = interp_->typed_input_tensor<float>(1);
        const int n = context.rows * context.cols;
        std::memcpy(ctx_in, context.ptr<float>(0), n * sizeof(float));
    }

    // ── Input 2: cache_len (current step count) ──────────────────────────────
    {
        int32_t* cl = interp_->typed_input_tensor<int32_t>(2);
        cl[0] = step_idx;
    }

    // ── Inputs 3..34: KV cache per layer ─────────────────────────────────────
    for (int i = 0; i < NUM_KV; ++i) {
        float* kv_in = interp_->typed_input_tensor<float>(3 + i);
        if (!kv_cache[i].empty()) {
            const int n = kv_cache[i].rows * kv_cache[i].cols;
            std::memcpy(kv_in, kv_cache[i].ptr<float>(0), n * sizeof(float));
        }
    }

    interp_->Invoke();

    // ── Output 0: logits → argmax ─────────────────────────────────────────────
    int32_t best_id = 0;
    {
        const float* logits = interp_->typed_output_tensor<float>(0);
        float best_val = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < vocab_size_; ++v) {
            if (logits[v] > best_val) { best_val = logits[v]; best_id = v; }
        }
    }

    // ── Outputs 1..32: update KV cache ───────────────────────────────────────
    for (int i = 0; i < NUM_KV; ++i) {
        const TfLiteTensor* kv_out_t = interp_->output_tensor(1 + i);
        // Shape: [1, NUM_HEADS, step_idx+1, HEAD_DIM]
        const int new_cache_len = step_idx + 1;
        const int n = NUM_HEADS * new_cache_len * HEAD_DIM;
        kv_cache[i].create(new_cache_len * NUM_HEADS, HEAD_DIM, CV_32FC1);
        std::memcpy(kv_cache[i].ptr<float>(0), kv_out_t->data.f, n * sizeof(float));
    }

    return best_id;
}

// ─────────────────────────────────────────────────────────────────────────────
//  KV cache initialisation (empty mats = zero past steps)
// ─────────────────────────────────────────────────────────────────────────────

std::vector<cv::Mat> DecoderRunner::init_kv_cache() const {
    // Each cache tensor starts empty (0 time steps).
    // Its first real dimension (cache_len) grows each step.
    return std::vector<cv::Mat>(NUM_KV);   // NUM_KV empty cv::Mat objects
}

} // namespace omr
