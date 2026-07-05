#include "decoder_runner.hpp"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

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
    std::vector<float> logits;
    step_logits(prev_token, context, step_idx, kv_cache, logits);

    int32_t best_id = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int v = 0; v < vocab_size_; ++v) {
        if (logits[v] > best_val) { best_val = logits[v]; best_id = v; }
    }
    return best_id;
}

// Shared TFLite step used by both step() (greedy) and decode_beam().
// Fills out_logits (size vocab_size_) and updates kv_cache in-place.
void DecoderRunner::step_logits(int32_t                 prev_token,
                                 const cv::Mat&          context,
                                 int                     step_idx,
                                 std::vector<cv::Mat>&   kv_cache,
                                 std::vector<float>&     out_logits) const {
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

    // ── Output 0: logits ──────────────────────────────────────────────────────
    {
        const float* logits = interp_->typed_output_tensor<float>(0);
        out_logits.assign(logits, logits + vocab_size_);
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
}

std::vector<cv::Mat> DecoderRunner::clone_kv_cache(const std::vector<cv::Mat>& src) {
    std::vector<cv::Mat> out;
    out.reserve(src.size());
    for (const auto& m : src) out.push_back(m.clone());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Beam-search autoregressive decoding
//
//  Each beam owns an independent KV cache. At every step, every not-yet-
//  finished beam is advanced one token (one TFLite invoke per beam — the
//  exported model has no batch dimension, so beams are processed
//  sequentially rather than in one batched call). Its logits are turned into
//  log-probabilities and the beam's own top `beam_width` continuations become
//  candidates. Already-finished beams (hit EOS_ID) are carried forward
//  unchanged as single candidates so they keep competing on equal footing.
//  All candidates are merged and the global top `beam_width` become the next
//  beam set. Ties for which candidates share a parent are resolved by
//  cloning that parent's (already-updated) KV cache once per child.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct Beam {
    int32_t               last_token;
    std::vector<cv::Mat>  kv;
    double                score;      // cumulative log-probability
    std::vector<int32_t>  tokens;     // generated so far, excl. SOS/EOS/PAD
    bool                  finished;
};

// log-softmax over raw logits, written into out_logp (same size as logits).
void log_softmax(const std::vector<float>& logits, std::vector<double>& out_logp) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (float v : logits) sum_exp += std::exp(static_cast<double>(v) - max_logit);
    const double log_sum_exp = std::log(sum_exp) + max_logit;
    out_logp.resize(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
        out_logp[i] = static_cast<double>(logits[i]) - log_sum_exp;
}

// Indices of the top-k values in `values`, descending. PAD (index 0) is
// never a valid decode target and is excluded.
std::vector<int> top_k_indices(const std::vector<double>& values, int k) {
    std::vector<int> idx(values.size());
    std::iota(idx.begin(), idx.end(), 0);
    // +1 so there is always a spare slot to drop PAD if it lands in the top-k.
    const int limit = std::min<int>(k + 1, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + limit, idx.end(),
                       [&](int a, int b) { return values[a] > values[b]; });
    std::vector<int> out;
    out.reserve(k);
    for (int n = 0; n < limit; ++n) {
        const int i = idx[n];
        if (i == DecoderRunner::PAD_ID) continue;
        out.push_back(i);
        if (static_cast<int>(out.size()) == k) break;
    }
    return out;
}

} // namespace

std::vector<int32_t> DecoderRunner::decode_beam(const cv::Mat& encoder_out,
                                                  int   beam_width,
                                                  float length_penalty) const {
    if (beam_width <= 1) return decode(encoder_out);

    std::vector<Beam> beams;
    beams.push_back(Beam{SOS_ID, init_kv_cache(), 0.0, {}, false});

    for (int t = 0; t < MAX_SEQ; ++t) {
        const bool all_finished = std::all_of(beams.begin(), beams.end(),
                                               [](const Beam& b) { return b.finished; });
        if (all_finished) break;

        cv::Mat context = (t == 0) ? encoder_out : encoder_out.rowRange(0, 1);

        // (score, source-beam-index, chosen-token, is-new-token)
        struct Candidate { double score; int parent; int32_t token; bool advances; };
        std::vector<Candidate> candidates;
        candidates.reserve(beams.size() * beam_width);

        for (int i = 0; i < static_cast<int>(beams.size()); ++i) {
            if (beams[i].finished) {
                // Carry forward unchanged so it keeps competing fairly.
                // (token value is unused when advances=false)
                candidates.push_back({beams[i].score, i, DecoderRunner::PAD_ID, false});
                continue;
            }
            std::vector<float> logits;
            step_logits(beams[i].last_token, context, t, beams[i].kv, logits);
            std::vector<double> logp;
            log_softmax(logits, logp);

            for (int tok : top_k_indices(logp, beam_width))
                candidates.push_back({beams[i].score + logp[tok], i, tok, true});
        }

        const size_t keep = std::min(candidates.size(), static_cast<size_t>(beam_width));
        std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                           [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
        candidates.resize(keep);

        std::vector<Beam> next_beams;
        next_beams.reserve(candidates.size());
        for (const auto& c : candidates) {
            if (!c.advances) {
                next_beams.push_back(beams[c.parent]);  // already-finished beam, unchanged
                continue;
            }
            Beam nb;
            nb.kv    = clone_kv_cache(beams[c.parent].kv);  // parent's post-step cache
            nb.score = c.score;
            nb.tokens = beams[c.parent].tokens;
            if (c.token == EOS_ID) {
                nb.finished   = true;
                nb.last_token = EOS_ID;
            } else {
                nb.finished = false;
                nb.tokens.push_back(c.token);
                nb.last_token = c.token;
            }
            next_beams.push_back(std::move(nb));
        }
        beams = std::move(next_beams);
    }

    // Pick the best beam by length-normalised score.
    const Beam* best = nullptr;
    double best_norm = -std::numeric_limits<double>::infinity();
    for (const auto& b : beams) {
        const double len = std::max<size_t>(b.tokens.size(), 1);
        const double norm = b.score / std::pow(len, length_penalty);
        if (norm > best_norm) { best_norm = norm; best = &b; }
    }
    return best ? best->tokens : std::vector<int32_t>{};
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
