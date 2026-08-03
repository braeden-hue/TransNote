#include "decoder_runner.hpp"
#include <tensorflow/lite/c/c_api.h>
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
    model_ = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model_)
        throw std::runtime_error("DecoderRunner: failed to load model: " + model_path);

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    interp_ = TfLiteInterpreterCreate(model_, options);
    TfLiteInterpreterOptionsDelete(options);
    if (!interp_)
        throw std::runtime_error("DecoderRunner: failed to build interpreter");

    // Note: tensors are (re)allocated per-step in step_logits() because
    // past_ids grows by one token every step (dynamic seq_len axis).
}

DecoderRunner::~DecoderRunner() {
    if (interp_) TfLiteInterpreterDelete(interp_);
    if (model_)  TfLiteModelDelete(model_);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Greedy autoregressive decoding
//
//  Algorithm: standard Transformer greedy decoding. The exported model has no
//  external KV-cache (see decoder_runner.hpp) -- every step re-feeds the full
//  token history (past_ids) plus the constant encoder memory, and the model
//  recomputes self-attention over the whole history internally.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int32_t> DecoderRunner::decode(const cv::Mat& encoder_out) const {
    std::vector<int64_t> past_ids = {SOS_ID};
    std::vector<int32_t> result;

    for (int t = 0; t < MAX_SEQ; ++t) {
        std::vector<float> logits;
        step_logits(past_ids, encoder_out, logits);

        int32_t best_id = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < vocab_size_; ++v) {
            if (logits[v] > best_val) { best_val = logits[v]; best_id = v; }
        }

        if (best_id == EOS_ID) break;

        // Always advance past_ids (position in the sequence is implicit in its
        // length -- there's no separate step_idx/cache_len input any more), but
        // keep stray PAD predictions out of the returned token list (safety
        // guard, mirrors the previous KV-cache implementation's behaviour).
        past_ids.push_back(best_id);
        if (best_id != PAD_ID) result.push_back(best_id);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Single decoder step
//
//  Model input tensors (round3train/export_tflite.py::_DecoderStepWrapper):
//    [0] past_ids : [1, T]      int64    tokens so far, incl. leading SOS
//    [1] memory   : [1, S, 512] float32  full encoder output (constant)
//  Model output tensor:
//    [0] next_logits : [1, vocab_size] float32
// ─────────────────────────────────────────────────────────────────────────────

void DecoderRunner::step_logits(const std::vector<int64_t>& past_ids,
                                 const cv::Mat&              memory,
                                 std::vector<float>&         out_logits) const {
    const int T = static_cast<int>(past_ids.size());
    const int S = memory.rows;
    const int embed_dim = memory.cols;

    // Both inputs have dynamic axes in the exported model (past_ids' seq_len,
    // memory's enc_seq) -- resize before (re)allocating. Both calls' status is
    // checked: a silently-failed resize would leave a stale (too-small) tensor
    // allocated, and the memcpy calls below would overrun it, corrupting the
    // interpreter's shared tensor arena -- this was observed on-device as a
    // SIGSEGV inside this function, tens/hundreds of steps into a decode, once
    // the corruption finally hit an unmapped page (see git history for the
    // crash backtrace this replaced -- reading past a too-small next_logits
    // buffer was the *symptom*, not the cause).
    int dims_ids[2]    = {1, T};
    int dims_memory[3] = {1, S, embed_dim};
    if (TfLiteInterpreterResizeInputTensor(interp_, 0, dims_ids, 2) != kTfLiteOk ||
        TfLiteInterpreterResizeInputTensor(interp_, 1, dims_memory, 3) != kTfLiteOk ||
        TfLiteInterpreterAllocateTensors(interp_) != kTfLiteOk) {
        throw std::runtime_error("DecoderRunner: failed to resize/allocate tensors for T=" +
                                  std::to_string(T) + " S=" + std::to_string(S));
    }

    // ── Input 0: past_ids ────────────────────────────────────────────────────
    {
        TfLiteTensor* t_in = TfLiteInterpreterGetInputTensor(interp_, 0);
        if (TfLiteTensorByteSize(t_in) < T * sizeof(int64_t))
            throw std::runtime_error("DecoderRunner: past_ids tensor smaller than expected");
        int64_t* data = reinterpret_cast<int64_t*>(TfLiteTensorData(t_in));
        std::memcpy(data, past_ids.data(), T * sizeof(int64_t));
    }

    // ── Input 1: encoder memory ──────────────────────────────────────────────
    {
        TfLiteTensor* t_in = TfLiteInterpreterGetInputTensor(interp_, 1);
        if (TfLiteTensorByteSize(t_in) < S * embed_dim * sizeof(float))
            throw std::runtime_error("DecoderRunner: memory tensor smaller than expected");
        float* data = reinterpret_cast<float*>(TfLiteTensorData(t_in));
        std::memcpy(data, memory.ptr<float>(0), S * embed_dim * sizeof(float));
    }

    if (TfLiteInterpreterInvoke(interp_) != kTfLiteOk)
        throw std::runtime_error("DecoderRunner: invoke failed at T=" + std::to_string(T));

    // ── Output 0: next_logits ────────────────────────────────────────────────
    {
        const TfLiteTensor* out_tensor = TfLiteInterpreterGetOutputTensor(interp_, 0);
        if (TfLiteTensorByteSize(out_tensor) < static_cast<size_t>(vocab_size_) * sizeof(float))
            throw std::runtime_error("DecoderRunner: next_logits tensor smaller than vocab_size_ ("
                                      + std::to_string(vocab_size_) + ")");
        const float* logits = reinterpret_cast<const float*>(TfLiteTensorData(out_tensor));
        out_logits.assign(logits, logits + vocab_size_);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Beam-search autoregressive decoding
//
//  Each beam only needs to remember its own generated token list -- since the
//  model has no external KV-cache, past_ids is simply rebuilt from
//  SOS + beam.tokens every step, so there is no per-beam cache to clone (this
//  is simpler than the old KV-cache design, which had to deep-copy per-layer
//  cv::Mat caches on divergence).
//  Already-finished beams (hit EOS_ID) are carried forward unchanged as single
//  candidates so they keep competing on equal footing. All candidates are
//  merged and the global top `beam_width` become the next beam set.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct Beam {
    std::vector<int32_t> tokens;   // generated so far, excl. SOS/EOS/PAD
    double                score;   // cumulative log-probability
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
    beams.push_back(Beam{{}, 0.0, false});

    for (int t = 0; t < MAX_SEQ; ++t) {
        const bool all_finished = std::all_of(beams.begin(), beams.end(),
                                               [](const Beam& b) { return b.finished; });
        if (all_finished) break;

        // (score, source-beam-index, chosen-token, is-new-token)
        struct Candidate { double score; int parent; int32_t token; bool advances; };
        std::vector<Candidate> candidates;
        candidates.reserve(beams.size() * beam_width);

        for (int i = 0; i < static_cast<int>(beams.size()); ++i) {
            if (beams[i].finished) {
                // Carry forward unchanged so it keeps competing fairly.
                candidates.push_back({beams[i].score, i, DecoderRunner::PAD_ID, false});
                continue;
            }

            std::vector<int64_t> past_ids;
            past_ids.reserve(beams[i].tokens.size() + 1);
            past_ids.push_back(SOS_ID);
            for (int32_t tok : beams[i].tokens) past_ids.push_back(tok);

            std::vector<float> logits;
            step_logits(past_ids, encoder_out, logits);
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
            nb.score  = c.score;
            nb.tokens = beams[c.parent].tokens;
            if (c.token == EOS_ID) {
                nb.finished = true;
            } else {
                nb.finished = false;
                nb.tokens.push_back(c.token);
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

} // namespace omr
