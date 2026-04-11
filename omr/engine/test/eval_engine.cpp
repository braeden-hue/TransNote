/**
 * eval_engine.cpp  –  Batch accuracy evaluator for the OMR C++ engine.
 *
 * For each sample in <test_dir> that has both a .png (or .jpg) image
 * and a .json ground-truth label, the engine runs inference and the
 * predicted token sequence is compared with the ground-truth sequence
 * using Token Error Rate (TER):
 *
 *   TER  = Levenshtein(pred, gt) / len(gt)       (lower is better)
 *   Acc  = max(0, 1 - TER)                        (higher is better)
 *
 * Additionally, note-only accuracy is computed by filtering out all
 * non-note tokens (clef, key, time, barlines) from both sequences.
 *
 * Exit code: 0 if mean accuracy >= threshold, 1 otherwise.
 *
 * Build:
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release \
 *         -DTFLITE_ROOT=... -DOpenCV_DIR=...
 *   cmake --build build -j$(nproc)
 *
 * Usage:
 *   ./build/omr_eval <test_dir> <segnet.tflite> <encoder.tflite> \
 *                    <decoder.tflite> <tokenizer.json> \
 *                    [--threshold 0.90] [--report report.csv]
 */

#include "omr_engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#else
#  include <dirent.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Utility: read a whole file into a byte vector
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Levenshtein distance on integer-coded token sequences
//
//  Standard dynamic-programming algorithm, O(m×n) time, O(min(m,n)) space.
//  Ref: Levenshtein V.I. (1966) "Binary Codes Capable of Correcting Deletions,
//       Insertions, and Reversals." Soviet Physics Doklady, 10(8):707–710.
// ─────────────────────────────────────────────────────────────────────────────

static int levenshtein(const std::vector<std::string>& a,
                       const std::vector<std::string>& b) {
    const int m = (int)a.size();
    const int n = (int)b.size();
    if (m == 0) return n;
    if (n == 0) return m;

    // Use two rolling rows instead of the full (m+1)×(n+1) matrix.
    std::vector<int> prev(n + 1), curr(n + 1);
    for (int j = 0; j <= n; ++j) prev[j] = j;

    for (int i = 1; i <= m; ++i) {
        curr[0] = i;
        for (int j = 1; j <= n; ++j) {
            if (a[i-1] == b[j-1]) {
                curr[j] = prev[j-1];
            } else {
                curr[j] = 1 + std::min({prev[j],       // deletion
                                         curr[j-1],     // insertion
                                         prev[j-1]});   // substitution
            }
        }
        std::swap(prev, curr);
    }
    return prev[n];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ground-truth JSON parser
//
//  Parses the format written by generate_dataset.py:
//    { "tokens": [ "clef-G", "key-C", "time-4/4", "note-C4-1/4", ... ] }
//
//  No third-party JSON library is used.
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::string> parse_gt_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    std::vector<std::string> tokens;

    // Locate the "tokens" array: find "tokens" then the opening '['.
    size_t key_pos = content.find("\"tokens\"");
    if (key_pos == std::string::npos) return {};
    size_t arr_pos = content.find('[', key_pos);
    if (arr_pos == std::string::npos) return {};

    // Walk through the array, collecting quoted strings.
    bool in_string = false;
    std::string tok;
    for (size_t i = arr_pos + 1; i < content.size(); ++i) {
        char ch = content[i];
        if (ch == ']' && !in_string) break;   // end of array
        if (ch == '"') {
            if (!in_string) {
                in_string = true;
                tok.clear();
            } else {
                in_string = false;
                if (!tok.empty()) tokens.push_back(tok);
            }
        } else if (in_string) {
            tok += ch;
        }
    }
    return tokens;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Filter: keep only note/chord/rest tokens (drop structural tokens)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::string> note_only(const std::vector<std::string>& seq) {
    std::vector<std::string> out;
    for (const auto& t : seq) {
        if (t.rfind("note-",  0) == 0 ||
            t.rfind("chord-", 0) == 0 ||
            t.rfind("rest-",  0) == 0)
        {
            out.push_back(t);
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Directory listing (returns .png/.jpg files, without extension)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::string> list_image_stems(const std::string& dir) {
    std::vector<std::string> stems;

#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return stems;
    do {
        std::string name(fd.cFileName);
        for (auto& suffix : {".png", ".jpg", ".jpeg", ".PNG", ".JPG"}) {
            size_t pos = name.rfind(suffix);
            if (pos != std::string::npos && pos + strlen(suffix) == name.size()) {
                stems.push_back(name.substr(0, pos));
                break;
            }
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* dp = opendir(dir.c_str());
    if (!dp) return stems;
    struct dirent* entry;
    while ((entry = readdir(dp)) != nullptr) {
        std::string name(entry->d_name);
        for (const char* suffix : {".png", ".jpg", ".jpeg", ".PNG", ".JPG"}) {
            size_t pos = name.rfind(suffix);
            if (pos != std::string::npos && pos + strlen(suffix) == name.size()) {
                stems.push_back(name.substr(0, pos));
                break;
            }
        }
    }
    closedir(dp);
#endif

    std::sort(stems.begin(), stems.end());
    return stems;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Find image file for a stem (try .png then .jpg)
// ─────────────────────────────────────────────────────────────────────────────

static std::string find_image(const std::string& dir, const std::string& stem) {
    for (const char* ext : {".png", ".jpg", ".jpeg"}) {
        std::string p = dir + "/" + stem + ext;
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return "";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Per-sample result
// ─────────────────────────────────────────────────────────────────────────────

struct SampleResult {
    std::string stem;
    bool        has_gt       = false;
    bool        infer_ok     = false;
    int         gt_len       = 0;
    int         pred_len     = 0;
    int         edit_dist    = 0;    // full sequence
    int         note_gt_len  = 0;
    int         note_edit    = 0;    // note-only sequence
    double      ter          = 1.0;  // token error rate (full)
    double      note_ter     = 1.0;  // token error rate (notes only)
    double      accuracy     = 0.0;  // = max(0, 1 - ter)
    std::string error_msg;
};

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Usage: omr_eval <test_dir> <segnet.tflite> <encoder.tflite>"
            " <decoder.tflite> <tokenizer.json>"
            " [--threshold 0.90] [--report out.csv]\n");
        return 2;
    }

    const std::string test_dir   = argv[1];
    const std::string seg_path   = argv[2];
    const std::string enc_path   = argv[3];
    const std::string dec_path   = argv[4];
    const std::string tok_path   = argv[5];

    double      threshold   = 0.90;
    std::string report_path;

    for (int i = 6; i < argc; ++i) {
        if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc)
            threshold = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--report") == 0 && i + 1 < argc)
            report_path = argv[++i];
    }

    // ── Load engine ──────────────────────────────────────────────────────────
    printf("Loading OMR engine...\n");
    OmrEngineHandle engine = omr_create(seg_path.c_str(), enc_path.c_str(),
                                         dec_path.c_str(), tok_path.c_str());
    if (!engine) {
        fprintf(stderr, "ERROR: omr_create() failed. Check model paths.\n");
        return 2;
    }
    printf("Engine ready.\n\n");

    // ── Collect test samples ─────────────────────────────────────────────────
    auto stems = list_image_stems(test_dir);
    if (stems.empty()) {
        fprintf(stderr, "ERROR: no image files found in %s\n", test_dir.c_str());
        omr_destroy(engine);
        return 2;
    }

    // ── Run evaluation ────────────────────────────────────────────────────────
    std::vector<SampleResult> results;
    results.reserve(stems.size());

    int n_valid = 0;  // samples with both image and GT

    for (const auto& stem : stems) {
        SampleResult res;
        res.stem = stem;

        // Ground-truth label
        std::string gt_path = test_dir + "/" + stem + ".json";
        auto gt_tokens = parse_gt_json(gt_path);
        if (gt_tokens.empty()) {
            res.error_msg = "no ground truth";
            results.push_back(res);
            continue;
        }
        res.has_gt  = true;
        res.gt_len  = (int)gt_tokens.size();
        ++n_valid;

        // Image bytes
        std::string img_path = find_image(test_dir, stem);
        if (img_path.empty()) {
            res.error_msg = "image file not found";
            results.push_back(res);
            continue;
        }

        auto img_bytes = read_bytes(img_path);
        if (img_bytes.empty()) {
            res.error_msg = "failed to read image";
            results.push_back(res);
            continue;
        }

        // Run inference
        OmrResultC* r = omr_process(engine,
                                     img_bytes.data(),
                                     (int32_t)img_bytes.size());

        if (!r || !r->success) {
            res.error_msg = r ? r->error : "null result";
            omr_free_result(r);
            results.push_back(res);
            continue;
        }

        res.infer_ok = true;

        // Collect predicted token texts
        std::vector<std::string> pred_tokens;
        pred_tokens.reserve(r->token_count);
        for (int i = 0; i < r->token_count; ++i)
            pred_tokens.emplace_back(r->tokens[i].text);
        res.pred_len = (int)pred_tokens.size();

        omr_free_result(r);

        // ── Full-sequence TER ────────────────────────────────────────────────
        res.edit_dist = levenshtein(pred_tokens, gt_tokens);
        res.ter       = (res.gt_len > 0)
                        ? (double)res.edit_dist / res.gt_len
                        : (pred_tokens.empty() ? 0.0 : 1.0);
        res.accuracy  = std::max(0.0, 1.0 - res.ter);

        // ── Note-only TER ────────────────────────────────────────────────────
        auto pred_notes = note_only(pred_tokens);
        auto gt_notes   = note_only(gt_tokens);
        res.note_gt_len = (int)gt_notes.size();
        res.note_edit   = levenshtein(pred_notes, gt_notes);
        res.note_ter    = (res.note_gt_len > 0)
                          ? (double)res.note_edit / res.note_gt_len
                          : (pred_notes.empty() ? 0.0 : 1.0);

        results.push_back(res);
    }

    omr_destroy(engine);

    // ── Compute aggregate metrics ─────────────────────────────────────────────
    double sum_acc      = 0.0, sum_note_acc = 0.0;
    int    n_pass       = 0,   n_inferred   = 0;

    for (const auto& res : results) {
        if (!res.has_gt || !res.infer_ok) continue;
        ++n_inferred;
        sum_acc      += res.accuracy;
        sum_note_acc += std::max(0.0, 1.0 - res.note_ter);
        if (res.accuracy >= threshold) ++n_pass;
    }

    const double mean_acc      = n_inferred ? sum_acc      / n_inferred : 0.0;
    const double mean_note_acc = n_inferred ? sum_note_acc / n_inferred : 0.0;
    const bool   global_pass   = (mean_acc >= threshold);

    // ── Print report ──────────────────────────────────────────────────────────
    const char* SEP =
        "=================================================================\n";

    printf("%s", SEP);
    printf("  OMR Evaluation Report\n");
    printf("%s", SEP);
    printf("  Test directory : %s\n", test_dir.c_str());
    printf("  Total images   : %d  (with GT: %d  evaluated: %d)\n",
           (int)stems.size(), n_valid, n_inferred);
    printf("  Threshold      : %.1f %% accuracy (TER <= %.1f %%)\n",
           threshold * 100.0, (1.0 - threshold) * 100.0);
    printf("\n");

    if (global_pass)
        printf("  OVERALL RESULT : PASS\n");
    else
        printf("  OVERALL RESULT : FAIL\n");

    printf("  Mean accuracy  : %5.1f %%  (TER: %.1f %%)\n",
           mean_acc * 100.0, (1.0 - mean_acc) * 100.0);
    printf("  Note accuracy  : %5.1f %%  (note TER: %.1f %%)\n",
           mean_note_acc * 100.0, (1.0 - mean_note_acc) * 100.0);
    printf("  Per-threshold  : %d / %d samples pass\n", n_pass, n_inferred);
    printf("\n");

    printf("  %-20s  %6s  %6s  %8s  %8s  %5s\n",
           "Sample", "GT", "Pred", "TER", "NoteTER", "Pass");
    printf("  %-20s  %6s  %6s  %8s  %8s  %5s\n",
           "------", "--", "----", "---", "-------", "----");

    for (const auto& res : results) {
        if (!res.has_gt) {
            printf("  %-20s  (no ground truth)\n", res.stem.c_str());
            continue;
        }
        if (!res.infer_ok) {
            printf("  %-20s  ERROR: %s\n", res.stem.c_str(), res.error_msg.c_str());
            continue;
        }
        printf("  %-20s  %6d  %6d  %7.1f%%  %7.1f%%  %s\n",
               res.stem.c_str(),
               res.gt_len,
               res.pred_len,
               res.ter      * 100.0,
               res.note_ter * 100.0,
               (res.accuracy >= threshold) ? "PASS" : "FAIL");
    }
    printf("%s", SEP);

    // ── Optional CSV report ───────────────────────────────────────────────────
    if (!report_path.empty()) {
        std::ofstream csv(report_path);
        if (csv) {
            csv << "sample,gt_len,pred_len,edit_dist,ter,note_gt_len"
                   ",note_edit,note_ter,accuracy,pass\n";
            for (const auto& res : results) {
                if (!res.has_gt || !res.infer_ok) continue;
                csv << res.stem        << ","
                    << res.gt_len      << ","
                    << res.pred_len    << ","
                    << res.edit_dist   << ","
                    << res.ter         << ","
                    << res.note_gt_len << ","
                    << res.note_edit   << ","
                    << res.note_ter    << ","
                    << res.accuracy    << ","
                    << (res.accuracy >= threshold ? 1 : 0) << "\n";
            }
            printf("  CSV report written to: %s\n", report_path.c_str());
        }
    }

    return global_pass ? 0 : 1;
}
