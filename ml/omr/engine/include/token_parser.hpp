#pragma once
#include "types.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace omr {

/**
 * TokenParser – loads the unified vocabulary and converts token IDs ↔ strings.
 *
 * Vocabulary file format (tokenizer.json):
 *   { "<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "clef-G": 4, "note-C4": 112, "dur-1/4": 117, ... }
 *
 * Token naming conventions (our DeepScore format):
 *   note-{pitch}{oct}          e.g. note-C4, note-F#5   (pitch only — no duration)
 *   dur-{dur}                  e.g. dur-1/4             (always follows a note-* token)
 *   chord-{pitch}{oct}         e.g. chord-E4            (continuation of prev note)
 *   rest-{dur}                 e.g. rest-1/4
 *   clef-{G|F|C}               e.g. clef-G
 *   key-{name}                 e.g. key-C, key-Bb
 *   time-{num}/{den}           e.g. time-4/4
 *   barline / barline-final / barline-double / ...
 *   <PAD> / <SOS> / <EOS> / <UNK>
 *
 * note was previously a single combined note-{pitch}-{dur} token; it was split
 * into note-{pitch} + dur-{dur} to reduce vocab size (1013 -> 258) and improve
 * pitch classification accuracy. See docs/round3-phase2-retrain.md.
 *
 * Duration tokens (fraction of whole note):
 *   1/1  3/4  1/2  3/8  1/4  3/16  1/8  3/32  1/16  1/32
 */
class TokenParser {
public:
    static constexpr int PAD_ID = 0;
    static constexpr int SOS_ID = 1;
    static constexpr int EOS_ID = 2;
    static constexpr int UNK_ID = 3;

    /**
     * Load vocabulary from a tokenizer.json file.
     * Throws std::runtime_error on failure.
     */
    explicit TokenParser(const std::string& tokenizer_path);

    int         vocab_size()              const { return (int)id_to_text_.size(); }
    std::string id_to_token(int id)       const;
    int         token_to_id(const std::string& tok) const;

    /**
     * Convert a raw decoder output (list of IDs including SOS/EOS/PAD)
     * into clean OmrToken objects, stripping control tokens.
     */
    std::vector<OmrToken> decode_ids(const std::vector<int32_t>& ids) const;

    /**
     * Helper: is this token ID a note token (starts with "note-")?
     */
    bool is_note(int id) const;
    bool is_rest(int id) const;
    bool is_barline(int id) const;

private:
    std::unordered_map<int, std::string> id_to_text_;
    std::unordered_map<std::string, int> text_to_id_;

    // Parse flat JSON  { "token": id, ... } without a JSON library.
    void load_json(const std::string& path);
};

} // namespace omr
