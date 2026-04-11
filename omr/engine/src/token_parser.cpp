#include "token_parser.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace omr {

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────

TokenParser::TokenParser(const std::string& tokenizer_path) {
    load_json(tokenizer_path);
}

// ─────────────────────────────────────────────────────────────────────────────
//  ID ↔ string lookups
// ─────────────────────────────────────────────────────────────────────────────

std::string TokenParser::id_to_token(int id) const {
    auto it = id_to_text_.find(id);
    return (it != id_to_text_.end()) ? it->second : "<UNK>";
}

int TokenParser::token_to_id(const std::string& tok) const {
    auto it = text_to_id_.find(tok);
    return (it != text_to_id_.end()) ? it->second : UNK_ID;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Decode a raw ID sequence to OmrToken list
// ─────────────────────────────────────────────────────────────────────────────

std::vector<OmrToken> TokenParser::decode_ids(const std::vector<int32_t>& ids) const {
    std::vector<OmrToken> out;
    out.reserve(ids.size());

    for (int32_t id : ids) {
        // Skip control tokens.
        if (id == PAD_ID || id == SOS_ID || id == EOS_ID) continue;

        OmrToken tok;
        tok.id   = id;
        tok.text = id_to_token(id);
        out.push_back(std::move(tok));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Token type helpers
// ─────────────────────────────────────────────────────────────────────────────

bool TokenParser::is_note(int id) const {
    auto it = id_to_text_.find(id);
    if (it == id_to_text_.end()) return false;
    return it->second.rfind("note-", 0) == 0;
}

bool TokenParser::is_rest(int id) const {
    auto it = id_to_text_.find(id);
    if (it == id_to_text_.end()) return false;
    return it->second.rfind("rest-", 0) == 0;
}

bool TokenParser::is_barline(int id) const {
    auto it = id_to_text_.find(id);
    if (it == id_to_text_.end()) return false;
    return it->second.rfind("barline", 0) == 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Minimal flat-JSON parser
//
//  Handles the format produced by generate_dataset.py:
//    { "token_string" : integer_id , ... }
//
//  No third-party JSON library is used.
// ─────────────────────────────────────────────────────────────────────────────

void TokenParser::load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("TokenParser: cannot open " + path);

    std::string content((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

    // Walk character by character, extract key-value pairs.
    // State machine: look for " to start a key, then : then integer value.
    enum State { SEEK_KEY, IN_KEY, SEEK_COLON, SEEK_VALUE, IN_VALUE } state = SEEK_KEY;
    std::string key, val_str;

    for (char ch : content) {
        switch (state) {
        case SEEK_KEY:
            if (ch == '"') { key.clear(); state = IN_KEY; }
            break;
        case IN_KEY:
            if (ch == '"') state = SEEK_COLON;
            else           key += ch;
            break;
        case SEEK_COLON:
            if (ch == ':') state = SEEK_VALUE;
            break;
        case SEEK_VALUE:
            if (std::isdigit(ch)) { val_str = std::string(1, ch); state = IN_VALUE; }
            else if (ch == '"' || ch == '{') { state = SEEK_KEY; } // nested / bad value
            break;
        case IN_VALUE:
            if (std::isdigit(ch)) {
                val_str += ch;
            } else {
                // End of value.
                int id = std::stoi(val_str);
                id_to_text_[id]  = key;
                text_to_id_[key] = id;
                state = SEEK_KEY;
            }
            break;
        }
    }

    if (id_to_text_.empty())
        throw std::runtime_error("TokenParser: no tokens loaded from " + path);
}

} // namespace omr
