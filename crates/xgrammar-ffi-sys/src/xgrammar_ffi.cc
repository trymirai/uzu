#include "xgrammar_ffi.h"

#include <exception>
#include <string>

// XGrammar C++ headers
#include <xgrammar/grammar_matcher.h>
#include <xgrammar/compiled_grammar.h>
#include <xgrammar/tokenizer_info.h>
#include <xgrammar/serialization/json.h>

// DLPack
#include <dlpack/dlpack.h>

using xgrammar::CompiledGrammar;
using xgrammar::GrammarMatcher;
using xgrammar::TokenizerInfo;

extern "C" {

xgr_matcher_t xgr_matcher_new_from_json(const char* ti_json,
                                        size_t ti_json_len,
                                        const char* cg_json,
                                        size_t cg_json_len) {
  try {
    std::string ti_str(ti_json, ti_json_len);
    std::string cg_str(cg_json, cg_json_len);

    TokenizerInfo ti = TokenizerInfo::deserialize_json(ti_str);
    CompiledGrammar cg = CompiledGrammar::deserialize_json(cg_str, ti);

    return static_cast<void*>(new GrammarMatcher(std::move(cg)));
  } catch (...) {
    return nullptr;
  }
}

void xgr_matcher_free(xgr_matcher_t m) {
  delete static_cast<GrammarMatcher*>(m);
}

int xgr_fill_next_token_bitmask_packed(
    xgr_matcher_t m,
    uint32_t* data, int batch, int nwords32, int index, int debug) {
  try {
    int64_t shape[2] = {batch, nwords32};

    DLTensor t{};
    t.data = data;
    t.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
    t.ndim = 2;
    t.dtype = DLDataType{.code = kDLInt, .bits = 32, .lanes = 1};
    t.shape = shape;
    t.strides = nullptr;
    t.byte_offset = 0;

    auto* gm = static_cast<GrammarMatcher*>(m);
    return gm->FillNextTokenBitmask(&t, index, debug) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

int xgr_accept_token(xgr_matcher_t m, int32_t token_id, int debug) {
  try {
    auto* gm = static_cast<GrammarMatcher*>(m);
    return gm->AcceptToken(token_id, debug) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

}  // extern "C"
