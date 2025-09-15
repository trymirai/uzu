#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void* xgr_matcher_t;

// lifecycle
xgr_matcher_t xgr_matcher_new_from_json(const char* tokenizer_info_json,
                                        size_t tokenizer_info_json_len,
                                        const char* compiled_grammar_json,
                                        size_t compiled_grammar_json_len);
void xgr_matcher_free(xgr_matcher_t m);

// operations
int xgr_fill_next_token_bitmask_packed(xgr_matcher_t m,
    uint32_t* data,       // out: packed bitset buffer
    int batch,            // rows in the bitmask
    int nwords32,         // cols per row = ceil(vocab/32)
    int index,            // row to fill
    int debug_print);     // 0/1

int xgr_accept_token(xgr_matcher_t m, int32_t token_id, int debug_print);

#ifdef __cplusplus
}
#endif


