#include "xgrammar_ffi.h"

extern "C" {

xgr_matcher_t xgr_matcher_new_from_json(const char*, size_t, const char*, size_t) {
    return (xgr_matcher_t)0x1; // non-null dummy
}

void xgr_matcher_free(xgr_matcher_t) {}

int xgr_fill_next_token_bitmask_packed(xgr_matcher_t, uint32_t* data, int batch, int nwords32, int index, int) {
    // Allow-all mask for row 0
    if (index == 0 && batch >= 1) {
        for (int i = 0; i < nwords32; ++i) data[i] = 0xFFFFFFFFu;
    }
    return 1;
}

int xgr_accept_token(xgr_matcher_t, int32_t, int) { return 1; }

}