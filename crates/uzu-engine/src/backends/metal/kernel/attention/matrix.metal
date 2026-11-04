#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/attention.h"
#include "../generated/trie.h"
#include "matrix_layout.h"
#include "matrix_mxu.h"
#include "matrix_simd.h"

using namespace metal;

template <typename ScaleType, uint HEAD_DIM, bool USE_MXU, uint NUM_SIMDGROUPS>
VARIANTS(ScaleType, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
VARIANTS(USE_MXU, false, true)
VARIANTS(NUM_SIMDGROUPS, 1, 2, 4, 6, 8)
CONSTRAINT(!USE_MXU || NUM_SIMDGROUPS % (HEAD_DIM / 64) == 0)
CONSTRAINT(HEAD_DIM != 256 || NUM_SIMDGROUPS % 2 == 0)
CONSTRAINT(HEAD_DIM != 512 || NUM_SIMDGROUPS % 4 == 0)
KERNEL(AttentionMatrix)(
    const device int8_t* query_codes,
    const device ScaleType* query_scales,
    const device int8_t* key_codes,
    const device ScaleType* key_scales,
    const device uchar* value_codes,
    const device ScaleType* value_scales,
    const device uzu::trie::TrieNode* trie OPTIONAL(is_trie),
    const constant uzu::ring::RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    device bfloat* partial_values,
    device float* partial_maxima,
    device float* partial_denominators,
    const constant uzu::attention::AttnParams& params,
    const constant uint& query_tiles,
    const constant uint& num_splits,
    const constant uint& split_base,
    const constant uint& split_count,
    const constant uint& trie_nodes,
    const bool is_causal SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const bool mask_free SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const uint32_t num_kv_heads SPECIALIZE,
    threadgroup float probability_exchange[AttentionMatrixScratch<HEAD_DIM, USE_MXU, NUM_SIMDGROUPS>::EXCHANGE_FLOATS],
    threadgroup ulong row_masks[AttentionMatrixScratch<HEAD_DIM, USE_MXU, NUM_SIMDGROUPS>::ROW_MASKS],
    threadgroup uint4 query_stage[AttentionMatrixScratch<HEAD_DIM, USE_MXU, NUM_SIMDGROUPS>::STAGING_VECTORS],
    const ThreadContext thread_context,
    const uint tile_split_index GROUPS(query_tiles * split_count),
    const uint kv_head_index GROUPS(num_kv_heads),
    const uint thread_index THREADS(NUM_SIMDGROUPS * METAL_SIMD_SIZE)
) {
  if constexpr (USE_MXU) {
    attention_matrix_mxu<ScaleType, HEAD_DIM, NUM_SIMDGROUPS>(
        query_codes,
        query_scales,
        key_codes,
        key_scales,
        value_codes,
        value_scales,
        trie,
        ring_params,
        sliding_window_size,
        is_causal,
        is_kv_cache_ring,
        is_sliding_window,
        partial_values,
        partial_maxima,
        partial_denominators,
        params,
        query_tiles,
        num_splits,
        split_base,
        trie_nodes,
        mask_free,
        is_trie,
        num_kv_heads,
        probability_exchange,
        row_masks,
        query_stage,
        thread_context,
        tile_split_index,
        kv_head_index
    );
  } else {
    attention_matrix_simd<ScaleType, HEAD_DIM, NUM_SIMDGROUPS>(
        query_codes,
        query_scales,
        key_codes,
        key_scales,
        value_codes,
        value_scales,
        trie,
        ring_params,
        sliding_window_size,
        is_causal,
        is_kv_cache_ring,
        is_sliding_window,
        partial_values,
        partial_maxima,
        partial_denominators,
        params,
        query_tiles,
        num_splits,
        trie_nodes,
        is_trie,
        num_kv_heads,
        probability_exchange,
        reinterpret_cast<threadgroup half*>(query_stage),
        thread_context,
        tile_split_index,
        kv_head_index
    );
  }
}
