#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/attention.h"
#include "../generated/trie.h"
#include "matrix_layout.h"
#include "matrix_mxu.h"
#include "matrix_simd.h"

using namespace metal;

template <typename ScaleType, uint HEAD_DIM, bool USE_MXU, uint NUM_SIMDGROUPS, bool IS_SPARSE>
VARIANTS(ScaleType, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
VARIANTS(USE_MXU, false, true)
VARIANTS(NUM_SIMDGROUPS, 1, 2, 4, 6, 8)
VARIANTS(IS_SPARSE, false, true)
CONSTRAINT(!USE_MXU || NUM_SIMDGROUPS % (HEAD_DIM / 64) == 0)
CONSTRAINT(HEAD_DIM != 256 || NUM_SIMDGROUPS % 2 == 0)
CONSTRAINT(HEAD_DIM != 512 || NUM_SIMDGROUPS % 4 == 0)
CONSTRAINT(USE_MXU || NUM_SIMDGROUPS != 1)
CONSTRAINT(!IS_SPARSE || (HEAD_DIM == 256 && NUM_SIMDGROUPS == HEAD_DIM / 64))
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
    device bfloat* partial_values OPTIONAL(is_split),
    device float* partial_maxima OPTIONAL(is_split),
    device float* partial_denominators OPTIONAL(is_split),
    device bfloat* output OPTIONAL(!is_split),
    const device bfloat* gate OPTIONAL(has_gate),
    const constant uint& gate_row_stride OPTIONAL(has_gate),
    const device bfloat* sinks OPTIONAL(has_sinks),
    const constant uzu::attention::AttnParams& params,
    const constant uint& query_tiles,
    const constant uint& num_splits,
    const constant uint& split_base,
    const constant uint& split_count,
    const constant uint& trie_nodes,
    const device int* selected_blocks OPTIONAL(IS_SPARSE),
    const constant uint& blocks_per_query OPTIONAL(IS_SPARSE),
    const bool is_causal SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const bool mask_free SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool is_split SPECIALIZE,
    const bool has_gate SPECIALIZE,
    const bool has_sinks SPECIALIZE,
    const uint32_t num_kv_heads SPECIALIZE,
    const uint compression_ratio SPECIALIZE_IF(IS_SPARSE),
    threadgroup float probability_exchange[AttentionMatrixScratch<HEAD_DIM, USE_MXU, NUM_SIMDGROUPS, IS_SPARSE>::EXCHANGE_FLOATS],
    // Staged queries; output tile after the loop.
    threadgroup uint4 tile_scratch[AttentionMatrixScratch<HEAD_DIM, USE_MXU, NUM_SIMDGROUPS, IS_SPARSE>::TILE_SCRATCH_VECTORS],
    // Each row's visible window; row stats after the loop.
    threadgroup uint2 row_windows[AttentionMatrixScratch<HEAD_DIM, USE_MXU, NUM_SIMDGROUPS, IS_SPARSE>::ROW_WINDOWS],
    const ThreadContext thread_context,
    const uint tile_split_index GROUPS(query_tiles * split_count),
    const uint kv_head_index GROUPS(num_kv_heads),
    const uint thread_index THREADS(NUM_SIMDGROUPS * METAL_SIMD_SIZE)
) {
  if constexpr (USE_MXU) {
    attention_matrix_mxu<ScaleType, HEAD_DIM, NUM_SIMDGROUPS, IS_SPARSE>(
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
        output,
        gate,
        gate_row_stride,
        sinks,
        params,
        query_tiles,
        num_splits,
        split_base,
        trie_nodes,
        mask_free,
        is_trie,
        is_split,
        has_gate,
        has_sinks,
        selected_blocks,
        blocks_per_query,
        compression_ratio,
        num_kv_heads,
        probability_exchange,
        row_windows,
        tile_scratch,
        thread_context,
        tile_split_index,
        kv_head_index
    );
  } else {
    attention_matrix_simd<ScaleType, HEAD_DIM, NUM_SIMDGROUPS, IS_SPARSE>(
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
        output,
        gate,
        gate_row_stride,
        sinks,
        params,
        query_tiles,
        num_splits,
        trie_nodes,
        is_trie,
        is_split,
        has_gate,
        has_sinks,
        selected_blocks,
        blocks_per_query,
        compression_ratio,
        num_kv_heads,
        probability_exchange,
        reinterpret_cast<threadgroup half*>(tile_scratch),
        reinterpret_cast<threadgroup ulong*>(row_windows),
        thread_context,
        tile_split_index,
        kv_head_index
    );
  }
}
