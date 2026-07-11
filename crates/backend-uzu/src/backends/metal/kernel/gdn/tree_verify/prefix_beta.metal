#include <metal_stdlib>
#include "../../activation/activations.h"
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../generated/trie.h"

#define ROWS_PER_THREADGROUP 4u

using namespace metal;
using namespace uzu::trie;

// trie:        [B, T] TrieNode — col is an ancestor-or-self of row iff
//              trie[col].trie_start <= row <= trie[col].trie_end.
// a_transposed: [B, HV, T]
// b:           [B, T, HV]
// prefix,beta: [B, T, HV] fp32
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(BuildPrefixBeta)(
    const device TrieNode* trie,
    const device T* a_transposed,
    const device T* b,
    const device float* a_log,
    const device float* dt_bias,
    device float* prefix,
    device float* beta,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& value_heads,
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint row_idx GROUPS(tree_size),
    const uint head_group_idx GROUPS(value_heads.div_ceil(ROWS_PER_THREADGROUP)),
    const uint tid THREADS(ROWS_PER_THREADGROUP * METAL_SIMD_SIZE)
) {
  const uint simd_group = thread_context.simdgroup_index;
  const uint lane_id = thread_context.simd_lane_id;
  const uint head_idx = head_group_idx * ROWS_PER_THREADGROUP + simd_group;
  if (head_idx >= value_heads) {
    return;
  }

  const uint batch_offset = batch_idx * tree_size * value_heads;
  const uint trie_offset = batch_idx * tree_size;
  const uint out_idx = batch_offset + row_idx * value_heads + head_idx;

  const float scale = exp(float(a_log[head_idx]));
  const float dt = float(dt_bias[head_idx]);
  float partial = 0.0f;
  for (uint col_idx = lane_id; col_idx < tree_size; col_idx += METAL_SIMD_SIZE) {
    const TrieNode node = trie[trie_offset + col_idx];
    if (row_idx >= node.trie_start && row_idx <= node.trie_end) {
      const uint a_idx = batch_offset + head_idx * tree_size + col_idx;
      partial -= scale * activate_softplus(float(a_transposed[a_idx]) + dt);
    }
  }

  const float sum = simd_sum(partial);
  if (lane_id == 0) {
    const float b_val = float(b[out_idx]);
    beta[out_idx] = 1.0f / (1.0f + exp(-b_val));
    prefix[out_idx] = sum;
  }
}
