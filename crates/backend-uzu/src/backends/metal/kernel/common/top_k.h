#pragma once

#include <metal_stdlib>
using namespace metal;

constant uint TOP_K_SIGN_BIT = 0x80000000u;

static inline uint top_k_score_key(float score) {
  const uint bits = as_type<uint>(score);
  return (bits & TOP_K_SIGN_BIT) ? ~bits : (bits ^ TOP_K_SIGN_BIT);
}

static inline float top_k_score_from_key(uint key) {
  const uint bits = (key & TOP_K_SIGN_BIT) ? (key ^ TOP_K_SIGN_BIT) : ~key;
  return as_type<float>(bits);
}

// Larger keys follow the f32 total order; equal scores prefer smaller indices.
static inline ulong top_k_ordered_key(float score, uint index, uint index_bits) {
  const ulong index_mask = (1ul << index_bits) - 1ul;
  return (ulong(top_k_score_key(score)) << index_bits) | (index_mask - ulong(index));
}

static inline bool top_k_better(
    uint left_score,
    uint left_token,
    uint left_index,
    uint right_score,
    uint right_token,
    uint right_index
) {
  return left_score > right_score ||
         (left_score == right_score &&
          (left_token < right_token || (left_token == right_token && left_index < right_index)));
}
