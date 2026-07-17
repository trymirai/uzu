#pragma once

#include <metal_stdlib>
using namespace metal;

static inline uint top_k_score_key(float score) {
  const uint bits = as_type<uint>(score);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
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
