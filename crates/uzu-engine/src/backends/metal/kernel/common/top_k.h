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

static inline uint top_k_bf16_key(ushort bits) { return uint(bits ^ ((bits & 0x8000) ? 0xffff : 0x8000)); }
