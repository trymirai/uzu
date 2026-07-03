#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define CHUNKED_APPLY_THREADS 128
#define CHUNKED_APPLY_DV_PER_SIMDGROUP 4
#define CHUNKED_STATE_A2_VALUE_TILE 32
#define CHUNKED_STATE_A2_TOKEN_TILE 16
#define CHUNKED_STATE_A2_KEY_TILE 32
#define CHUNKED_OUTPUT_VALUE_COLS 32

static_assert(CHUNKED_APPLY_THREADS % METAL_SIMD_SIZE == 0, "thread count must be a multiple of simd size");

template <bool RECOMPUTE_G>
METAL_FUNC float chunked_g(
    device const float* g_or_log_decay,
    uint token_base,
    uint local_t,
    uint num_v_heads,
    uint hv_idx
) {
  if constexpr (RECOMPUTE_G) {
    float acc = 0.0f;
    for (uint i = 0; i <= local_t; ++i) {
      acc += g_or_log_decay[(token_base + i) * num_v_heads + hv_idx];
    }
    return acc;
  } else {
    return g_or_log_decay[(token_base + local_t) * num_v_heads + hv_idx];
  }
}

template <uint HEAD_K_DIM, uint CHUNK_SIZE, bool RECOMPUTE_G>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(RECOMPUTE_G, false, true)
PUBLIC KERNEL(DeltaNetChunkedStateA)(
    device const float* k_norm,
    device const float* w,
    device const float* u,
    device const float* g_or_log_decay,
    device float* state,
    device float* h,
    device float* v_new,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    constant const uint& num_dv_groups,
    const uint hv_idx GROUPS(num_v_heads),
    const uint dv_group GROUPS(num_dv_groups),
    const uint tid THREADS(CHUNKED_APPLY_THREADS)
) {
  static_assert(HEAD_K_DIM % METAL_SIMD_SIZE == 0, "HEAD_K_DIM must be a multiple of simd size");
  constexpr uint ELEMS = HEAD_K_DIM / METAL_SIMD_SIZE;
  constexpr uint NUM_SG = CHUNKED_APPLY_THREADS / METAL_SIMD_SIZE;
  static_assert(ELEMS == 4, "chunked apply uses float4 K lanes");

  const uint lane = tid % METAL_SIMD_SIZE;
  const uint dv_local = tid / METAL_SIMD_SIZE;
  const uint dv_idx = (dv_group * NUM_SG + dv_local) * CHUNKED_APPLY_DV_PER_SIMDGROUP;
  const uint dk_base = lane * ELEMS;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint num_chunks = (suffix_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

  const bool valid_dv0 = dv_idx + 0 < head_v_dim;
  const bool valid_dv1 = dv_idx + 1 < head_v_dim;
  const bool valid_dv2 = dv_idx + 2 < head_v_dim;
  const bool valid_dv3 = dv_idx + 3 < head_v_dim;

  float4 s0 = 0.0f;
  float4 s1 = 0.0f;
  float4 s2 = 0.0f;
  float4 s3 = 0.0f;
  if (valid_dv0) {
    const device float* state_row = state + (hv_idx * head_v_dim + dv_idx + 0) * HEAD_K_DIM + dk_base;
    s0 = *reinterpret_cast<const device float4*>(state_row);
  }
  if (valid_dv1) {
    const device float* state_row = state + (hv_idx * head_v_dim + dv_idx + 1) * HEAD_K_DIM + dk_base;
    s1 = *reinterpret_cast<const device float4*>(state_row);
  }
  if (valid_dv2) {
    const device float* state_row = state + (hv_idx * head_v_dim + dv_idx + 2) * HEAD_K_DIM + dk_base;
    s2 = *reinterpret_cast<const device float4*>(state_row);
  }
  if (valid_dv3) {
    const device float* state_row = state + (hv_idx * head_v_dim + dv_idx + 3) * HEAD_K_DIM + dk_base;
    s3 = *reinterpret_cast<const device float4*>(state_row);
  }

  for (uint chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const uint token_base = chunk_idx * CHUNK_SIZE;
    const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;

    if (valid_dv0) {
      device float* h_row = h + ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + dv_idx + 0) * HEAD_K_DIM + dk_base;
      *reinterpret_cast<device float4*>(h_row) = s0;
    }
    if (valid_dv1) {
      device float* h_row = h + ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + dv_idx + 1) * HEAD_K_DIM + dk_base;
      *reinterpret_cast<device float4*>(h_row) = s1;
    }
    if (valid_dv2) {
      device float* h_row = h + ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + dv_idx + 2) * HEAD_K_DIM + dk_base;
      *reinterpret_cast<device float4*>(h_row) = s2;
    }
    if (valid_dv3) {
      device float* h_row = h + ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + dv_idx + 3) * HEAD_K_DIM + dk_base;
      *reinterpret_cast<device float4*>(h_row) = s3;
    }

#pragma clang loop unroll(disable)
    for (uint local_t = 0; local_t < CHUNK_SIZE; ++local_t) {
      float4 correction = 0.0f;
      if (local_t < valid_tokens) {
        const device float* w_row =
            w + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t) * HEAD_K_DIM + dk_base;
        const float4 w_vec = *reinterpret_cast<const device float4*>(w_row);
        correction = float4(dot(w_vec, s0), dot(w_vec, s1), dot(w_vec, s2), dot(w_vec, s3));
        correction = simd_sum(correction);
      }

      if (lane == 0) {
        device float* v_row = v_new + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t) * head_v_dim + dv_idx;
        const device float* u_row =
            u + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t) * head_v_dim + dv_idx;
        if (valid_dv0) {
          v_row[0] = local_t < valid_tokens ? u_row[0] - correction[0] : 0.0f;
        }
        if (valid_dv1) {
          v_row[1] = local_t < valid_tokens ? u_row[1] - correction[1] : 0.0f;
        }
        if (valid_dv2) {
          v_row[2] = local_t < valid_tokens ? u_row[2] - correction[2] : 0.0f;
        }
        if (valid_dv3) {
          v_row[3] = local_t < valid_tokens ? u_row[3] - correction[3] : 0.0f;
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_device);

    if (valid_tokens > 0) {
      const float g_last = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, valid_tokens - 1, num_v_heads, hv_idx);
      const float g_last_exp = fast::exp(g_last);
      s0 *= g_last_exp;
      s1 *= g_last_exp;
      s2 *= g_last_exp;
      s3 *= g_last_exp;

#pragma clang loop unroll(disable)
      for (uint local_t = 0; local_t < valid_tokens; ++local_t) {
        const uint token = token_base + local_t;
        const float decay_scale =
            fast::exp(g_last - chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_t, num_v_heads, hv_idx));
        const device float* k_row = k_norm + token * key_dim + hk_idx * HEAD_K_DIM + dk_base;
        const float4 k_vec = *reinterpret_cast<const device float4*>(k_row);
        const device float* v_row =
            v_new + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t) * head_v_dim + dv_idx;
        if (valid_dv0) {
          s0 += k_vec * (v_row[0] * decay_scale);
        }
        if (valid_dv1) {
          s1 += k_vec * (v_row[1] * decay_scale);
        }
        if (valid_dv2) {
          s2 += k_vec * (v_row[2] * decay_scale);
        }
        if (valid_dv3) {
          s3 += k_vec * (v_row[3] * decay_scale);
        }
      }
    }
  }

  if (valid_dv0) {
    device float* state_row = state + (hv_idx * head_v_dim + dv_idx + 0) * HEAD_K_DIM + dk_base;
    *reinterpret_cast<device float4*>(state_row) = s0;
  }
  if (valid_dv1) {
    device float* state_row = state + (hv_idx * head_v_dim + dv_idx + 1) * HEAD_K_DIM + dk_base;
    *reinterpret_cast<device float4*>(state_row) = s1;
  }
  if (valid_dv2) {
    device float* state_row = state + (hv_idx * head_v_dim + dv_idx + 2) * HEAD_K_DIM + dk_base;
    *reinterpret_cast<device float4*>(state_row) = s2;
  }
  if (valid_dv3) {
    device float* state_row = state + (hv_idx * head_v_dim + dv_idx + 3) * HEAD_K_DIM + dk_base;
    *reinterpret_cast<device float4*>(state_row) = s3;
  }
}

template <typename WU, typename H, uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(WU, float, bfloat)
VARIANTS(H, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
PUBLIC KERNEL(DeltaNetChunkedStateA2Vnew)(
    device const WU* w,
    device const WU* u,
    device const float* state,
    device H* h,
    device float* v_new,
    constant const uint& num_v_heads,
    constant const uint& head_v_dim,
    constant const uint& suffix_len,
    constant const uint& chunk_idx,
    const uint hv_idx GROUPS(num_v_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(CHUNKED_STATE_A2_VALUE_TILE)),
    const uint token_tile_idx GROUPS(CHUNK_SIZE.div_ceil(CHUNKED_STATE_A2_TOKEN_TILE)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = MxuFragmentOps<>;
  constexpr ushort TOKEN_FRAGMENTS = CHUNKED_STATE_A2_TOKEN_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort VALUE_FRAGMENTS = CHUNKED_STATE_A2_VALUE_TILE / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using WFragment = OperandFragment<float, TOKEN_FRAGMENTS, 1, Ops>;
  using SFragment = OperandFragment<float, 1, VALUE_FRAGMENTS, Ops>;

  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  const uint row_base = token_tile_idx * CHUNKED_STATE_A2_TOKEN_TILE;
  const uint valid_rows =
      row_base < valid_tokens ? min(uint(CHUNKED_STATE_A2_TOKEN_TILE), valid_tokens - row_base) : 0u;
  const uint value_base = value_tile_idx * CHUNKED_STATE_A2_VALUE_TILE;
  const uint valid_cols =
      value_base < head_v_dim ? min(uint(CHUNKED_STATE_A2_VALUE_TILE), head_v_dim - value_base) : 0u;

  if (valid_cols == 0) {
    return;
  }

  const uint state_base = (hv_idx * head_v_dim + value_base) * HEAD_K_DIM;
  if (token_tile_idx == 0) {
    device H* h_tile = h + ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + value_base) * HEAD_K_DIM;
    for (uint i = lane; i < valid_cols * HEAD_K_DIM; i += METAL_SIMD_SIZE) {
      h_tile[i] = H(state[state_base + i]);
    }
  }

  AccFragment acc;
  acc.clear();
  for (uint k0 = 0; k0 < HEAD_K_DIM; k0 += Ops::FRAGMENT_ROWS) {
    WFragment w_frag;
    SFragment s_frag;
    const device WU* w_tile = w + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + row_base) * HEAD_K_DIM + k0;
    w_frag.load_from(lane, fragment_source(w_tile, int(HEAD_K_DIM)).bounded(valid_rows, Ops::FRAGMENT_ROWS));
    s_frag.load_from(
        lane,
        fragment_source(state + state_base + k0, 1, int(HEAD_K_DIM)).bounded(Ops::FRAGMENT_ROWS, valid_cols)
    );
    fragment_mma(acc, w_frag, s_frag);
  }

  const device WU* u_tile = u + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + row_base) * head_v_dim + value_base;
  acc.map_coords(lane, [&](short row, short col, float correction) {
    if (uint(row) >= valid_rows || uint(col) >= valid_cols) {
      return 0.0f;
    }
    return u_tile[uint(row) * head_v_dim + uint(col)] - correction;
  });
  device float* v_tile = v_new + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + row_base) * head_v_dim + value_base;
  acc.store_safe(lane, v_tile, int(head_v_dim), short2(valid_cols, valid_rows));
}

template <uint HEAD_K_DIM, uint CHUNK_SIZE, bool RECOMPUTE_G>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(RECOMPUTE_G, false, true)
PUBLIC KERNEL(DeltaNetChunkedStateA2Update)(
    device const float* k_norm,
    device const float* g_or_log_decay,
    device const float* v_new,
    device float* state,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    constant const uint& chunk_idx,
    const uint hv_idx GROUPS(num_v_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(CHUNKED_STATE_A2_VALUE_TILE)),
    const uint key_tile_idx GROUPS(HEAD_K_DIM.div_ceil(CHUNKED_STATE_A2_KEY_TILE)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = MxuFragmentOps<>;
  constexpr ushort VALUE_FRAGMENTS = CHUNKED_STATE_A2_VALUE_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort KEY_FRAGMENTS = CHUNKED_STATE_A2_KEY_TILE / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, VALUE_FRAGMENTS, KEY_FRAGMENTS, Ops>;
  using VFragment = OperandFragment<float, VALUE_FRAGMENTS, 1, Ops>;
  using KFragment = OperandFragment<float, 1, KEY_FRAGMENTS, Ops>;

  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  const uint value_base = value_tile_idx * CHUNKED_STATE_A2_VALUE_TILE;
  const uint key_base = key_tile_idx * CHUNKED_STATE_A2_KEY_TILE;
  const uint valid_value_rows =
      value_base < head_v_dim ? min(uint(CHUNKED_STATE_A2_VALUE_TILE), head_v_dim - value_base) : 0u;
  const uint valid_key_cols = min(uint(CHUNKED_STATE_A2_KEY_TILE), uint(HEAD_K_DIM) - key_base);

  if (valid_tokens == 0 || valid_value_rows == 0) {
    return;
  }

  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const float g_last = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, valid_tokens - 1, num_v_heads, hv_idx);
  const float alpha = fast::exp(g_last);

  AccFragment acc;
  acc.clear();
  for (uint t0 = 0; t0 < CHUNK_SIZE; t0 += Ops::FRAGMENT_ROWS) {
    const uint valid_t = t0 < valid_tokens ? min(uint(Ops::FRAGMENT_ROWS), valid_tokens - t0) : 0u;
    VFragment v_frag;
    KFragment k_frag;
    const device float* v_tile =
        v_new + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + t0) * head_v_dim + value_base;
    v_frag.load_from(lane, fragment_source(v_tile, 1, int(head_v_dim)).bounded(valid_value_rows, valid_t));
    const device float* k_tile = k_norm + (token_base + t0) * key_dim + hk_idx * HEAD_K_DIM + key_base;
    k_frag.load_from(lane, fragment_source(k_tile, int(key_dim)).bounded(valid_t, valid_key_cols));
    k_frag.map_coords(lane, [&](short row, short col, float value) {
      if (uint(row) >= valid_t || uint(col) >= valid_key_cols) {
        return 0.0f;
      }
      const float g_t = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, t0 + uint(row), num_v_heads, hv_idx);
      return value * fast::exp(g_last - g_t);
    });
    fragment_mma(acc, v_frag, k_frag);
  }

  device float* state_tile = state + (hv_idx * head_v_dim + value_base) * HEAD_K_DIM + key_base;
  acc.map_coords(lane, [&](short row, short col, float value) {
    if (uint(row) >= valid_value_rows || uint(col) >= valid_key_cols) {
      return 0.0f;
    }
    return alpha * state_tile[uint(row) * HEAD_K_DIM + uint(col)] + value;
  });
  acc.store_safe(lane, state_tile, int(HEAD_K_DIM), short2(valid_key_cols, valid_value_rows));
}

template <uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
PUBLIC KERNEL(DeltaNetChunkedStateA2DecayScale)(
    device const float* g,
    device float* decay_scale,
    constant const uint& num_v_heads,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(CHUNKED_APPLY_THREADS)
) {
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  if (valid_tokens == 0) {
    return;
  }

  const float g_last = g[(token_base + valid_tokens - 1) * num_v_heads + hv_idx];
  device float* dst_chunk = decay_scale + (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE;

  for (uint local_t = tid; local_t < CHUNK_SIZE; local_t += CHUNKED_APPLY_THREADS) {
    if (local_t >= valid_tokens) {
      dst_chunk[local_t] = 0.0f;
      continue;
    }
    const uint token = token_base + local_t;
    dst_chunk[local_t] = fast::exp(g_last - g[token * num_v_heads + hv_idx]);
  }
}

template <uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
PUBLIC KERNEL(DeltaNetChunkedStateA2UpdateDecayScale)(
    device const float* k_norm,
    device const float* g,
    device const float* decay_scale,
    device const float* v_new,
    device float* state,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    constant const uint& chunk_idx,
    const uint hv_idx GROUPS(num_v_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(CHUNKED_STATE_A2_VALUE_TILE)),
    const uint key_tile_idx GROUPS(HEAD_K_DIM.div_ceil(CHUNKED_STATE_A2_KEY_TILE)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = MxuFragmentOps<>;
  constexpr ushort VALUE_FRAGMENTS = CHUNKED_STATE_A2_VALUE_TILE / Ops::FRAGMENT_ROWS;
  constexpr ushort KEY_FRAGMENTS = CHUNKED_STATE_A2_KEY_TILE / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, VALUE_FRAGMENTS, KEY_FRAGMENTS, Ops>;
  using VFragment = OperandFragment<float, VALUE_FRAGMENTS, 1, Ops>;
  using KFragment = OperandFragment<float, 1, KEY_FRAGMENTS, Ops>;

  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  const uint value_base = value_tile_idx * CHUNKED_STATE_A2_VALUE_TILE;
  const uint key_base = key_tile_idx * CHUNKED_STATE_A2_KEY_TILE;
  const uint valid_value_rows =
      value_base < head_v_dim ? min(uint(CHUNKED_STATE_A2_VALUE_TILE), head_v_dim - value_base) : 0u;
  const uint valid_key_cols = min(uint(CHUNKED_STATE_A2_KEY_TILE), uint(HEAD_K_DIM) - key_base);

  if (valid_tokens == 0 || valid_value_rows == 0) {
    return;
  }

  const float g_last = g[(token_base + valid_tokens - 1) * num_v_heads + hv_idx];
  const float alpha = fast::exp(g_last);
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;

  AccFragment acc;
  acc.clear();
  for (uint t0 = 0; t0 < CHUNK_SIZE; t0 += Ops::FRAGMENT_ROWS) {
    const uint valid_t = t0 < valid_tokens ? min(uint(Ops::FRAGMENT_ROWS), valid_tokens - t0) : 0u;
    VFragment v_frag;
    KFragment k_frag;
    const device float* v_tile =
        v_new + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + t0) * head_v_dim + value_base;
    v_frag.load_from(lane, fragment_source(v_tile, 1, int(head_v_dim)).bounded(valid_value_rows, valid_t));
    const device float* k_tile = k_norm + (token_base + t0) * key_dim + hk_idx * HEAD_K_DIM + key_base;
    k_frag.load_from(lane, fragment_source(k_tile, int(key_dim)).bounded(valid_t, valid_key_cols));
    k_frag.map_coords(lane, [&](short row, short col, float value) {
      if (uint(row) >= valid_t || uint(col) >= valid_key_cols) {
        return 0.0f;
      }
      const uint local_t = t0 + uint(row);
      return value * decay_scale[(chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t];
    });
    fragment_mma(acc, v_frag, k_frag);
  }

  device float* state_tile = state + (hv_idx * head_v_dim + value_base) * HEAD_K_DIM + key_base;
  acc.map_coords(lane, [&](short row, short col, float value) {
    if (uint(row) >= valid_value_rows || uint(col) >= valid_key_cols) {
      return 0.0f;
    }
    return alpha * state_tile[uint(row) * HEAD_K_DIM + uint(col)] + value;
  });
  acc.store_safe(lane, state_tile, int(HEAD_K_DIM), short2(valid_key_cols, valid_value_rows));
}

template <typename T, typename H, uint HEAD_K_DIM, uint CHUNK_SIZE, bool RECOMPUTE_G>
VARIANTS(T, float, half, bfloat)
VARIANTS(H, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(RECOMPUTE_G, false, true)
PUBLIC KERNEL(DeltaNetChunkedOutputA)(
    device const float* q_norm,
    device const float* qk,
    device const float* g_or_log_decay,
    device const H* h,
    device const float* v_new,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    constant const uint& num_dv_groups,
    const ThreadContext thread_context,
    const uint chunk_row_tile_group_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE) * CHUNK_SIZE.div_ceil(32)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint v_tile_idx GROUPS(head_v_dim.div_ceil(32)),
    const uint tid THREADS(CHUNKED_APPLY_THREADS)
) {
  using Ops = MxuFragmentOps<>;
  constexpr ushort ROWS = Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = CHUNKED_OUTPUT_VALUE_COLS / Ops::FRAGMENT_COLS;
  constexpr uint ROWS_PER_TG = (CHUNKED_APPLY_THREADS / METAL_SIMD_SIZE) * ROWS;
  using AccFragment = Fragment<float, 1, COL_FRAGMENTS, Ops>;
  using QFragment = OperandFragment<float, 1, 1, Ops>;
  using HFragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops>;
  using QkFragment = OperandFragment<float, 1, 1, Ops>;
  using VFragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops>;

  const uint row_tile_groups_per_chunk = (CHUNK_SIZE + ROWS_PER_TG - 1) / ROWS_PER_TG;
  const uint chunk_idx = chunk_row_tile_group_idx / row_tile_groups_per_chunk;
  const uint row_tile_in_chunk = chunk_row_tile_group_idx - chunk_idx * row_tile_groups_per_chunk;
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  const uint row_base = row_tile_in_chunk * ROWS_PER_TG + thread_context.simdgroup_index * ROWS;
  const uint value_base = v_tile_idx * CHUNKED_OUTPUT_VALUE_COLS;
  const short valid_rows = short(min(uint(ROWS), valid_tokens - min(row_base, valid_tokens)));
  const short valid_cols = short(min(uint(CHUNKED_OUTPUT_VALUE_COLS), head_v_dim - min(value_base, head_v_dim)));

  if (valid_rows == 0 || valid_cols == 0) {
    return;
  }

  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint q_base = (token_base + row_base) * key_dim + hk_idx * HEAD_K_DIM;
  const uint h_base = ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + value_base) * HEAD_K_DIM;
  const uint qk_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE;
  const uint v_base = (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * head_v_dim + value_base;
  const uint out_base = (token_base + row_base) * value_dim + hv_idx * head_v_dim + value_base;

  AccFragment acc;
  acc.clear();

  for (uint k0 = 0; k0 < HEAD_K_DIM; k0 += ROWS) {
    const short valid_k = short(min(uint(ROWS), uint(HEAD_K_DIM) - k0));
    QFragment q_frag;
    HFragment h_frag;
    q_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(q_norm + q_base + k0, int(key_dim)).bounded(valid_rows, valid_k)
    );
    h_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(h + h_base + k0, 1, int(HEAD_K_DIM)).bounded(valid_k, valid_cols)
    );
    fragment_mma(acc, q_frag, h_frag);
  }

  acc.map_coords(thread_context.simd_lane_id, [&](short row, short, float value) {
    if (uint(row) >= uint(valid_rows)) {
      return 0.0f;
    }
    const uint local_row = row_base + uint(row);
    return value * fast::exp(chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_row, num_v_heads, hv_idx));
  });

  for (uint j0 = 0; j0 < CHUNK_SIZE; j0 += ROWS) {
    const short valid_j = short(min(uint(ROWS), valid_tokens - min(j0, valid_tokens)));
    QkFragment qk_frag;
    VFragment v_frag;
    qk_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(qk + qk_base + j0, int(CHUNK_SIZE)).bounded(valid_rows, valid_j)
    );
    qk_frag.map_coords(thread_context.simd_lane_id, [&](short row, short col, float value) {
      if (uint(row) >= uint(valid_rows) || uint(col) >= uint(valid_j)) {
        return 0.0f;
      }
      const uint local_row = row_base + uint(row);
      const uint local_col = j0 + uint(col);
      if (local_col > local_row) {
        return 0.0f;
      }
      const float g_row = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_row, num_v_heads, hv_idx);
      const float g_col = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_col, num_v_heads, hv_idx);
      return value * fast::exp(g_row - g_col);
    });
    v_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(v_new + v_base + j0 * head_v_dim, int(head_v_dim)).bounded(valid_j, valid_cols)
    );
    fragment_mma(acc, qk_frag, v_frag);
  }

  acc.store_safe(thread_context.simd_lane_id, out + out_base, int(value_dim), short2(valid_cols, valid_rows));
}

template <typename T, typename H, uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(T, float, half, bfloat)
VARIANTS(H, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
PUBLIC KERNEL(DeltaNetChunkedOutputAScaledQk)(
    device const float* q_norm,
    device const float* qk_scaled,
    device const float* g,
    device const H* h,
    device const float* v_new,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    constant const uint& num_dv_groups,
    const ThreadContext thread_context,
    const uint chunk_row_tile_group_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE) * CHUNK_SIZE.div_ceil(32)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint v_tile_idx GROUPS(head_v_dim.div_ceil(32)),
    const uint tid THREADS(CHUNKED_APPLY_THREADS)
) {
  using Ops = MxuFragmentOps<>;
  constexpr ushort ROWS = Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = CHUNKED_OUTPUT_VALUE_COLS / Ops::FRAGMENT_COLS;
  constexpr uint ROWS_PER_TG = (CHUNKED_APPLY_THREADS / METAL_SIMD_SIZE) * ROWS;
  using AccFragment = Fragment<float, 1, COL_FRAGMENTS, Ops>;
  using QFragment = OperandFragment<float, 1, 1, Ops>;
  using HFragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops>;
  using QkFragment = OperandFragment<float, 1, 1, Ops>;
  using VFragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops>;

  const uint row_tile_groups_per_chunk = (CHUNK_SIZE + ROWS_PER_TG - 1) / ROWS_PER_TG;
  const uint chunk_idx = chunk_row_tile_group_idx / row_tile_groups_per_chunk;
  const uint row_tile_in_chunk = chunk_row_tile_group_idx - chunk_idx * row_tile_groups_per_chunk;
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  const uint row_base = row_tile_in_chunk * ROWS_PER_TG + thread_context.simdgroup_index * ROWS;
  const uint value_base = v_tile_idx * CHUNKED_OUTPUT_VALUE_COLS;
  const short valid_rows = short(min(uint(ROWS), valid_tokens - min(row_base, valid_tokens)));
  const short valid_cols = short(min(uint(CHUNKED_OUTPUT_VALUE_COLS), head_v_dim - min(value_base, head_v_dim)));

  if (valid_rows == 0 || valid_cols == 0) {
    return;
  }

  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint q_base = (token_base + row_base) * key_dim + hk_idx * HEAD_K_DIM;
  const uint h_base = ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + value_base) * HEAD_K_DIM;
  const uint qk_base = (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE;
  const uint v_base = (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * head_v_dim + value_base;
  const uint out_base = (token_base + row_base) * value_dim + hv_idx * head_v_dim + value_base;

  AccFragment acc;
  acc.clear();

  for (uint k0 = 0; k0 < HEAD_K_DIM; k0 += ROWS) {
    const short valid_k = short(min(uint(ROWS), uint(HEAD_K_DIM) - k0));
    QFragment q_frag;
    HFragment h_frag;
    q_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(q_norm + q_base + k0, int(key_dim)).bounded(valid_rows, valid_k)
    );
    h_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(h + h_base + k0, 1, int(HEAD_K_DIM)).bounded(valid_k, valid_cols)
    );
    fragment_mma(acc, q_frag, h_frag);
  }

  acc.map_coords(thread_context.simd_lane_id, [&](short row, short, float value) {
    if (uint(row) >= uint(valid_rows)) {
      return 0.0f;
    }
    const uint local_row = row_base + uint(row);
    return value * fast::exp(g[(token_base + local_row) * num_v_heads + hv_idx]);
  });

  for (uint j0 = 0; j0 < CHUNK_SIZE; j0 += ROWS) {
    const short valid_j = short(min(uint(ROWS), valid_tokens - min(j0, valid_tokens)));
    QkFragment qk_frag;
    VFragment v_frag;
    qk_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(qk_scaled + qk_base + j0, int(CHUNK_SIZE)).bounded(valid_rows, valid_j)
    );
    qk_frag.map_coords(thread_context.simd_lane_id, [&](short row, short col, float value) {
      if (uint(row) >= uint(valid_rows) || uint(col) >= uint(valid_j)) {
        return 0.0f;
      }
      return value;
    });
    v_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(v_new + v_base + j0 * head_v_dim, int(head_v_dim)).bounded(valid_j, valid_cols)
    );
    fragment_mma(acc, qk_frag, v_frag);
  }

  acc.store_safe(thread_context.simd_lane_id, out + out_base, int(value_dim), short2(valid_cols, valid_rows));
}

template <uint HEAD_K_DIM, uint CHUNK_SIZE, bool RECOMPUTE_G>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(RECOMPUTE_G, false, true)
PUBLIC KERNEL(DeltaNetChunkedStateC)(
    device const float* k_norm,
    device const float* w,
    device const float* u,
    device const float* g_or_log_decay,
    device float* state,
    device float* h,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    constant const uint& num_dv_groups,
    const uint hv_idx GROUPS(num_v_heads),
    const uint dv_group GROUPS(num_dv_groups),
    const uint tid THREADS(CHUNKED_APPLY_THREADS)
) {
  static_assert(HEAD_K_DIM % METAL_SIMD_SIZE == 0, "HEAD_K_DIM must be a multiple of simd size");
  constexpr uint ELEMS = HEAD_K_DIM / METAL_SIMD_SIZE;
  constexpr uint NUM_SG = CHUNKED_APPLY_THREADS / METAL_SIMD_SIZE;
  static_assert(ELEMS == 4, "chunked apply uses float4 K lanes");

  const uint lane = tid % METAL_SIMD_SIZE;
  const uint dv_local = tid / METAL_SIMD_SIZE;
  const uint dv_idx = (dv_group * NUM_SG + dv_local) * CHUNKED_APPLY_DV_PER_SIMDGROUP;
  const uint dk_base = lane * ELEMS;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint num_chunks = (suffix_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

  float4 s[CHUNKED_APPLY_DV_PER_SIMDGROUP];
  for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
    if (dv_idx + r < head_v_dim) {
      const device float* state_row = state + (hv_idx * head_v_dim + dv_idx + r) * HEAD_K_DIM + dk_base;
      s[r] = *reinterpret_cast<const device float4*>(state_row);
    } else {
      s[r] = 0.0f;
    }
  }

  for (uint chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const uint token_base = chunk_idx * CHUNK_SIZE;
    const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;

    float4 h_rows[CHUNKED_APPLY_DV_PER_SIMDGROUP];
    for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
      h_rows[r] = s[r];
      if (dv_idx + r < head_v_dim) {
        device float* h_row = h + ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + dv_idx + r) * HEAD_K_DIM + dk_base;
        *reinterpret_cast<device float4*>(h_row) = h_rows[r];
      }
    }

    if (valid_tokens > 0) {
      const uint last_token = token_base + valid_tokens - 1;
      const float g_last = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, valid_tokens - 1, num_v_heads, hv_idx);
      const float g_last_exp = fast::exp(g_last);
      for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
        s[r] *= g_last_exp;
      }

      for (uint local_t = 0; local_t < valid_tokens; ++local_t) {
        const uint token = token_base + local_t;
        const device float* w_row =
            w + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t) * HEAD_K_DIM + dk_base;
        const float4 w_vec = *reinterpret_cast<const device float4*>(w_row);
        float4 correction =
            float4(dot(w_vec, h_rows[0]), dot(w_vec, h_rows[1]), dot(w_vec, h_rows[2]), dot(w_vec, h_rows[3]));
        correction = simd_sum(correction);

        const float decay_scale =
            fast::exp(g_last - chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_t, num_v_heads, hv_idx));
        const device float* k_row = k_norm + token * key_dim + hk_idx * HEAD_K_DIM + dk_base;
        const float4 k_vec = *reinterpret_cast<const device float4*>(k_row);
        const device float* u_row =
            u + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t) * head_v_dim + dv_idx;
        for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
          if (dv_idx + r < head_v_dim) {
            s[r] += k_vec * ((u_row[r] - correction[r]) * decay_scale);
          }
        }
      }
    }
  }

  for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
    if (dv_idx + r < head_v_dim) {
      device float* state_row = state + (hv_idx * head_v_dim + dv_idx + r) * HEAD_K_DIM + dk_base;
      *reinterpret_cast<device float4*>(state_row) = s[r];
    }
  }
}

template <typename T, uint HEAD_K_DIM, uint CHUNK_SIZE, bool RECOMPUTE_G>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(RECOMPUTE_G, false, true)
PUBLIC KERNEL(DeltaNetChunkedOutputC)(
    device const float* q_norm,
    device const float* qk,
    device const float* g_or_log_decay,
    device const float* h,
    device const float* w,
    device const float* u,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    constant const uint& num_dv_groups,
    const ThreadContext thread_context,
    const uint chunk_row_tile_group_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE) * CHUNK_SIZE.div_ceil(32)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint v_tile_idx GROUPS(head_v_dim.div_ceil(32)),
    const uint tid THREADS(CHUNKED_APPLY_THREADS)
) {
  using Ops = SimdgroupFragmentOps;
  constexpr ushort ROWS = Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = CHUNKED_OUTPUT_VALUE_COLS / Ops::FRAGMENT_COLS;
  constexpr uint ROWS_PER_TG = (CHUNKED_APPLY_THREADS / METAL_SIMD_SIZE) * ROWS;
  using AccFragment = Fragment<float, 1, COL_FRAGMENTS, Ops>;
  using QFragment = OperandFragment<float, 1, 1, Ops>;
  using HFragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops>;
  using QkFragment = OperandFragment<float, 1, 1, Ops>;
  using UFragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops>;
  using WFragment = OperandFragment<float, 1, 1, Ops>;

  const uint row_tile_groups_per_chunk = (CHUNK_SIZE + ROWS_PER_TG - 1) / ROWS_PER_TG;
  const uint chunk_idx = chunk_row_tile_group_idx / row_tile_groups_per_chunk;
  const uint row_tile_in_chunk = chunk_row_tile_group_idx - chunk_idx * row_tile_groups_per_chunk;
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  const uint row_base = row_tile_in_chunk * ROWS_PER_TG + thread_context.simdgroup_index * ROWS;
  const uint value_base = v_tile_idx * CHUNKED_OUTPUT_VALUE_COLS;
  const short valid_rows = short(min(uint(ROWS), valid_tokens - min(row_base, valid_tokens)));
  const short valid_cols = short(min(uint(CHUNKED_OUTPUT_VALUE_COLS), head_v_dim - min(value_base, head_v_dim)));

  if (valid_rows == 0 || valid_cols == 0) {
    return;
  }

  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint q_base = (token_base + row_base) * key_dim + hk_idx * HEAD_K_DIM;
  const uint h_base = ((chunk_idx * num_v_heads + hv_idx) * head_v_dim + value_base) * HEAD_K_DIM;
  const uint qk_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE + row_base * CHUNK_SIZE;
  const uint u_base = (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * head_v_dim + value_base;
  const uint w_base = (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * HEAD_K_DIM;
  const uint out_base = (token_base + row_base) * value_dim + hv_idx * head_v_dim + value_base;

  AccFragment acc;
  acc.clear();

  for (uint k0 = 0; k0 < HEAD_K_DIM; k0 += ROWS) {
    const short valid_k = short(min(uint(ROWS), uint(HEAD_K_DIM) - k0));
    QFragment q_frag;
    HFragment h_frag;
    q_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(q_norm + q_base + k0, int(key_dim)).bounded(valid_rows, valid_k)
    );
    h_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(h + h_base + k0, 1, int(HEAD_K_DIM)).bounded(valid_k, valid_cols)
    );
    fragment_mma(acc, q_frag, h_frag);
  }

  acc.map_coords(thread_context.simd_lane_id, [&](short row, short, float value) {
    if (uint(row) >= uint(valid_rows)) {
      return 0.0f;
    }
    const uint local_row = row_base + uint(row);
    return value * fast::exp(chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_row, num_v_heads, hv_idx));
  });

  for (uint j0 = 0; j0 < CHUNK_SIZE; j0 += ROWS) {
    const short valid_j = short(min(uint(ROWS), valid_tokens - min(j0, valid_tokens)));
    QkFragment qk_frag;
    qk_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(qk + qk_base + j0, int(CHUNK_SIZE)).bounded(valid_rows, valid_j)
    );
    qk_frag.map_coords(thread_context.simd_lane_id, [&](short row, short col, float value) {
      if (uint(row) >= uint(valid_rows) || uint(col) >= uint(valid_j)) {
        return 0.0f;
      }
      const uint local_row = row_base + uint(row);
      const uint local_col = j0 + uint(col);
      if (local_col > local_row) {
        return 0.0f;
      }
      const float g_row = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_row, num_v_heads, hv_idx);
      const float g_col = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_col, num_v_heads, hv_idx);
      return value * fast::exp(g_row - g_col);
    });

    UFragment u_frag;
    u_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(u + u_base + j0 * head_v_dim, int(head_v_dim)).bounded(valid_j, valid_cols)
    );
    fragment_mma(acc, qk_frag, u_frag);

    AccFragment correction;
    correction.clear();
    for (uint k0 = 0; k0 < HEAD_K_DIM; k0 += ROWS) {
      const short valid_k = short(min(uint(ROWS), uint(HEAD_K_DIM) - k0));
      WFragment w_frag;
      HFragment h_frag;
      w_frag.load_from(
          thread_context.simd_lane_id,
          fragment_source(w + w_base + j0 * HEAD_K_DIM + k0, int(HEAD_K_DIM)).bounded(valid_j, valid_k)
      );
      h_frag.load_from(
          thread_context.simd_lane_id,
          fragment_source(h + h_base + k0, 1, int(HEAD_K_DIM)).bounded(valid_k, valid_cols)
      );
      fragment_mma(correction, w_frag, h_frag);
    }
    correction.map([](float value) { return -value; });
    fragment_mma(acc, qk_frag, correction);
  }

  acc.store_safe(thread_context.simd_lane_id, out + out_base, int(value_dim), short2(valid_cols, valid_rows));
}

template <typename T, uint HEAD_K_DIM, uint CHUNK_SIZE, bool RECOMPUTE_G>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(RECOMPUTE_G, false, true)
PUBLIC KERNEL(DeltaNetChunkedApplyB)(
    device const float* q_norm,
    device const float* k_norm,
    device const float* qk,
    device const float* g_or_log_decay,
    device const float* w,
    device const float* u,
    device float* state,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    constant const uint& num_dv_groups,
    threadgroup float
        v_scratch[(CHUNKED_APPLY_THREADS / METAL_SIMD_SIZE) * CHUNK_SIZE * CHUNKED_APPLY_DV_PER_SIMDGROUP],
    const uint hv_idx GROUPS(num_v_heads),
    const uint dv_group GROUPS(num_dv_groups),
    const uint tid THREADS(CHUNKED_APPLY_THREADS)
) {
  static_assert(HEAD_K_DIM % METAL_SIMD_SIZE == 0, "HEAD_K_DIM must be a multiple of simd size");
  constexpr uint ELEMS = HEAD_K_DIM / METAL_SIMD_SIZE;
  constexpr uint NUM_SG = CHUNKED_APPLY_THREADS / METAL_SIMD_SIZE;
  static_assert(ELEMS == 4, "chunked apply uses float4 K lanes");

  const uint lane = tid % METAL_SIMD_SIZE;
  const uint dv_local = tid / METAL_SIMD_SIZE;
  const uint dv_idx = (dv_group * NUM_SG + dv_local) * CHUNKED_APPLY_DV_PER_SIMDGROUP;
  const uint dk_base = lane * ELEMS;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint num_chunks = (suffix_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
  const uint scratch_base = dv_local * CHUNK_SIZE * CHUNKED_APPLY_DV_PER_SIMDGROUP;

  float4 s[CHUNKED_APPLY_DV_PER_SIMDGROUP];
  for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
    if (dv_idx + r < head_v_dim) {
      const device float* state_row = state + (hv_idx * head_v_dim + dv_idx + r) * HEAD_K_DIM + dk_base;
      s[r] = *reinterpret_cast<const device float4*>(state_row);
    } else {
      s[r] = 0.0f;
    }
  }

  for (uint chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const uint token_base = chunk_idx * CHUNK_SIZE;
    const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;

    float4 h_rows[CHUNKED_APPLY_DV_PER_SIMDGROUP];
    for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
      h_rows[r] = s[r];
    }

    for (uint local_t = 0; local_t < CHUNK_SIZE; ++local_t) {
      float4 v_value = 0.0f;
      if (local_t < valid_tokens) {
        const device float* w_row =
            w + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t) * HEAD_K_DIM + dk_base;
        const float4 w_vec = *reinterpret_cast<const device float4*>(w_row);
        float4 correction =
            float4(dot(w_vec, h_rows[0]), dot(w_vec, h_rows[1]), dot(w_vec, h_rows[2]), dot(w_vec, h_rows[3]));
        correction = simd_sum(correction);

        const device float* u_row =
            u + ((chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE + local_t) * head_v_dim + dv_idx;
        for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
          if (dv_idx + r < head_v_dim) {
            v_value[r] = u_row[r] - correction[r];
          }
        }
      }

      if (lane == 0) {
        threadgroup float* scratch = v_scratch + scratch_base + local_t * CHUNKED_APPLY_DV_PER_SIMDGROUP;
        scratch[0] = v_value[0];
        scratch[1] = v_value[1];
        scratch[2] = v_value[2];
        scratch[3] = v_value[3];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local_t = 0; local_t < valid_tokens; ++local_t) {
      const uint token = token_base + local_t;
      const float g_row = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_t, num_v_heads, hv_idx);
      const device float* q_row = q_norm + token * key_dim + hk_idx * HEAD_K_DIM + dk_base;
      const float4 q_vec = *reinterpret_cast<const device float4*>(q_row);

      float4 value = float4(dot(q_vec, h_rows[0]), dot(q_vec, h_rows[1]), dot(q_vec, h_rows[2]), dot(q_vec, h_rows[3]));
      value = simd_sum(value) * fast::exp(g_row);

      const device float* qk_row =
          qk + (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE + local_t * CHUNK_SIZE;
      for (uint local_j = 0; local_j <= local_t; ++local_j) {
        const float scale =
            qk_row[local_j] *
            fast::exp(g_row - chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_j, num_v_heads, hv_idx));
        threadgroup float* scratch = v_scratch + scratch_base + local_j * CHUNKED_APPLY_DV_PER_SIMDGROUP;
        value += scale * float4(scratch[0], scratch[1], scratch[2], scratch[3]);
      }

      if (lane == 0) {
        device T* out_row = out + token * value_dim + hv_idx * head_v_dim + dv_idx;
        for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
          if (dv_idx + r < head_v_dim) {
            out_row[r] = static_cast<T>(value[r]);
          }
        }
      }
    }

    if (valid_tokens > 0) {
      const uint last_token = token_base + valid_tokens - 1;
      const float g_last = chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, valid_tokens - 1, num_v_heads, hv_idx);
      const float g_last_exp = fast::exp(g_last);
      for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
        s[r] = h_rows[r] * g_last_exp;
      }

      for (uint local_t = 0; local_t < valid_tokens; ++local_t) {
        const uint token = token_base + local_t;
        const float decay_scale =
            fast::exp(g_last - chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, local_t, num_v_heads, hv_idx));
        const device float* k_row = k_norm + token * key_dim + hk_idx * HEAD_K_DIM + dk_base;
        const float4 k_vec = *reinterpret_cast<const device float4*>(k_row);
        threadgroup float* scratch = v_scratch + scratch_base + local_t * CHUNKED_APPLY_DV_PER_SIMDGROUP;
        s[0] += k_vec * (scratch[0] * decay_scale);
        s[1] += k_vec * (scratch[1] * decay_scale);
        s[2] += k_vec * (scratch[2] * decay_scale);
        s[3] += k_vec * (scratch[3] * decay_scale);
      }
    }
  }

  for (uint r = 0; r < CHUNKED_APPLY_DV_PER_SIMDGROUP; ++r) {
    if (dv_idx + r < head_v_dim) {
      device float* state_row = state + (hv_idx * head_v_dim + dv_idx + r) * HEAD_K_DIM + dk_base;
      *reinterpret_cast<device float4*>(state_row) = s[r];
    }
  }
}
