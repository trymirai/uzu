#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define BUILD_WU_BLOCK 16u

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

template <typename T, typename O, uint CHUNK_SIZE, uint BV, bool USE_MXU>
VARIANTS(T, float, half, bfloat)
VARIANTS(O, float, bfloat)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(BV, 16, 32)
VARIANTS(USE_MXU, false, true)
CONSTRAINT(!USE_MXU || T != "float")
CONSTRAINT(!USE_MXU || BV >= 32)
PUBLIC KERNEL(DeltaNetChunkedBuildU)(
    device const T* in_proj,
    device const float* beta,
    device const float* a_packed,
    device const float* a_inv,
    device O* u_out,
    constant const uint& num_v_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(BV)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr ushort TOKEN_FRAGMENTS = BUILD_WU_BLOCK / Ops::FRAGMENT_ROWS;
  constexpr ushort VALUE_FRAGMENTS = BV / Ops::FRAGMENT_COLS;
  static_assert(BV % Ops::FRAGMENT_COLS == 0, "BV must align to fragment columns");
  using TileFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using MatrixFragment = OperandFragment<float, TOKEN_FRAGMENTS, TOKEN_FRAGMENTS, Ops>;
  using ValueFragment = OperandFragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using PairMatrixFragment = OperandFragment<float, TOKEN_FRAGMENTS, 2 * TOKEN_FRAGMENTS, Ops>;
  using PairValueFragment = OperandFragment<float, 2 * TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;

  constexpr uint num_blocks = (CHUNK_SIZE + BUILD_WU_BLOCK - 1) / BUILD_WU_BLOCK;
  constexpr uint num_col_pairs = (num_blocks + 1) / 2;
  const uint value_base = value_tile_idx * BV;
  const uint tile_value_cols = min(BV, head_v_dim - value_base);
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const device float* a_blocks = a_packed + (chunk_idx * num_v_heads + hv_idx) * num_blocks * num_col_pairs *
                                                (BUILD_WU_BLOCK * 2 * BUILD_WU_BLOCK);
  const device float* inv_blocks =
      a_inv + (chunk_idx * num_v_heads + hv_idx) * num_blocks * (BUILD_WU_BLOCK * BUILD_WU_BLOCK);
  device O* u_head = u_out + (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * head_v_dim + value_base;

  for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
    const uint row_base = block_idx * BUILD_WU_BLOCK;
    const uint row_token_base = token_base + row_base;
    const uint valid_rows = row_token_base < suffix_len ? min(uint(BUILD_WU_BLOCK), suffix_len - row_token_base) : 0u;

    ValueFragment v_frag;
    const device T* v_block =
        in_proj + row_token_base * total_proj_dim + 2 * key_dim + hv_idx * head_v_dim + value_base;
    v_frag.load_from(lane, fragment_source(v_block, total_proj_dim).bounded(valid_rows, tile_value_cols));

    float lane_beta = 0.0f;
    if (lane < BUILD_WU_BLOCK && lane < valid_rows) {
      lane_beta = beta[(row_token_base + lane) * num_v_heads + hv_idx];
    }

    TileFragment acc;
    TileFragment::zip_for_each_coord(
        lane,
        [&](ushort local_row, ushort, thread float& acc_value, thread float& v_value) {
          acc_value = simd_shuffle(lane_beta, local_row) * v_value;
        },
        acc,
        v_frag
    );

    const uint num_full_pairs = block_idx / 2;
    for (uint pair_idx = 0; pair_idx < num_full_pairs; ++pair_idx) {
      PairMatrixFragment a_frag;
      PairValueFragment u_prev_frag;
      const device float* a_pair =
          a_blocks + (block_idx * num_col_pairs + pair_idx) * (BUILD_WU_BLOCK * 2 * BUILD_WU_BLOCK);
      const device O* u_prev = u_head + pair_idx * 2 * BUILD_WU_BLOCK * head_v_dim;
      a_frag.load_from(lane, fragment_source(a_pair, 2 * BUILD_WU_BLOCK));
      u_prev_frag.load_from(lane, fragment_source(u_prev, head_v_dim).bounded(2 * BUILD_WU_BLOCK, tile_value_cols));
      a_frag.map([](float value) { return -value; });
      fragment_mma(acc, a_frag, u_prev_frag);
    }

    for (uint prev_block_idx = num_full_pairs * 2; prev_block_idx < block_idx; ++prev_block_idx) {
      MatrixFragment a_frag;
      ValueFragment u_prev_frag;
      const device float* a_block =
          a_blocks + (block_idx * num_col_pairs + prev_block_idx / 2) * (BUILD_WU_BLOCK * 2 * BUILD_WU_BLOCK) +
          (prev_block_idx % 2) * BUILD_WU_BLOCK;
      const device O* u_prev = u_head + prev_block_idx * BUILD_WU_BLOCK * head_v_dim;
      a_frag.load_from(lane, fragment_source(a_block, 2 * BUILD_WU_BLOCK));
      u_prev_frag.load_from(lane, fragment_source(u_prev, head_v_dim).bounded(BUILD_WU_BLOCK, tile_value_cols));
      a_frag.map([](float value) { return -value; });
      fragment_mma(acc, a_frag, u_prev_frag);
    }

    MatrixFragment inv_frag;
    TileFragment solved;
    solved.clear();
    inv_frag.load_from(lane, fragment_source(inv_blocks + block_idx * BUILD_WU_BLOCK * BUILD_WU_BLOCK, BUILD_WU_BLOCK));
    fragment_mma(solved, inv_frag, acc);
    solved.store_safe(lane, u_head + row_base * head_v_dim, head_v_dim, short2(tile_value_cols, BUILD_WU_BLOCK));
    simdgroup_barrier(mem_flags::mem_device);
  }
}

template <typename O, uint HEAD_K_DIM, uint CHUNK_SIZE, uint BV, bool RECOMPUTE_G>
VARIANTS(O, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
VARIANTS(BV, 16, 32)
VARIANTS(RECOMPUTE_G, false, true)
PUBLIC KERNEL(DeltaNetChunkedBuildW)(
    device const float* k_norm,
    device const float* beta,
    device const float* g_or_log_decay,
    device const float* a_packed,
    device const float* a_inv,
    device O* w_out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& key_dim,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint key_tile_idx GROUPS(HEAD_K_DIM.div_ceil(BV)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = SimdgroupFragmentOps;
  constexpr ushort TOKEN_FRAGMENTS = BUILD_WU_BLOCK / Ops::FRAGMENT_ROWS;
  constexpr ushort VALUE_FRAGMENTS = BV / Ops::FRAGMENT_COLS;
  using TileFragment = Fragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using MatrixFragment = OperandFragment<float, TOKEN_FRAGMENTS, TOKEN_FRAGMENTS, Ops>;
  using ValueFragment = OperandFragment<float, TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;
  using PairMatrixFragment = OperandFragment<float, TOKEN_FRAGMENTS, 2 * TOKEN_FRAGMENTS, Ops>;
  using PairValueFragment = OperandFragment<float, 2 * TOKEN_FRAGMENTS, VALUE_FRAGMENTS, Ops>;

  constexpr uint num_blocks = (CHUNK_SIZE + BUILD_WU_BLOCK - 1) / BUILD_WU_BLOCK;
  constexpr uint num_col_pairs = (num_blocks + 1) / 2;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint key_base = key_tile_idx * BV;
  const uint tile_key_cols = min(BV, HEAD_K_DIM - key_base);
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const device float* a_blocks = a_packed + (chunk_idx * num_v_heads + hv_idx) * num_blocks * num_col_pairs *
                                                (BUILD_WU_BLOCK * 2 * BUILD_WU_BLOCK);
  const device float* inv_blocks =
      a_inv + (chunk_idx * num_v_heads + hv_idx) * num_blocks * (BUILD_WU_BLOCK * BUILD_WU_BLOCK);
  device O* w_head = w_out + (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * HEAD_K_DIM + key_base;

  for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
    const uint row_base = block_idx * BUILD_WU_BLOCK;
    const uint row_token_base = token_base + row_base;
    const uint valid_rows = row_token_base < suffix_len ? min(uint(BUILD_WU_BLOCK), suffix_len - row_token_base) : 0u;

    ValueFragment k_frag;
    const device float* k_block = k_norm + row_token_base * key_dim + hk_idx * HEAD_K_DIM + key_base;
    k_frag.load_from(lane, fragment_source(k_block, key_dim).bounded(valid_rows, tile_key_cols));

    float lane_scale = 0.0f;
    if (lane < BUILD_WU_BLOCK && lane < valid_rows) {
      const uint token = row_token_base + lane;
      lane_scale = beta[token * num_v_heads + hv_idx] *
                   fast::exp(chunked_g<RECOMPUTE_G>(g_or_log_decay, token_base, row_base + lane, num_v_heads, hv_idx));
    }

    TileFragment acc;
    TileFragment::zip_for_each_coord(
        lane,
        [&](ushort local_row, ushort, thread float& acc_value, thread float& k_value) {
          acc_value = simd_shuffle(lane_scale, local_row) * k_value;
        },
        acc,
        k_frag
    );

    const uint num_full_pairs = block_idx / 2;
    for (uint pair_idx = 0; pair_idx < num_full_pairs; ++pair_idx) {
      PairMatrixFragment a_frag;
      PairValueFragment w_prev_frag;
      const device float* a_pair =
          a_blocks + (block_idx * num_col_pairs + pair_idx) * (BUILD_WU_BLOCK * 2 * BUILD_WU_BLOCK);
      const device O* w_prev = w_head + pair_idx * 2 * BUILD_WU_BLOCK * HEAD_K_DIM;
      a_frag.load_from(lane, fragment_source(a_pair, 2 * BUILD_WU_BLOCK));
      w_prev_frag.load_from(lane, fragment_source(w_prev, HEAD_K_DIM).bounded(2 * BUILD_WU_BLOCK, tile_key_cols));
      a_frag.map([](float value) { return -value; });
      fragment_mma(acc, a_frag, w_prev_frag);
    }

    for (uint prev_block_idx = num_full_pairs * 2; prev_block_idx < block_idx; ++prev_block_idx) {
      MatrixFragment a_frag;
      ValueFragment w_prev_frag;
      const device float* a_block =
          a_blocks + (block_idx * num_col_pairs + prev_block_idx / 2) * (BUILD_WU_BLOCK * 2 * BUILD_WU_BLOCK) +
          (prev_block_idx % 2) * BUILD_WU_BLOCK;
      const device O* w_prev = w_head + prev_block_idx * BUILD_WU_BLOCK * HEAD_K_DIM;
      a_frag.load_from(lane, fragment_source(a_block, 2 * BUILD_WU_BLOCK));
      w_prev_frag.load_from(lane, fragment_source(w_prev, HEAD_K_DIM).bounded(BUILD_WU_BLOCK, tile_key_cols));
      a_frag.map([](float value) { return -value; });
      fragment_mma(acc, a_frag, w_prev_frag);
    }

    MatrixFragment inv_frag;
    TileFragment solved;
    solved.clear();
    inv_frag.load_from(lane, fragment_source(inv_blocks + block_idx * BUILD_WU_BLOCK * BUILD_WU_BLOCK, BUILD_WU_BLOCK));
    fragment_mma(solved, inv_frag, acc);
    solved.store_safe(lane, w_head + row_base * HEAD_K_DIM, HEAD_K_DIM, short2(tile_key_cols, BUILD_WU_BLOCK));
    simdgroup_barrier(mem_flags::mem_device);
  }
}
