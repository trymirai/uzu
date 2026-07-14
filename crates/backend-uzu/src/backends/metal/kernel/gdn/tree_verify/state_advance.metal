#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"

using namespace metal;

// Advances the committed GDN state along one accepted proposal-tree path.
//
// Shapes (T = proposal tree size, L = accepted path length):
//   k_norm          [T, num_k_heads, 128] f32
//   v               [T, num_v_heads, 128] T
//   log_decay, beta [T, num_v_heads] f32
//   accepted_indices [L] u32
//   state           [num_v_heads, 128, 128] f32 input/output
//
// Grid: eight 128-thread groups per value head. Each group's four simdgroups
// own four adjacent value rows each; their lanes cover key dimension in
// float4 slices. State stays in registers while the kernel follows the
// accepted indices, then is written once. K/V/G/Beta are read indirectly;
// no accepted-path gather is required.
#define STATE_ADVANCE_THREADS 128
#define DV_PER_SIMDGROUP 4
#define STATE_ADVANCE_DV_GROUPS 8

template <typename T, uint HEAD_K_DIM>
VARIANTS(T, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
PUBLIC KERNEL(StateAdvance)(
    device const float* k_norm,
    device const T* v,
    device const float* log_decay_buf,
    device const float* beta_buf,
    device const uint* accepted_indices,
    device float* state,
    constant const uint& accepted_len,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint dv_group_idx GROUPS(STATE_ADVANCE_DV_GROUPS),
    const uint thread_idx THREADS(STATE_ADVANCE_THREADS)
) {
  constexpr uint ELEMS = HEAD_K_DIM / METAL_SIMD_SIZE;
  constexpr uint NUM_SG = STATE_ADVANCE_THREADS / METAL_SIMD_SIZE;
  constexpr uint HEAD_V_DIM = HEAD_K_DIM;
  static_assert(ELEMS == 4, "float4 state advance requires ELEMS == 4");

  const uint lane = thread_context.simd_lane_id;
  const uint simdgroup_idx = thread_context.simdgroup_index;
  const uint dv_base = (dv_group_idx * NUM_SG + simdgroup_idx) * DV_PER_SIMDGROUP;
  const uint dk_base = lane * ELEMS;
  const uint hk_idx = hv_idx / (num_v_heads / num_k_heads);
  const uint key_dim = num_k_heads * HEAD_K_DIM;
  const uint value_dim = num_v_heads * HEAD_V_DIM;

  device float* state_tile = state + (hv_idx * HEAD_V_DIM + dv_base) * HEAD_K_DIM + dk_base;
  float4 state_rows[DV_PER_SIMDGROUP];
  state_rows[0] = *reinterpret_cast<device const float4*>(state_tile + 0 * HEAD_K_DIM);
  state_rows[1] = *reinterpret_cast<device const float4*>(state_tile + 1 * HEAD_K_DIM);
  state_rows[2] = *reinterpret_cast<device const float4*>(state_tile + 2 * HEAD_K_DIM);
  state_rows[3] = *reinterpret_cast<device const float4*>(state_tile + 3 * HEAD_K_DIM);

  for (uint accepted_idx = 0; accepted_idx < accepted_len; ++accepted_idx) {
    const uint tree_idx = accepted_indices[accepted_idx];
    const uint tree_head_offset = tree_idx * num_v_heads + hv_idx;
    const float decay = fast::exp(log_decay_buf[tree_head_offset]);
    const float beta = beta_buf[tree_head_offset];
    const uint k_offset = tree_idx * key_dim + hk_idx * HEAD_K_DIM + dk_base;
    const float4 k = *reinterpret_cast<device const float4*>(k_norm + k_offset);

    state_rows[0] *= decay;
    state_rows[1] *= decay;
    state_rows[2] *= decay;
    state_rows[3] *= decay;
    float4 kv_mem = float4(dot(state_rows[0], k), dot(state_rows[1], k), dot(state_rows[2], k), dot(state_rows[3], k));
    kv_mem = simd_sum(kv_mem);

    const device T* v_row = v + tree_idx * value_dim + hv_idx * HEAD_V_DIM + dv_base;
    const float4 v_value = float4(float(v_row[0]), float(v_row[1]), float(v_row[2]), float(v_row[3]));
    const float4 delta = beta * (v_value - kv_mem);
    state_rows[0] += k * delta[0];
    state_rows[1] += k * delta[1];
    state_rows[2] += k * delta[2];
    state_rows[3] += k * delta[3];
  }

  *reinterpret_cast<device float4*>(state_tile + 0 * HEAD_K_DIM) = state_rows[0];
  *reinterpret_cast<device float4*>(state_tile + 1 * HEAD_K_DIM) = state_rows[1];
  *reinterpret_cast<device float4*>(state_tile + 2 * HEAD_K_DIM) = state_rows[2];
  *reinterpret_cast<device float4*>(state_tile + 3 * HEAD_K_DIM) = state_rows[3];
}
