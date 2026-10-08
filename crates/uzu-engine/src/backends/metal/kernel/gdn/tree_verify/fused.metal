#include <metal_stdlib>
#include "../../activation/activations.h"
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../generated/trie.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::trie;

#define HEAD_DIM 128u
#define TREE_CAP 32u
#define NUM_SIMDGROUPS 4u
#define FUSED_THREADS (NUM_SIMDGROUPS * METAL_SIMD_SIZE)

// The whole GDN tree verify of one value head in one threadgroup, for trees of up to MAX_TREE nodes
// (DFS order, so every ancestor precedes its descendants): path decay prefixes, the ancestor gram
// matrices, the h0 products as simdgroup matmuls ([MAX_TREE x 128] x [128 x 32] per simdgroup, the
// state streamed once), the (I + A) U = rhs forward substitution with each lane owning one state row
// dv, the output and the gated RMS norm. All per-lane loops have compile-time bounds so the node
// arrays stay in registers; A and qkd are zero above the diagonal, which makes the fixed-bound
// substitution exact.
//
// q, k:            [tree, k_heads, 128] normalized (q pre-scaled by 1/sqrt(128))
// v:               [tree, value_heads, 128]
// trie:            [tree]; col is an ancestor-or-self of row iff start[col] <= row <= end[col]
// log_decay, beta: [tree, value_heads]
// h0:              [value_heads, 128 (dv), 128 (dk)] fp32 layer state, read only
// in_proj:         [tree, total_proj_dim]; the gate z of head hv sits at conv_dim + hv * 128
// out:             [tree, value_heads, 128] = rmsnorm(o) * norm_weight * silu(z)
template <typename T, uint MAX_TREE>
VARIANTS(T, bfloat)
VARIANTS(MAX_TREE, 16, 32)
PUBLIC KERNEL(TreeVerifyFused)(
    const device T* q,
    const device T* k,
    const device T* v,
    const device TrieNode* trie,
    const device float* log_decay,
    const device float* beta,
    const device float* h0,
    const device T* in_proj,
    const device float* norm_weight,
    device T* out,
    constant const uint& tree_size,
    constant const uint& k_heads,
    constant const uint& value_heads,
    constant const uint& conv_dim,
    constant const uint& total_proj_dim,
    constant const float& norm_epsilon,
    threadgroup float products_s[NUM_SIMDGROUPS * TREE_CAP * METAL_SIMD_SIZE],
    threadgroup float a_s[TREE_CAP * TREE_CAP],
    threadgroup float qkd_s[TREE_CAP * TREE_CAP],
    threadgroup float log_decay_s[TREE_CAP],
    threadgroup float prefix_s[TREE_CAP],
    threadgroup float decay_s[TREE_CAP],
    threadgroup float beta_s[TREE_CAP],
    threadgroup uint start_s[TREE_CAP],
    threadgroup uint end_s[TREE_CAP],
    threadgroup float sumsq_s[NUM_SIMDGROUPS * TREE_CAP],
    const ThreadContext thread_context,
    const uint hv GROUPS(value_heads),
    const uint tid THREADS(FUSED_THREADS)
) {
  using Ops = SimdgroupFragmentOps;
  constexpr ushort ROW_FRAGMENTS = MAX_TREE / Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = METAL_SIMD_SIZE / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, ROW_FRAGMENTS, COL_FRAGMENTS, Ops>;
  using LeftFragment = OperandFragment<float, ROW_FRAGMENTS, 1, Ops>;
  using RightFragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops, ReadTranspose>;

  const ushort lane = thread_context.simd_lane_id;
  const uint simdgroup = thread_context.simdgroup_index;
  const uint dv_base = simdgroup * METAL_SIMD_SIZE;
  const uint dv = dv_base + lane;
  const uint hk = hv / (value_heads / k_heads);
  const uint qk_stride = k_heads * HEAD_DIM;
  const device T* k_rows = k + hk * HEAD_DIM;
  const device T* q_rows = q + hk * HEAD_DIM;
  threadgroup float* products = products_s + simdgroup * TREE_CAP * METAL_SIMD_SIZE;

  // Node scalars and trie intervals (empty interval beyond the tree).
  if (tid < TREE_CAP) {
    const bool valid = tid < tree_size;
    uint start = 1;
    uint end = 0;
    if (valid) {
      const TrieNode node = trie[tid];
      start = node.trie_start;
      end = node.trie_end;
    }
    start_s[tid] = start;
    end_s[tid] = end;
    log_decay_s[tid] = valid ? log_decay[tid * value_heads + hv] : 0.0f;
    beta_s[tid] = valid ? beta[tid * value_heads + hv] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Path decay prefix: the log decays of the ancestors-or-self.
  if (tid < TREE_CAP) {
    float prefix = 0.0f;
    for (uint col = 0; col < TREE_CAP; ++col) {
      prefix += (start_s[col] <= tid && tid <= end_s[col]) ? log_decay_s[col] : 0.0f;
    }
    prefix_s[tid] = prefix;
    decay_s[tid] = exp(prefix);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // A[r][c] = beta_r exp(p_r - p_c) k_r.k_c for proper ancestors c of r, qkd[r][c] = exp(p_r - p_c) q_r.k_c
  // for ancestors-or-self; zero elsewhere (so above the diagonal).
  for (uint entry = tid; entry < TREE_CAP * TREE_CAP; entry += FUSED_THREADS) {
    const uint row = entry / TREE_CAP;
    const uint col = entry % TREE_CAP;
    const bool related = start_s[col] <= row && row <= end_s[col];
    float kk = 0.0f;
    float qk = 0.0f;
    if (related) {
      const device T* k_col = k_rows + col * qk_stride;
      const device T* k_row = k_rows + row * qk_stride;
      const device T* q_row = q_rows + row * qk_stride;
      for (uint d = 0; d < HEAD_DIM; d += 4) {
        const float4 k_value = float4(*reinterpret_cast<const device vec<T, 4>*>(k_col + d));
        kk += dot(float4(*reinterpret_cast<const device vec<T, 4>*>(k_row + d)), k_value);
        qk += dot(float4(*reinterpret_cast<const device vec<T, 4>*>(q_row + d)), k_value);
      }
    }
    const float decay = related ? exp(prefix_s[row] - prefix_s[col]) : 0.0f;
    a_s[entry] = related && row != col ? beta_s[row] * decay * kk : 0.0f;
    qkd_s[entry] = decay * qk;
  }

  // kh = k . h0^T and qh = q . h0^T for this simdgroup's 32 state rows, the state streamed once.
  AccFragment kh_acc;
  AccFragment qh_acc;
  kh_acc.clear();
  qh_acc.clear();
  const device float* h0_tile = h0 + (hv * HEAD_DIM + dv_base) * HEAD_DIM;
  const bool full_rows = tree_size >= MAX_TREE;
  for (uint kb = 0; kb < HEAD_DIM; kb += Ops::FRAGMENT_ROWS) {
    LeftFragment k_left;
    LeftFragment q_left;
    RightFragment h0_right;
    k_left.load_maybe_bounded(lane, k_rows + kb, qk_stride, full_rows, tree_size, Ops::FRAGMENT_ROWS);
    q_left.load_maybe_bounded(lane, q_rows + kb, qk_stride, full_rows, tree_size, Ops::FRAGMENT_ROWS);
    h0_right.load_maybe_bounded(lane, h0_tile + kb, HEAD_DIM, true, METAL_SIMD_SIZE, Ops::FRAGMENT_ROWS);
    fragment_mma(kh_acc, k_left, h0_right);
    fragment_mma(qh_acc, q_left, h0_right);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // rhs = beta (v - exp(p) kh), then (I + A) u = rhs by forward substitution; lane = dv.
  AccFragment::zip_for_each_coord(
      lane,
      [&](ushort row, ushort col, thread float& value) { products[row * METAL_SIMD_SIZE + col] = value; },
      kh_acc
  );
  simdgroup_barrier(mem_flags::mem_threadgroup);
  // The loops are fully unrolled, so the triangular guards are resolved at compile time and the
  // node arrays stay in registers.
  float u[MAX_TREE];
#pragma clang loop unroll(full)
  for (uint t = 0; t < MAX_TREE; ++t) {
    u[t] = t < tree_size ? beta_s[t] * (float(v[(t * value_heads + hv) * HEAD_DIM + dv]) -
                                        decay_s[t] * products[t * METAL_SIMD_SIZE + lane])
                         : 0.0f;
  }
#pragma clang loop unroll(full)
  for (uint i = 1; i < MAX_TREE; ++i) {
    float value = u[i];
#pragma clang loop unroll(full)
    for (uint j = 0; j < MAX_TREE; ++j) {
      if (j < i) {
        value -= a_s[i * TREE_CAP + j] * u[j];
      }
    }
    u[i] = value;
  }

  // o_r = exp(p_r) q_r . h0[dv] + sum over ancestors-or-self j of qkd[r][j] u_j.
  simdgroup_barrier(mem_flags::mem_threadgroup);
  AccFragment::zip_for_each_coord(
      lane,
      [&](ushort row, ushort col, thread float& value) { products[row * METAL_SIMD_SIZE + col] = value; },
      qh_acc
  );
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float o[MAX_TREE];
#pragma clang loop unroll(full)
  for (uint r = 0; r < MAX_TREE; ++r) {
    float value = decay_s[r] * products[r * METAL_SIMD_SIZE + lane];
#pragma clang loop unroll(full)
    for (uint j = 0; j < MAX_TREE; ++j) {
      if (j <= r) {
        value += qkd_s[r * TREE_CAP + j] * u[j];
      }
    }
    // The flat path hands o to the norm gate as T; round the same way so both paths agree.
    o[r] = float(static_cast<T>(value));
  }

  // Gated RMS norm over the head's 128 values: simdgroup partial sums of squares combined per row.
#pragma clang loop unroll(full)
  for (uint r = 0; r < MAX_TREE; ++r) {
    const float partial = simd_sum(o[r] * o[r]);
    if (lane == 0) {
      sumsq_s[simdgroup * TREE_CAP + r] = partial;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
  for (uint r = 0; r < MAX_TREE; ++r) {
    if (r < tree_size) {
      float sumsq = 0.0f;
      for (uint sg = 0; sg < NUM_SIMDGROUPS; ++sg) {
        sumsq += sumsq_s[sg * TREE_CAP + r];
      }
      const float inv_rms = rsqrt(sumsq / float(HEAD_DIM) + norm_epsilon);
      const float gate = activate_silu(float(in_proj[r * total_proj_dim + conv_dim + hv * HEAD_DIM + dv]));
      out[(r * value_heads + hv) * HEAD_DIM + dv] = static_cast<T>(o[r] * inv_rms * norm_weight[dv] * gate);
    }
  }
}
