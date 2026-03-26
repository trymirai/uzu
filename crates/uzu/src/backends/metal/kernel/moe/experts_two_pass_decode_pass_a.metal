#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "moe_commons.h"
using namespace metal;

// === Pass A: Vectorized GEMV with float4 loads ===
// Structure: 4 simdgroups (128 threads), each outputs 1 hidden element
// Each simdgroup: 32 threads reduce d_model using float4 vectorized loads
// Grid: (h_blocks, rows, 1) where h_blocks = ceil(d_ff/4)
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MoeExpertsDecodePassA)(
    device const T* x_perm,             // [total_rows, d_model]
    device const uint* expert_offsets,  // [E + 1]
    device const T* w13_all,            // [E, 2*d_ff, d_model]
    device float* hidden_out,           // [E, 2*d_ff]
    device const T* up_biases,          // [total_rows, d_ff]
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& e,
    constant float& gate_clip_min,
    constant float& gate_clip_max,
    constant float& up_clip_min,
    constant float& up_clip_max,
    constant float& silu_alpha,
    device const uint* tile_map,
    const uint gating_sel SPECIALIZE,
    const ThreadContext thread_context,
    const uint tile_idx GROUPS(INDIRECT),
    const uint tid THREADS(128)
) {
  // Read tile descriptor
  const uint h_block_idx = tile_map[tile_idx * 3 + 0];
  const uint expert_idx = tile_map[tile_idx * 3 + 1];
  const uint row_in_expert = tile_map[tile_idx * 3 + 2];

  // Validate row bounds
  const uint seg_start = expert_offsets[expert_idx];
  const uint seg_end = expert_offsets[expert_idx + 1];
  const uint global_row = seg_start + row_in_expert;
  if (global_row >= seg_end)
    return;

  // Each simdgroup outputs one hidden element
  const uint h_idx = h_block_idx * 4 + thread_context.threadgroup_index;
  if (h_idx >= d_ff)
    return;

  // Base addresses
  const ulong w13_stride = (ulong)d_model * (ulong)(2 * d_ff);
  const ulong w13_base = (ulong)expert_idx * w13_stride;
  const ulong bias_base = (ulong)expert_idx * (ulong)(2 * d_ff);
  const ulong x_row_base = (ulong)global_row * (ulong)d_model;

  device const T* x_ptr = x_perm + x_row_base;
  device const T* w_up_row = w13_all + w13_base + (ulong)h_idx * (ulong)d_model;
  device const T* w_gate_row =
      w13_all + w13_base + (ulong)(d_ff + h_idx) * (ulong)d_model;

  float acc_up = 0.0f;
  float acc_gate = 0.0f;

  // Vectorized reduction: 32 threads × 4 elements = 128 elements per iteration
  const uint vec_iters = d_model / 128;

  for (uint i = 0; i < vec_iters; ++i) {
    uint base_idx = i * 128 + thread_context.simdgroup_index * 4;

    device const T* x_vec = reinterpret_cast<device const T*>(x_ptr + base_idx);
    device const T* w_up_vec =
        reinterpret_cast<device const T*>(w_up_row + base_idx);
    acc_up += float(x_vec[0]) * float(w_up_vec[0]);
    acc_up += float(x_vec[1]) * float(w_up_vec[1]);
    acc_up += float(x_vec[2]) * float(w_up_vec[2]);
    acc_up += float(x_vec[3]) * float(w_up_vec[3]);

    if (gating_sel > 1) {
      device const T* w_gate_vec =
          reinterpret_cast<device const T*>(w_gate_row + base_idx);
      acc_gate += float(x_vec[0]) * float(w_gate_vec[0]);
      acc_gate += float(x_vec[1]) * float(w_gate_vec[1]);
      acc_gate += float(x_vec[2]) * float(w_gate_vec[2]);
      acc_gate += float(x_vec[3]) * float(w_gate_vec[3]);
    }
  }

  // Handle leftover elements
  uint leftover_start = vec_iters * 128 + thread_context.simdgroup_index;
  for (uint idx = leftover_start; idx < d_model; idx += 32) {
    float xv = float(x_ptr[idx]);
    acc_up += xv * float(w_up_row[idx]);
    if (gating_sel > 1) {
      acc_gate += xv * float(w_gate_row[idx]);
    }
  }

  // Simdgroup reduction
  acc_up = simd_sum(acc_up);
  if (gating_sel > 1) {
    acc_gate = simd_sum(acc_gate);
  }

  // Lane 0 applies activation and writes result
  if (thread_context.simdgroup_index == 0) {
    float up_val = acc_up + float(up_biases[bias_base + h_idx]);
    up_val = clamp(up_val, up_clip_min, up_clip_max);

    float activated;
    if (gating_sel <= 1) {
      activated =
          (gating_sel == 0) ? gelu_approx(up_val) : silu(up_val, silu_alpha);
    } else {
      float gate_val = acc_gate + float(up_biases[bias_base + d_ff + h_idx]);
      gate_val = clamp(gate_val, gate_clip_min, gate_clip_max);
      float gate_act = (gating_sel == 2) ? silu(gate_val, silu_alpha)
                                         : gelu_approx(gate_val);
      activated = gate_act * up_val;
    }

    hidden_out[(ulong)global_row * (ulong)d_ff + (ulong)h_idx] = activated;
  }
}