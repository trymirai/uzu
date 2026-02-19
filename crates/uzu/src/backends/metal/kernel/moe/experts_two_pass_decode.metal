#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"
#include "moe_commons.h"
using namespace metal;

// === Pass A: Vectorized GEMV with float4 loads ===
// Structure: 4 simdgroups (128 threads), each outputs 1 hidden element
// Each simdgroup: 32 threads reduce d_model using float4 vectorized loads
// Grid: (h_blocks, rows, 1) where h_blocks = ceil(d_ff/4)
template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MoeExpertsDecodePassA)(
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
    const Simd simd,
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
  const uint h_idx = h_block_idx * 4 + simd.group_idx;
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
    uint base_idx = i * 128 + simd.lane_idx * 4;

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
  uint leftover_start = vec_iters * 128 + simd.lane_idx;
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
  if (simd.lane_idx == 0) {
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

// === Pass B: Simdgroup cooperation along K for coalescing ===
// W2 layout [E, d_model, d_ff] - 32 threads cooperate on one output, reading
// consecutive K elements
#define THREADS_PER_SIMD 32
#define SIMDGROUPS_PER_TG 8

template <typename T, typename AccumT>
VARIANTS(T, float, half, bfloat)
VARIANTS(AccumT, float)
KERNEL(MoeExpertsDecodeDownFused2D)(
    device const float* hidden,         // [total_rows, d_ff] - f32 from Pass A
    device const uint* row_expert_map,  // [total_rows] - direct row->expert lookup
    device const T* w2_all,             // [E, d_model, d_ff] - layout
    device const T* down_biases,        // [E, d_model]
    device T* y_out,                    // [total_rows, d_model]
    constant uint& total_rows,
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& e,
    const Simd simd,
    const uint tgpig_x GROUPS(d_model.div_ceil(SIMDGROUPS_PER_TG)),
    const uint tgpig_y GROUPS(total_rows),
    const uint tid THREADS(256) 
) {
  const uint row_idx = tgpig_y;

  // Each simdgroup computes one output column
  const uint my_col = tgpig_x * SIMDGROUPS_PER_TG + simd.group_idx;
  if (my_col >= d_model)
    return;

  const uint expert_idx = row_expert_map[row_idx];

  // Base addresses for this output column
  const ulong hidden_base = (ulong)row_idx * (ulong)d_ff;
  const ulong w2_col_base = (ulong)expert_idx * (ulong)d_model * (ulong)d_ff +
                            (ulong)my_col * (ulong)d_ff;

  // Dual accumulators for ILP: breaks FMA dependency chains
  AccumT acc0 = AccumT(0.0);
  AccumT acc1 = AccumT(0.0);
  AccumT acc = AccumT(0.0);

  // Main loop: stride-32 with vec8 loads for ILP
  const uint k_iters = d_ff / THREADS_PER_SIMD;
  const uint k_vec_iters = k_iters / 8;

  for (uint iter = 0; iter < k_vec_iters; ++iter) {
    const uint k_base = iter * (8 * THREADS_PER_SIMD) + simd.lane_idx;

    // hidden: stride-32 per lane, already f32 from Pass A
    const AccumT h0 = hidden[hidden_base + k_base + 0 * THREADS_PER_SIMD];
    const AccumT h1 = hidden[hidden_base + k_base + 1 * THREADS_PER_SIMD];
    const AccumT h2 = hidden[hidden_base + k_base + 2 * THREADS_PER_SIMD];
    const AccumT h3 = hidden[hidden_base + k_base + 3 * THREADS_PER_SIMD];
    const AccumT h4 = hidden[hidden_base + k_base + 4 * THREADS_PER_SIMD];
    const AccumT h5 = hidden[hidden_base + k_base + 5 * THREADS_PER_SIMD];
    const AccumT h6 = hidden[hidden_base + k_base + 6 * THREADS_PER_SIMD];
    const AccumT h7 = hidden[hidden_base + k_base + 7 * THREADS_PER_SIMD];

    // W2: lane-coalesced
    const AccumT w0 =
        AccumT(w2_all[w2_col_base + k_base + 0 * THREADS_PER_SIMD]);
    const AccumT w1 =
        AccumT(w2_all[w2_col_base + k_base + 1 * THREADS_PER_SIMD]);
    const AccumT w2 =
        AccumT(w2_all[w2_col_base + k_base + 2 * THREADS_PER_SIMD]);
    const AccumT w3 =
        AccumT(w2_all[w2_col_base + k_base + 3 * THREADS_PER_SIMD]);
    const AccumT w4 =
        AccumT(w2_all[w2_col_base + k_base + 4 * THREADS_PER_SIMD]);
    const AccumT w5 =
        AccumT(w2_all[w2_col_base + k_base + 5 * THREADS_PER_SIMD]);
    const AccumT w6 =
        AccumT(w2_all[w2_col_base + k_base + 6 * THREADS_PER_SIMD]);
    const AccumT w7 =
        AccumT(w2_all[w2_col_base + k_base + 7 * THREADS_PER_SIMD]);

    // dual trees for ILP
    acc0 = fma(h0, w0, acc0);
    acc1 = fma(h1, w1, acc1);
    acc0 = fma(h2, w2, acc0);
    acc1 = fma(h3, w3, acc1);
    acc0 = fma(h4, w4, acc0);
    acc1 = fma(h5, w5, acc1);
    acc0 = fma(h6, w6, acc0);
    acc1 = fma(h7, w7, acc1);
  }

  // fold the vectorized contribution once
  acc += (acc0 + acc1);

  // Handle remaining full iterations
  for (uint iter = k_vec_iters * 8; iter < k_iters; ++iter) {
    const uint k = iter * THREADS_PER_SIMD + simd.lane_idx;
    acc =
        fma(AccumT(hidden[hidden_base + k]),
            AccumT(w2_all[w2_col_base + k]),
            acc);
  }

  // Handle leftover elements (d_ff % 32)
  const uint leftover_start = k_iters * THREADS_PER_SIMD;
  if (leftover_start + simd.lane_idx < d_ff) {
    const uint k = leftover_start + simd.lane_idx;
    acc =
        fma(AccumT(hidden[hidden_base + k]),
            AccumT(w2_all[w2_col_base + k]),
            acc);
  }

  // Simdgroup reduction
  AccumT result = simd_sum(acc);

  // Lane 0 writes result
  if (simd.lane_idx == 0) {
    const ulong bias_idx = (ulong)expert_idx * (ulong)d_model + (ulong)my_col;
    result += AccumT(down_biases[bias_idx]);

    const ulong out_idx = (ulong)row_idx * (ulong)d_model + (ulong)my_col;
    y_out[out_idx] = T(result);
  }
}