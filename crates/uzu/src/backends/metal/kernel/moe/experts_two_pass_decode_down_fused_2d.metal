#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "moe_commons.h"
using namespace metal;

// === Pass B: Simdgroup cooperation along K for coalescing ===
// W2 layout [E, d_model, d_ff] - 32 threads cooperate on one output, reading
// consecutive K elements
#define THREADS_PER_SIMD 32
#define SIMDGROUPS_PER_TG 8

template <typename T, typename AccumT>
VARIANTS(T, float, half, bfloat)
VARIANTS(AccumT, float)
PUBLIC KERNEL(MoeExpertsDecodeDownFused2D)(
    device const float* hidden,         // [total_rows, d_ff] - f32 from Pass A
    device const uint* row_expert_map,  // [total_rows] - direct row->expert lookup
    device const T* w2_all,             // [E, d_model, d_ff] - layout
    device const T* down_biases,        // [E, d_model]
    device T* y_out,                    // [total_rows, d_model]
    constant uint& total_rows,
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& e,
    const ThreadContext thread_context,
    const uint tgpig_x GROUPS(d_model.div_ceil(SIMDGROUPS_PER_TG)),
    const uint tgpig_y GROUPS(total_rows),
    const uint tid THREADS(256) 
) {
  const uint row_idx = tgpig_y;

  // Each simdgroup computes one output column
  const uint my_col =
      tgpig_x * SIMDGROUPS_PER_TG + thread_context.threadgroup_index;
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
    const uint k_base =
        iter * (8 * THREADS_PER_SIMD) + thread_context.simdgroup_index;

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
    const uint k = iter * THREADS_PER_SIMD + thread_context.simdgroup_index;
    acc =
        fma(AccumT(hidden[hidden_base + k]),
            AccumT(w2_all[w2_col_base + k]),
            acc);
  }

  // Handle leftover elements (d_ff % 32)
  const uint leftover_start = k_iters * THREADS_PER_SIMD;
  if (leftover_start + thread_context.simdgroup_index < d_ff) {
    const uint k = leftover_start + thread_context.simdgroup_index;
    acc =
        fma(AccumT(hidden[hidden_base + k]),
            AccumT(w2_all[w2_col_base + k]),
            acc);
  }

  // Simdgroup reduction
  AccumT result = simd_sum(acc);

  // Lane 0 writes result
  if (thread_context.simdgroup_index == 0) {
    const ulong bias_idx = (ulong)expert_idx * (ulong)d_model + (ulong)my_col;
    result += AccumT(down_biases[bias_idx]);

    const ulong out_idx = (ulong)row_idx * (ulong)d_model + (ulong)my_col;
    y_out[out_idx] = T(result);
  }
}