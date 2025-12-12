#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant uint GATING_SEL
    [[function_constant(30)]]; // 0=GELU,1=SiLU,2=SwiGLU,3=GEGLU

static inline float gelu_approx(float x) {
  const float k0 = 0.7978845608f;
  const float k1 = 0.044715f;
  if (x > 10.0f)
    return x;
  if (x < -10.0f)
    return 0.0f;
  return 0.5f * x *
         (1.0f + tanh(clamp(k0 * (x + k1 * x * x * x), -10.0f, 10.0f)));
}

static inline float silu(float x, float alpha) {
  return x / (1.0f + exp(-alpha * x));
}

// === Pass A: Vectorized GEMV with float4 loads ===
// Structure: 4 simdgroups (128 threads), each outputs 1 hidden element
// Each simdgroup: 32 threads reduce d_model using float4 vectorized loads
// Grid: (h_blocks, rows, 1) where h_blocks = ceil(d_ff/4)
template <typename T, typename T4>
void moe_experts_decode_pass_a_impl(
    device const T* X_perm,            // [total_rows, d_model]
    device const uint* expert_offsets, // [E + 1]
    device const T* W13_all,           // [E, 2*d_ff, d_model]
    device const T* up_biases,         // [E, 2*d_ff]
    device float* hidden_out,          // [total_rows, d_ff]
    uint d_model,
    uint d_ff,
    uint E,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    uint expert_idx,
    uint row_in_expert,
    uint h_block_idx,
    uint simd_gid,
    uint simd_lid
) {
  // Validate row bounds
  const uint seg_start = expert_offsets[expert_idx];
  const uint seg_end = expert_offsets[expert_idx + 1];
  const uint global_row = seg_start + row_in_expert;
  if (global_row >= seg_end)
    return;

  // Each simdgroup outputs one hidden element
  const uint h_idx = h_block_idx * 4 + simd_gid;
  if (h_idx >= d_ff)
    return;

  // Base addresses
  const ulong w13_stride = (ulong)d_model * (ulong)(2 * d_ff);
  const ulong w13_base = (ulong)expert_idx * w13_stride;
  const ulong bias_base = (ulong)expert_idx * (ulong)(2 * d_ff);
  const ulong x_row_base = (ulong)global_row * (ulong)d_model;

  device const T* x_ptr = X_perm + x_row_base;
  device const T* w_up_row = W13_all + w13_base + (ulong)h_idx * (ulong)d_model;
  device const T* w_gate_row =
      W13_all + w13_base + (ulong)(d_ff + h_idx) * (ulong)d_model;

  float acc_up = 0.0f;
  float acc_gate = 0.0f;

  // Vectorized reduction: 32 threads × 4 elements = 128 elements per iteration
  const uint vec_iters = d_model / 128;

  for (uint i = 0; i < vec_iters; ++i) {
    uint base_idx = i * 128 + simd_lid * 4;

    T4 x_vec = *reinterpret_cast<device const T4*>(x_ptr + base_idx);
    T4 w_up_vec = *reinterpret_cast<device const T4*>(w_up_row + base_idx);

    acc_up += float(x_vec.x) * float(w_up_vec.x);
    acc_up += float(x_vec.y) * float(w_up_vec.y);
    acc_up += float(x_vec.z) * float(w_up_vec.z);
    acc_up += float(x_vec.w) * float(w_up_vec.w);

    if (GATING_SEL > 1) {
      T4 w_gate_vec =
          *reinterpret_cast<device const T4*>(w_gate_row + base_idx);
      acc_gate += float(x_vec.x) * float(w_gate_vec.x);
      acc_gate += float(x_vec.y) * float(w_gate_vec.y);
      acc_gate += float(x_vec.z) * float(w_gate_vec.z);
      acc_gate += float(x_vec.w) * float(w_gate_vec.w);
    }
  }

  // Handle leftover elements
  uint leftover_start = vec_iters * 128 + simd_lid;
  for (uint idx = leftover_start; idx < d_model; idx += 32) {
    float xv = float(x_ptr[idx]);
    acc_up += xv * float(w_up_row[idx]);
    if (GATING_SEL > 1) {
      acc_gate += xv * float(w_gate_row[idx]);
    }
  }

  // Simdgroup reduction
  acc_up = simd_sum(acc_up);
  if (GATING_SEL > 1) {
    acc_gate = simd_sum(acc_gate);
  }

  // Lane 0 applies activation and writes result
  if (simd_lid == 0) {
    float up_val = acc_up + float(up_biases[bias_base + h_idx]);
    up_val = clamp(up_val, up_clip_min, up_clip_max);

    float activated;
    if (GATING_SEL <= 1) {
      activated =
          (GATING_SEL == 0) ? gelu_approx(up_val) : silu(up_val, silu_alpha);
    } else {
      float gate_val = acc_gate + float(up_biases[bias_base + d_ff + h_idx]);
      gate_val = clamp(gate_val, gate_clip_min, gate_clip_max);
      float gate_act = (GATING_SEL == 2) ? silu(gate_val, silu_alpha)
                                         : gelu_approx(gate_val);
      activated = gate_act * up_val;
    }

    hidden_out[(ulong)global_row * (ulong)d_ff + (ulong)h_idx] = activated;
  }
}

#define MOE_PASS_A_KERNEL(DTYPE, DTYPE4, SUFFIX)                               \
  kernel void moe_experts_decode_pass_a_##SUFFIX(                              \
      device const DTYPE* X_perm [[buffer(0)]],                                \
      device const uint* expert_offsets [[buffer(1)]],                         \
      device const DTYPE* W13_all [[buffer(2)]],                               \
      device float* hidden_out [[buffer(3)]],                                  \
      device const DTYPE* up_biases [[buffer(4)]],                             \
      constant uint& d_model [[buffer(5)]],                                    \
      constant uint& d_ff [[buffer(6)]],                                       \
      constant uint& E [[buffer(7)]],                                          \
      constant float& gate_clip_min [[buffer(8)]],                             \
      constant float& gate_clip_max [[buffer(9)]],                             \
      constant float& up_clip_min [[buffer(10)]],                              \
      constant float& up_clip_max [[buffer(11)]],                              \
      constant float& silu_alpha [[buffer(12)]],                               \
      uint3 tgpig [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]                              \
  ) {                                                                          \
    moe_experts_decode_pass_a_impl<DTYPE, DTYPE4>(                             \
        X_perm,                                                                \
        expert_offsets,                                                        \
        W13_all,                                                               \
        up_biases,                                                             \
        hidden_out,                                                            \
        d_model,                                                               \
        d_ff,                                                                  \
        E,                                                                     \
        gate_clip_min,                                                         \
        gate_clip_max,                                                         \
        up_clip_min,                                                           \
        up_clip_max,                                                           \
        silu_alpha,                                                            \
        tgpig.y,                                                               \
        tgpig.z,                                                               \
        tgpig.x,                                                               \
        simd_gid,                                                              \
        simd_lid                                                               \
    );                                                                         \
  }

MOE_PASS_A_KERNEL(bfloat, bfloat4, bf16)
MOE_PASS_A_KERNEL(half, half4, f16)
MOE_PASS_A_KERNEL(float, float4, f32)

// === Helper kernels for indirect dispatch of Pass A ===

// Count tiles per expert: tiles = (num_rows > 0) ? num_rows * h_blocks : 0
kernel void moe_pass_a_tile_counts(
    device const uint* expert_offsets [[buffer(0)]], // [E+1]
    device uint* tile_counts [[buffer(1)]],          // [E]
    constant uint& E [[buffer(2)]],
    constant uint& h_blocks [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
  if (tid >= E)
    return;
  const uint start = expert_offsets[tid];
  const uint end = expert_offsets[tid + 1];
  const uint num_rows = end - start;
  tile_counts[tid] = (num_rows > 0) ? (num_rows * h_blocks) : 0;
}

// Exclusive scan of tile_counts to get tile_offsets and total_tiles
kernel void moe_pass_a_tile_scan(
    device const uint* tile_counts [[buffer(0)]], // [E]
    device uint* tile_offsets [[buffer(1)]],      // [E+1]
    device uint* total_tiles [[buffer(2)]],       // [1]
    constant uint& E [[buffer(3)]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup uint* scratch [[threadgroup(0)]]
) {
  // Simple single-threadgroup scan (works for E <= 1024)
  const uint idx = lid;

  // Load into threadgroup memory
  if (idx < E) {
    scratch[idx] = tile_counts[idx];
  } else {
    scratch[idx] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Kogge-Stone scan
  uint val = scratch[idx];
  for (uint offset = 1; offset < 1024; offset *= 2) {
    uint temp = 0;
    if (idx >= offset && idx < E) {
      temp = scratch[idx - offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (idx >= offset && idx < E) {
      val += temp;
      scratch[idx] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write exclusive scan (shift right by 1)
  if (idx == 0) {
    tile_offsets[0] = 0;
  }
  if (idx < E) {
    tile_offsets[idx + 1] = scratch[idx];
    if (idx == E - 1) {
      total_tiles[0] = scratch[idx];
    }
  }
}

// Build row→expert map: one thread per routed row
kernel void moe_pass_a_build_row_map(
    device const uint* expert_offsets [[buffer(0)]], // [E+1]
    device uint* row_expert_map [[buffer(1)]],       // [total_rows]
    constant uint& total_rows [[buffer(2)]],
    constant uint& E [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
  if (tid >= total_rows)
    return;

  uint left = 0u;
  uint right = E;
  const uint row = tid;

  while (left + 1u < right) {
    const uint mid = (left + right) >> 1;
    if (row < expert_offsets[mid]) {
      right = mid;
    } else {
      left = mid;
    }
  }

  row_expert_map[row] = left;
}

// Build tile map entries from row→expert map
kernel void moe_pass_a_build_tile_map(
    device const uint* expert_offsets [[buffer(0)]], // [E+1]
    device const uint* tile_offsets [[buffer(1)]],   // [E+1]
    device const uint* row_expert_map [[buffer(2)]], // [total_rows]
    device uint* tile_map [[buffer(3)]],             // [total_tiles * 3]
    constant uint& total_rows [[buffer(4)]],
    constant uint& h_blocks [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
  const uint total_tiles = total_rows * h_blocks;
  if (tid >= total_tiles)
    return;

  const uint row_idx = tid / h_blocks;
  const uint h_block = tid % h_blocks;

  if (row_idx >= total_rows)
    return;

  const uint expert_idx = row_expert_map[row_idx];
  const uint row_start = expert_offsets[expert_idx];
  const uint row_in_expert = row_idx - row_start;
  const uint tile_base =
      tile_offsets[expert_idx] + row_in_expert * h_blocks + h_block;

  tile_map[tile_base * 3 + 0] = h_block;
  tile_map[tile_base * 3 + 1] = expert_idx;
  tile_map[tile_base * 3 + 2] = row_in_expert;
}

// Write dispatch args for indirect dispatch (reusable from tiled version)
kernel void moe_pass_a_write_dispatch_args(
    device const uint* total_tiles [[buffer(0)]], // [1]
    device uint* dispatch_args
    [[buffer(1)]], // [3] - MTLDispatchThreadgroupsIndirectArguments
    constant uint& num_tiles_y [[buffer(2)]], // usually 1 for Pass A
    uint tid [[thread_position_in_grid]]
) {
  if (tid > 0)
    return;
  dispatch_args[0] = total_tiles[0]; // x dimension = total tiles
  dispatch_args[1] = num_tiles_y;    // y dimension
  dispatch_args[2] = 1;              // z dimension
}

// Modified Pass A that reads from tile map for indirect dispatch
template <typename T, typename T4>
void moe_experts_decode_pass_a_indirect_impl(
    device const T* X_perm,
    device const uint* expert_offsets,
    device const T* W13_all,
    device const T* up_biases,
    device float* hidden_out,
    device const uint* tile_map, // [total_tiles * 3]
    uint d_model,
    uint d_ff,
    uint E,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    uint tile_idx, // flat threadgroup index
    uint simd_gid,
    uint simd_lid
) {
  // Read tile descriptor
  const uint h_block_idx = tile_map[tile_idx * 3 + 0];
  const uint expert_idx = tile_map[tile_idx * 3 + 1];
  const uint row_in_expert = tile_map[tile_idx * 3 + 2];

  // Call original implementation
  moe_experts_decode_pass_a_impl<T, T4>(
      X_perm,
      expert_offsets,
      W13_all,
      up_biases,
      hidden_out,
      d_model,
      d_ff,
      E,
      gate_clip_min,
      gate_clip_max,
      up_clip_min,
      up_clip_max,
      silu_alpha,
      expert_idx,
      row_in_expert,
      h_block_idx,
      simd_gid,
      simd_lid
  );
}

#define MOE_PASS_A_INDIRECT_KERNEL(DTYPE, DTYPE4, SUFFIX)                      \
  kernel void moe_experts_decode_pass_a_indirect_##SUFFIX(                     \
      device const DTYPE* X_perm [[buffer(0)]],                                \
      device const uint* expert_offsets [[buffer(1)]],                         \
      device const DTYPE* W13_all [[buffer(2)]],                               \
      device float* hidden_out [[buffer(3)]],                                  \
      device const DTYPE* up_biases [[buffer(4)]],                             \
      constant uint& d_model [[buffer(5)]],                                    \
      constant uint& d_ff [[buffer(6)]],                                       \
      constant uint& E [[buffer(7)]],                                          \
      constant float& gate_clip_min [[buffer(8)]],                             \
      constant float& gate_clip_max [[buffer(9)]],                             \
      constant float& up_clip_min [[buffer(10)]],                              \
      constant float& up_clip_max [[buffer(11)]],                              \
      constant float& silu_alpha [[buffer(12)]],                               \
      device const uint* tile_map [[buffer(13)]],                              \
      uint3 tgpig [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]                              \
  ) {                                                                          \
    moe_experts_decode_pass_a_indirect_impl<DTYPE, DTYPE4>(                    \
        X_perm,                                                                \
        expert_offsets,                                                        \
        W13_all,                                                               \
        up_biases,                                                             \
        hidden_out,                                                            \
        tile_map,                                                              \
        d_model,                                                               \
        d_ff,                                                                  \
        E,                                                                     \
        gate_clip_min,                                                         \
        gate_clip_max,                                                         \
        up_clip_min,                                                           \
        up_clip_max,                                                           \
        silu_alpha,                                                            \
        tgpig.x,                                                               \
        simd_gid,                                                              \
        simd_lid                                                               \
    );                                                                         \
  }

MOE_PASS_A_INDIRECT_KERNEL(bfloat, bfloat4, bf16)
MOE_PASS_A_INDIRECT_KERNEL(half, half4, f16)
MOE_PASS_A_INDIRECT_KERNEL(float, float4, f32)

// === Pass B: Simdgroup cooperation along K for coalescing ===
// W2 layout [E, d_model, d_ff] - 32 threads cooperate on one output, reading
// consecutive K elements

template <typename T, typename AccumT>
void moe_experts_decode_down_fused_2d_impl(
    device const float* hidden, // [total_rows, d_ff] - f32 from Pass A
    device const uint*
        row_expert_map,          // [total_rows] - direct row->expert lookup
    device const T* w2_all,      // [E, d_model, d_ff] - layout
    device const T* down_biases, // [E, d_model]
    device T* y_out,             // [total_rows, d_model]
    uint total_rows,
    uint d_model,
    uint d_ff,
    uint E,
    uint2 tgpig,
    uint simd_gid,
    uint simd_lid
) {
  constexpr uint THREADS_PER_SIMD = 32;
  constexpr uint SIMDGROUPS_PER_TG = 8;

  const uint row_idx = tgpig.y;
  if (row_idx >= total_rows)
    return;

  // Each simdgroup computes one output column
  const uint my_col = tgpig.x * SIMDGROUPS_PER_TG + simd_gid;
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
    const uint k_base = iter * (8 * THREADS_PER_SIMD) + simd_lid;

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
    const uint k = iter * THREADS_PER_SIMD + simd_lid;
    acc =
        fma(AccumT(hidden[hidden_base + k]),
            AccumT(w2_all[w2_col_base + k]),
            acc);
  }

  // Handle leftover elements (d_ff % 32)
  const uint leftover_start = k_iters * THREADS_PER_SIMD;
  if (leftover_start + simd_lid < d_ff) {
    const uint k = leftover_start + simd_lid;
    acc =
        fma(AccumT(hidden[hidden_base + k]),
            AccumT(w2_all[w2_col_base + k]),
            acc);
  }

  // Simdgroup reduction
  AccumT result = simd_sum(acc);

  // Lane 0 writes result
  if (simd_lid == 0) {
    const ulong bias_idx = (ulong)expert_idx * (ulong)d_model + (ulong)my_col;
    result += AccumT(down_biases[bias_idx]);

    const ulong out_idx = (ulong)row_idx * (ulong)d_model + (ulong)my_col;
    y_out[out_idx] = T(result);
  }
}

#define MOE_PASS_B_FUSED_2D_KERNEL(DTYPE, SUFFIX)                              \
  kernel void moe_experts_decode_down_fused_2d_##SUFFIX(                       \
      device const float* hidden [[buffer(0)]],                                \
      device const uint* row_expert_map [[buffer(1)]],                         \
      device const DTYPE* w2_all [[buffer(2)]],                                \
      device const DTYPE* down_biases [[buffer(3)]],                           \
      device DTYPE* y_out [[buffer(4)]],                                       \
      constant uint& total_rows [[buffer(5)]],                                 \
      constant uint& d_model [[buffer(6)]],                                    \
      constant uint& d_ff [[buffer(7)]],                                       \
      constant uint& E [[buffer(8)]],                                          \
      uint2 tgpig [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]                              \
  ) {                                                                          \
    moe_experts_decode_down_fused_2d_impl<DTYPE, float>(                       \
        hidden,                                                                \
        row_expert_map,                                                        \
        w2_all,                                                                \
        down_biases,                                                           \
        y_out,                                                                 \
        total_rows,                                                            \
        d_model,                                                               \
        d_ff,                                                                  \
        E,                                                                     \
        tgpig,                                                                 \
        simd_gid,                                                              \
        simd_lid                                                               \
    );                                                                         \
  }

MOE_PASS_B_FUSED_2D_KERNEL(bfloat, bf16)
MOE_PASS_B_FUSED_2D_KERNEL(half, f16)
MOE_PASS_B_FUSED_2D_KERNEL(float, f32)
