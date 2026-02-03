#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"
using namespace metal;

#define SIMD_SIZE 32

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

// ============================================================================
// Pass A: x @ W13[expert] → hidden[k]
// Each threadgroup: 4 simdgroups, outputs 4 elements (1 per simdgroup)
// Each simdgroup: 32 threads reduce d_model with float4 vectorized loads
// Grid: (ceil(d_ff/4), K)
// ============================================================================
template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MoeExpertsDecodeSinglePassA)(
    device const T* x,
    device const int* topk_ids,
    device const T* w13_all,
    device const T* biases,
    device float* hidden_out,
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& k,
    constant float& silu_alpha,
    constant float& gate_clip_min,
    constant float& gate_clip_max,
    constant float& up_clip_min,
    constant float& up_clip_max,
    constant uint& gating_sel, // 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU
    const uint h_block_idx GROUPS(d_ff.div_ceil(4)),
    const uint k_slot GROUPS(k),
    const uint tid THREADS(128)
) {
  const uint simd_gid = tid / SIMD_SIZE;
  const uint simd_lid = tid % SIMD_SIZE;

  const int expert_id = topk_ids[k_slot];
  if (expert_id < 0)
    return;
  const uint expert_u = uint(expert_id);

  const uint h_idx = h_block_idx * 4 + simd_gid;
  if (h_idx >= d_ff)
    return;

  const ulong w13_stride = (ulong)d_model * (ulong)(2 * d_ff);
  const ulong w13_base = (ulong)expert_u * w13_stride;
  const ulong bias_base = (ulong)expert_u * (ulong)(2 * d_ff);

  device const T* w_up_row = w13_all + w13_base + (ulong)h_idx * (ulong)d_model;
  device const T* w_gate_row =
      w13_all + w13_base + (ulong)(d_ff + h_idx) * (ulong)d_model;

  float acc_up = 0.0f;
  float acc_gate = 0.0f;

  const uint vec_iters = d_model / 128;

  for (uint i = 0; i < vec_iters; ++i) {
    uint base_idx = i * 128 + simd_lid * 4;

    device const T* x_vec2 = reinterpret_cast<device const T*>(x + base_idx);
    device const T* w_up_vec2 = reinterpret_cast<device const T*>(w_up_row + base_idx);
    acc_up += float(x_vec2[0]) * float(w_up_vec2[0]);
    acc_up += float(x_vec2[1]) * float(w_up_vec2[1]);
    acc_up += float(x_vec2[2]) * float(w_up_vec2[2]);
    acc_up += float(x_vec2[3]) * float(w_up_vec2[3]);

    if (gating_sel > 1) {
      device const T* w_gate_vec2 = reinterpret_cast<device const T*>(w_gate_row + base_idx);
      acc_gate += float(x_vec2[0]) * float(w_gate_vec2[0]);
      acc_gate += float(x_vec2[1]) * float(w_gate_vec2[1]);
      acc_gate += float(x_vec2[2]) * float(w_gate_vec2[2]);
      acc_gate += float(x_vec2[3]) * float(w_gate_vec2[3]);
    }
  }

  uint leftover_start = vec_iters * 128 + simd_lid;
  for (uint idx = leftover_start; idx < d_model; idx += 32) {
    float xv = float(x[idx]);
    acc_up += xv * float(w_up_row[idx]);
    if (gating_sel > 1) {
      acc_gate += xv * float(w_gate_row[idx]);
    }
  }

  acc_up = simd_sum(acc_up);
  if (gating_sel > 1) {
    acc_gate = simd_sum(acc_gate);
  }

  if (simd_lid == 0) {
    float up_val = clamp(
        acc_up + float(biases[bias_base + h_idx]),
        up_clip_min,
        up_clip_max
    );

    float activated;
    if (gating_sel <= 1) {
      activated =
          (gating_sel == 0) ? gelu_approx(up_val) : silu(up_val, silu_alpha);
    } else {
      float gate_val = clamp(
          acc_gate + float(biases[bias_base + d_ff + h_idx]),
          gate_clip_min,
          gate_clip_max
      );
      float gate_act = (gating_sel == 2) ? silu(gate_val, silu_alpha)
                                         : gelu_approx(gate_val);
      activated = gate_act * up_val;
    }

    hidden_out[(ulong)k_slot * (ulong)d_ff + (ulong)h_idx] = activated;
  }
}


// ============================================================================
// Pass B (fused with finalize): hidden[k] @ W2[expert] → y (directly)
// Computes: y[d] = Σ_k prob[k] * (hidden[k] @ W2[expert_k, d] + bias)
// Each simdgroup computes one final output element
// Grid: (ceil(d_model/8), 1)  - NOT per K!
// ============================================================================
template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MoeExpertsDecodeSinglePassB)(
    device const float* hidden, // [K, d_ff]
    device const int* topk_ids, // [K]
    device const T* topk_probs, // [K]
    device const T* w2_all,     // [E, d_model, d_ff]
    device const T* biases,     // [E, d_model]
    device T* y,                // [d_model]
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& k_input,
    const uint d_block GROUPS(d_model.div_ceil(8)),
    const uint tid THREADS(256)
) {
  const uint simd_gid = tid / SIMD_SIZE;
  const uint simd_lid = tid % SIMD_SIZE;

  const uint my_col = d_block * 8 + simd_gid;
  if (my_col >= d_model)
    return;

  const uint vec_iters = d_ff / 128;
  const ulong w2_expert_stride = (ulong)d_model * (ulong)d_ff;

  float final_acc = 0.0f;

  // Loop over k_input experts
  for (uint k = 0; k < k_input; ++k) {
    const uint expert_u = uint(topk_ids[k]);
    const float prob = float(topk_probs[k]);

    device const float* hidden_ptr = hidden + (ulong)k * (ulong)d_ff;
    device const T* w2_ptr = w2_all + (ulong)expert_u * w2_expert_stride +
                             (ulong)my_col * (ulong)d_ff;

    float acc = 0.0f;

    // Vectorized reduction
    for (uint i = 0; i < vec_iters; ++i) {
      uint base_idx = i * 128 + simd_lid * 4;

      float4 h_vec = *reinterpret_cast<device const float4*>(hidden_ptr + base_idx);
      device const T* w_vec2 = reinterpret_cast<device const T*>(w2_ptr + base_idx);
      acc += h_vec.x * float(w_vec2[0]);
      acc += h_vec.y * float(w_vec2[1]);
      acc += h_vec.z * float(w_vec2[2]);
      acc += h_vec.w * float(w_vec2[3]);
    }

    // Remainder
    for (uint idx = vec_iters * 128 + simd_lid; idx < d_ff; idx += 32) {
      acc += hidden_ptr[idx] * float(w2_ptr[idx]);
    }

    // Simdgroup reduction and accumulate
    float result = simd_sum(acc);
    if (simd_lid == 0) {
      result += float(biases[(ulong)expert_u * (ulong)d_model + (ulong)my_col]);
      final_acc += prob * result;
    }
  }

  if (simd_lid == 0) {
    y[my_col] = T(final_acc);
  }
}