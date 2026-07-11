#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;

#define UPDATE_THREADS 512
// Bounds shared_o, which holds one o_i per dv for the cross-dv RMSNorm.
#define UPDATE_MAX_HEAD_V_DIM 128

// One simd group per output dim dv; 32 lanes cover Dk (coalesced state IO,
// reduction via simd_sum). Activations are model dtype T; state stays float.
template <typename T, uint HEAD_K_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_K_DIM, 128)
PUBLIC KERNEL(DeltaNetUpdate)(
    device const T* in_proj,
    device const float* a_log,
    device const float* dt_bias,
    device const float* norm_weight,
    device float* state,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const float& norm_epsilon,
    threadgroup float shared_o[UPDATE_MAX_HEAD_V_DIM],
    threadgroup float shared_scratch[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(UPDATE_THREADS)
) {
  static_assert(HEAD_K_DIM % METAL_SIMD_SIZE == 0, "HEAD_K_DIM must be a multiple of METAL_SIMD_SIZE");
  constexpr uint ELEMS = HEAD_K_DIM / METAL_SIMD_SIZE;      // Dk per lane (128/32 = 4)
  constexpr uint NUM_SG = UPDATE_THREADS / METAL_SIMD_SIZE; // simd groups / tg (16)

  const uint lane = tid % METAL_SIMD_SIZE;
  const uint sg = tid / METAL_SIMD_SIZE;

  const uint conv_dim = 2 * key_dim + value_dim;
  const uint hk = hv_idx / (num_v_heads / num_k_heads);

  // Load + normalize q/k; each simd group recomputes the norm to skip a
  // barrier.
  float q[ELEMS];
  float k[ELEMS];
  float q_sq = 0.0f;
  float k_sq = 0.0f;
  for (uint i = 0; i < ELEMS; ++i) {
    const uint dk = lane + METAL_SIMD_SIZE * i;
    q[i] = float(in_proj[hk * HEAD_K_DIM + dk]);
    k[i] = float(in_proj[key_dim + hk * HEAD_K_DIM + dk]);
    q_sq += q[i] * q[i];
    k_sq += k[i] * k[i];
  }
  const float q_inv_norm = rsqrt(simd_sum(q_sq) + 1e-6f);
  const float k_inv_norm = rsqrt(simd_sum(k_sq) + 1e-6f);
  const float q_scale = rsqrt(float(HEAD_K_DIM));
  float kq_partial = 0.0f;
  for (uint i = 0; i < ELEMS; ++i) {
    q[i] *= q_inv_norm * q_scale;
    k[i] *= k_inv_norm;
    kq_partial += q[i] * k[i];
  }
  const float kq_dot = simd_sum(kq_partial);

  // beta / decay (scalar per head)
  const float beta_raw = float(in_proj[conv_dim + value_dim + hv_idx]);
  const float beta = 1.0f / (1.0f + fast::exp(-beta_raw));
  const float a_raw = float(in_proj[conv_dim + value_dim + num_v_heads + hv_idx]);
  const float sp = activate_softplus(a_raw + float(dt_bias[hv_idx]));
  const float decay = fast::exp(-fast::exp(float(a_log[hv_idx])) * sp);

  // Delta rule over the dv owned by this simd group. State is [Hv, Dv, Dk].
  for (uint dv = sg; dv < head_v_dim; dv += NUM_SG) {
    const uint state_row = (hv_idx * head_v_dim + dv) * HEAD_K_DIM;
    const float v_i = float(in_proj[2 * key_dim + hv_idx * head_v_dim + dv]);

    float s[ELEMS];
    float sq_partial = 0.0f;
    float sk_partial = 0.0f;
    for (uint i = 0; i < ELEMS; ++i) {
      const uint dk = lane + METAL_SIMD_SIZE * i;
      s[i] = state[state_row + dk];
      sq_partial += s[i] * q[i];
      sk_partial += s[i] * k[i];
    }
    const float sq_acc = simd_sum(sq_partial);
    const float sk_acc = simd_sum(sk_partial);

    const float retrieved_i = decay * sk_acc;
    const float delta_i = beta * (v_i - retrieved_i);
    const float o_i = decay * sq_acc + delta_i * kq_dot;

    for (uint i = 0; i < ELEMS; ++i) {
      const uint dk = lane + METAL_SIMD_SIZE * i;
      state[state_row + dk] = decay * s[i] + k[i] * delta_i;
    }

    if (lane == 0) {
      shared_o[dv] = o_i;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // RMSNorm over all dv, then SiLU gate + write out.
  float o_sq = 0.0f;
  for (uint dv = tid; dv < head_v_dim; dv += UPDATE_THREADS) {
    const float o = shared_o[dv];
    o_sq += o * o;
  }
  const float o_sumsq =
      threadgroup_cooperative_reduce<SimdReduceSum<float>, UPDATE_THREADS>(o_sq, shared_scratch, thread_context);
  const float inv_rms = rsqrt(o_sumsq / float(head_v_dim) + norm_epsilon);

  for (uint dv = sg; dv < head_v_dim; dv += NUM_SG) {
    if (lane == 0) {
      const float nw = float(norm_weight[dv]);
      const float z_i = float(in_proj[conv_dim + hv_idx * head_v_dim + dv]);
      const float z_silu = activate_silu(z_i);
      out[hv_idx * head_v_dim + dv] = static_cast<T>(shared_o[dv] * inv_rms * nw * z_silu);
    }
  }
}
