#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../ssm/ssm_common.h"

using namespace metal;

// In-place RMSNorm + SiLU gate for DeltaNet prefill output.
// One threadgroup per (token, head).
//
// in_out[i] = in_out[i] * inv_rms * norm_weight[i] * silu(z[i])

// TODO: support different head_v_dim via VARIANTS template when needed
#define HEAD_V_DIM 128

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(DeltaNetNormGate)(
    device T* in_out,
    device const T* in_proj,
    device const T* norm_weight,
    constant const uint& num_v_heads,
    constant const uint& head_v_dim,
    constant const uint& value_dim,
    constant const uint& conv_dim,
    constant const uint& total_proj_dim,
    constant const float& norm_epsilon,
    constant const uint& suffix_len,
    threadgroup float shared_scratch[32],
    const ThreadContext thread_context,
    const uint token_idx GROUPS(suffix_len),
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(HEAD_V_DIM)
) {
  const bool active = (tid < head_v_dim);
  const uint base = token_idx * value_dim + hv_idx * head_v_dim;

  // Load value
  float o_i = active ? float(in_out[base + tid]) : 0.0f;

  // RMSNorm: cooperative sum of squares
  float o_sq = active ? o_i * o_i : 0.0f;
  float o_sumsq = threadgroup_cooperative_reduce_sum<HEAD_V_DIM>(
      o_sq,
      shared_scratch,
      tid,
      thread_context
  );
  float inv_rms = rsqrt(o_sumsq / float(head_v_dim) + norm_epsilon);

  // Apply norm + SiLU gate (in-place)
  if (active) {
    float nw = float(norm_weight[tid]);
    uint z_idx =
        token_idx * total_proj_dim + conv_dim + hv_idx * head_v_dim + tid;
    float z_silu = activate_silu(float(in_proj[z_idx]));
    in_out[base + tid] = static_cast<T>(o_i * inv_rms * nw * z_silu);
  }
}
