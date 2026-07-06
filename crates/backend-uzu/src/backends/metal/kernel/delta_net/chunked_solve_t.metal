#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

#define SOLVE_T_THREADS 128u

// Emits the DENSE unit-lower-triangular inverse T = (I + A)^{-1} per
// (chunk, v-head), where A[i,k] = beta_i * exp(g_i - g_k) * kk[i,k] for k < i
// (else 0) is the same strictly-lower chunk matrix that `DeltaNetChunkedSolve`
// factors into block inverses. Mode L's mega kernel applies T as one dense
// [C,C] x [C,VT] MMA (Vnew = T . R), so W/U (and BuildWU) disappear; this kernel
// replaces Solve in that pipeline. It is self-contained: it reads the same
// (kk, beta, g) inputs as Solve and does one forward substitution with an
// identity RHS. T is stored BF16 -- exactly the precision the old W/U operands
// carried, so numerics are no worse (state stays f32 throughout the scan).
//
// Layout: one threadgroup per (chunk, v-head). A is staged in threadgroup
// memory (16KB f32) so exp() runs once per entry; T is built in threadgroup
// memory (8KB bf16), column j owned by thread j (columns are independent). The
// per-column recurrence T[i,j] = delta(i,j) - sum_{k=j..i-1} A[i,k] T[k,j] reads
// only entries this thread wrote, so no barrier is needed inside it.
template <uint CHUNK_SIZE>
VARIANTS(CHUNK_SIZE, 16, 32, 64)
PUBLIC KERNEL(DeltaNetChunkedSolveT)(
    device const float* kk,
    device const float* beta,
    device const float* g,
    device bfloat* t_out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& suffix_len,
    threadgroup float a_tg[CHUNK_SIZE * CHUNK_SIZE],
    threadgroup bfloat t_tg[CHUNK_SIZE * CHUNK_SIZE],
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(SOLVE_T_THREADS)
) {
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk_idx = hv_idx / groups_per_head;
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  const uint kk_base = (chunk_idx * num_k_heads + hk_idx) * CHUNK_SIZE * CHUNK_SIZE;

  // Stage the scaled strictly-lower matrix A and zero T.
  for (uint idx = tid; idx < CHUNK_SIZE * CHUNK_SIZE; idx += SOLVE_T_THREADS) {
    const uint i = idx / CHUNK_SIZE;
    const uint k = idx - i * CHUNK_SIZE;
    float a = 0.0f;
    if (k < i && i < valid && k < valid) {
      const float beta_i = beta[(token_base + i) * num_v_heads + hv_idx];
      const float g_i = g[(token_base + i) * num_v_heads + hv_idx];
      const float g_k = g[(token_base + k) * num_v_heads + hv_idx];
      a = beta_i * fast::exp(g_i - g_k) * kk[kk_base + i * CHUNK_SIZE + k];
    }
    a_tg[idx] = a;
    t_tg[idx] = bfloat(0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Forward substitution, one column per thread. T is unit lower triangular:
  // rows above the diagonal stay 0 (already zeroed above).
  const uint j = tid;
  if (j < CHUNK_SIZE) {
    for (uint i = j; i < CHUNK_SIZE; ++i) {
      float acc = (i == j) ? 1.0f : 0.0f;
      for (uint k = j; k < i; ++k) {
        acc -= a_tg[i * CHUNK_SIZE + k] * float(t_tg[k * CHUNK_SIZE + j]);
      }
      t_tg[i * CHUNK_SIZE + j] = bfloat(acc);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint t_base = (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE * CHUNK_SIZE;
  for (uint idx = tid; idx < CHUNK_SIZE * CHUNK_SIZE; idx += SOLVE_T_THREADS) {
    t_out[t_base + idx] = t_tg[idx];
  }
}
