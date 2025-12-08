#include <metal_stdlib>
using namespace metal;

// Defaults must match Rust launcher
#define BM 32
#define BN 64

template <typename T>
inline void moe_finalize_impl(
    device const int* tok2row, // [T*K], -1 if dropped
    device const T* probs,     // [T*K]
    device const T* Y_partial, // [sum_k, d_model]
    device T* Y,               // [T, d_model]
    constant uint& T_count,
    constant uint& d_model,
    constant uint& K,
    uint lid,
    uint3 tgpig
) {
  if (T_count == 0u || d_model == 0u)
    return;
  const uint tile_m0 = tgpig.y * BM;
  const uint tile_n0 = tgpig.x * BN;
  if (tile_m0 >= T_count || tile_n0 >= d_model)
    return;
  const uint m_rows = min((uint)BM, T_count - tile_m0);
  const uint n_cols = min((uint)BN, d_model - tile_n0);

  // 128 threads per TG expected
  for (uint idx = lid; idx < m_rows * n_cols; idx += 128u) {
    const uint mi = idx / n_cols;
    const uint nj = idx % n_cols;
    const uint t = tile_m0 + mi;
    const uint f = tile_n0 + nj;
    float acc = 0.0f;
    const uint base = t * K;
    for (uint k = 0; k < K; ++k) {
      const int row = tok2row[base + k];
      if (row >= 0) {
        const ulong yidx = (ulong)(uint)row * (ulong)d_model + (ulong)f;
        float prob = (float)probs[base + k];
        if (!isfinite(prob)) {
          prob = 0.0f;
        }
        float val = (float)Y_partial[yidx];
        if (!isfinite(val)) {
          val = 0.0f;
        }
        acc = fma(prob, val, acc);
      }
    }
    if (!isfinite(acc)) {
      acc = 0.0f;
    }
    Y[t * d_model + f] = T(acc);
  }
}

#define DEFINE_MOE_FINALIZE_KERNEL(SUFFIX, T)                                  \
  kernel void moe_finalize_##SUFFIX(                                           \
      device const int* tok2row [[buffer(0)]],                                 \
      device const T* probs [[buffer(1)]],                                     \
      device const T* Y_partial [[buffer(2)]],                                 \
      device T* Y [[buffer(3)]],                                               \
      constant uint& T_count [[buffer(4)]],                                    \
      constant uint& d_model [[buffer(5)]],                                    \
      constant uint& K [[buffer(6)]],                                          \
      uint lid [[thread_index_in_threadgroup]],                                \
      uint3 tgpig [[threadgroup_position_in_grid]]                             \
  ) {                                                                          \
    moe_finalize_impl<                                                         \
        T>(tok2row, probs, Y_partial, Y, T_count, d_model, K, lid, tgpig);     \
  }

DEFINE_MOE_FINALIZE_KERNEL(f16, half)
DEFINE_MOE_FINALIZE_KERNEL(bf16, bfloat)
DEFINE_MOE_FINALIZE_KERNEL(f32, float)
