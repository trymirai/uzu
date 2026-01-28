#include <metal_stdlib>
#include "../definitions.metal"
using namespace metal;

// Defaults must match Rust launcher
#define BM 32
#define BN 64

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MoeFinalize)(
    device const int* tok2row, // [T*K], -1 if dropped
    device const T* probs,     // [T*K]
    device const T* y_partial, // [sum_k, d_model]
    device T* y,               // [T, d_model]
    constant uint& t_count,
    constant uint& d_model,
    constant uint& k_input,
    const uint lid THREADS(128),
    const uint tgpig_x GROUPS((d_model + BN - 1) / BN),
    const uint tgpig_y GROUPS((t_count + BM - 1) / BM)
) {
  if (t_count == 0u || d_model == 0u)
    return;
  const uint tile_m0 = tgpig_x * BM;
  const uint tile_n0 = tgpig_y * BN;
  if (tile_m0 >= t_count || tile_n0 >= d_model)
    return;
  const uint m_rows = min((uint)BM, t_count - tile_m0);
  const uint n_cols = min((uint)BN, d_model - tile_n0);

  // 128 threads per TG expected
  for (uint idx = lid; idx < m_rows * n_cols; idx += 128u) {
    const uint mi = idx / n_cols;
    const uint nj = idx % n_cols;
    const uint t = tile_m0 + mi;
    const uint f = tile_n0 + nj;
    float acc = 0.0f;
    const uint base = t * k_input;
    for (uint k = 0; k < k_input; ++k) {
      const int row = tok2row[base + k];
      if (row >= 0) {
        const ulong yidx = (ulong)(uint)row * (ulong)d_model + (ulong)f;
        float prob = (float)probs[base + k];
        if (!isfinite(prob)) {
          prob = 0.0f;
        }
        float val = (float)y_partial[yidx];
        if (!isfinite(val)) {
          val = 0.0f;
        }
        acc = fma(prob, val, acc);
      }
    }
    if (!isfinite(acc)) {
      acc = 0.0f;
    }
    y[t * d_model + f] = T(acc);
  }
}