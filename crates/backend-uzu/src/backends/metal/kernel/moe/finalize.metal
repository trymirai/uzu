#include <metal_stdlib>
#include "../common/dsl.h"

#define BM 32
#define BN 64

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MoeFinalize)(
    device const T* probs,         // [T*K]
    device const T* route_outputs, // [T*K, d_model]
    device T* y,                   // [T, d_model]
    constant uint& t_count,
    constant uint& d_model,
    constant uint& k_input,
    const uint lid THREADS(128),
    const uint tg_n GROUPS(d_model.div_ceil(BN)),
    const uint tg_m GROUPS(t_count.div_ceil(BM))
) {
  const uint tile_n0 = tg_n * BN;
  const uint tile_m0 = tg_m * BM;
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
      const uint route = base + k;
      const ulong output_index = ulong(route) * ulong(d_model) + ulong(f);
      float prob = (float)probs[route];
      if (!isfinite(prob)) {
        prob = 0.0f;
      }
      float val = (float)route_outputs[output_index];
      if (!isfinite(val)) {
        val = 0.0f;
      }
      acc = fma(prob, val, acc);
    }
    if (!isfinite(acc)) {
      acc = 0.0f;
    }
    y[t * d_model + f] = T(acc);
  }
}
