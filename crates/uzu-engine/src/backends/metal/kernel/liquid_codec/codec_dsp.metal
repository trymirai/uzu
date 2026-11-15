#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Liquid-Codec data-movement + GFSQ kernels.
//
// The codec interleaves a few tiny gathers with GEMMs; keeping them as their
// own kernels is optimal because every operand is small (<= 20k elements) and
// the gathers exist precisely so the GEMMs stay dense (the Metal matmul kernel
// has no GatherGEMM path).
// ---------------------------------------------------------------------------

// Frame stack for the k=6 / stride=6 front-end conv:
//   src [T, D] -> dst [T/6, 6*D], dst[row][d*6 + k] = src[row*6 + k][d]
// The column order (input-major, tap-minor) matches the conv weight rows
// [out, in, k] once flattened, so `frontend_out` stays a plain GEMM.
template <typename T>
VARIANTS(T, float, bfloat, half)
KERNEL(CodecPackFrames)(
    device const T* src,
    device T* dst,
    constant const uint& rows,
    constant const uint& dim,
    const uint index AXIS(rows * dim, 256)
) {
  const uint width = 6u * dim;
  const uint row = index / width;
  const uint col = index % width;
  const uint d = col / 6u;
  const uint k = col % 6u;
  dst[index] = src[(row * 6u + k) * dim + d];
}

// Causal k=3 window for the front-end conv (pad=2, right trim):
//   src [T, D] -> dst [T, 3*D], dst[t][d*3 + k] = src[t + k - 2][d]
template <typename T>
VARIANTS(T, float, bfloat, half)
KERNEL(CodecPackWindow3)(
    device const T* src,
    device T* dst,
    constant const uint& rows,
    constant const uint& dim,
    const uint index AXIS(rows * 3u * dim, 256)
) {
  const uint width = 3u * dim;
  const uint row = index / width;
  const uint col = index % width;
  const uint d = col / 3u;
  const uint k = col % 3u;
  const int src_row = int(row) + int(k) - 2;
  const T value = src_row >= 0 ? src[uint(src_row) * dim + d] : static_cast<T>(0.0f);
  dst[index] = value;
}

// Nearest-exact x6 upsample of the decoder embeddings:
//   src [T, D] -> dst [6*T, D], dst[t*6 + r][d] = src[t][d]
template <typename T>
VARIANTS(T, float, bfloat, half)
KERNEL(CodecPackRepeat)(
    device const T* src,
    device T* dst,
    constant const uint& rows,
    constant const uint& dim,
    const uint index AXIS(rows * 6u * dim, 256)
) {
  const uint row = index / dim;
  dst[index] = src[(row / 6u) * dim + (index % dim)];
}

// GFSQ encode: latents [T, 4*n_q] (already projected, bf16 grid) -> indices
// [n_q, T]. `params` holds, per FSQ dim (4), (half_levels, offset, half_width,
// shift=atanh(offset / half_levels)); one thread per (codebook, frame).
template <typename T>
VARIANTS(T, float, bfloat, half)
KERNEL(CodecFsqEncode)(
    device const T* latents,
    device const float4* params,
    device int32_t* indices,
    constant const uint& frames,
    constant const uint& codebooks,
    const uint index AXIS(frames * 12, 32)
) {
  const uint t = index / codebooks;
  const uint cb = index % codebooks;
  float index_value = 0.0f;
  for (uint d = 0; d < 4u; ++d) {
    const float4 p = params[d];
    const float x = float(latents[t * (4u * codebooks) + cb * 4u + d]);
    const float bounded = tanh(x + p.w) * p.x - p.y;
    const float r = rint(bounded) / p.z;
    const float basis = (d == 0u) ? 1.0f : (d == 1u ? 8.0f : (d == 2u ? 56.0f : 336.0f));
    index_value += (r * p.z + p.z) * basis;
  }
  indices[cb * frames + t] = int32_t(rint(index_value));
}

// GFSQ decode: indices [n_q, T] -> latents [T, 4*n_q] (bf16 grid).
// `params` holds, per FSQ dim (4), (levels, basis, half_width, 0).
KERNEL(CodecFsqDecode)(
    device const int32_t* indices,
    device const int4* params,
    device bfloat* latents,
    constant const uint& frames,
    constant const uint& codebooks,
    const uint index AXIS(frames * 12, 32)
) {
  const uint t = index / codebooks;
  const uint cb = index % codebooks;
  for (uint d = 0; d < 4u; ++d) {
    const int4 p = params[d];
    const int32_t value = (indices[cb * frames + t] / p.y) % p.x;
    const float f = float(value - p.z) / float(p.z);
    latents[t * (4u * codebooks) + cb * 4u + d] = bfloat(f);
  }
}

