#pragma once

#include <metal_stdlib>

using namespace metal;

// NF4 16-entry codebook from QLoRA paper.
// Stored as `constant` (program-scope, addr space 2).
constant half nf4_codebook[16] = {
    -1.0h,
    -0.6961928h,
    -0.5250730h,
    -0.39491748h,
    -0.28444138h,
    -0.18477343h,
    -0.09105003h,
    0.0h,
    0.07958029h,
    0.16093750h,
    0.24611230h,
    0.33791524h,
    0.44070983h,
    0.56261432h,
    0.72295684h,
    1.0h,
};

// Initialize a per-thread register array of NF4 codebook entries.
inline void nf4_init_register_codebook(thread half codebook[16]) {
  codebook[0] = -1.0h;
  codebook[1] = -0.6961928h;
  codebook[2] = -0.5250730h;
  codebook[3] = -0.39491748h;
  codebook[4] = -0.28444138h;
  codebook[5] = -0.18477343h;
  codebook[6] = -0.09105003h;
  codebook[7] = 0.0h;
  codebook[8] = 0.07958029h;
  codebook[9] = 0.16093750h;
  codebook[10] = 0.24611230h;
  codebook[11] = 0.33791524h;
  codebook[12] = 0.44070983h;
  codebook[13] = 0.56261432h;
  codebook[14] = 0.72295684h;
  codebook[15] = 1.0h;
}

// Pick the codebook entry this lane will own for the simd_shuffle path.
// Lane i (0..16) holds codebook[i]; lanes 16..31 mirror lanes 0..15.
inline half nf4_my_shuffle_entry(uint simd_lane) {
  const uint i = simd_lane & 15u;
  // Use a switch to keep this as scalar selects (no register-array
  // indirection).
  switch (i) {
  case 0:
    return -1.0h;
  case 1:
    return -0.6961928h;
  case 2:
    return -0.5250730h;
  case 3:
    return -0.39491748h;
  case 4:
    return -0.28444138h;
  case 5:
    return -0.18477343h;
  case 6:
    return -0.09105003h;
  case 7:
    return 0.0h;
  case 8:
    return 0.07958029h;
  case 9:
    return 0.16093750h;
  case 10:
    return 0.24611230h;
  case 11:
    return 0.33791524h;
  case 12:
    return 0.44070983h;
  case 13:
    return 0.56261432h;
  case 14:
    return 0.72295684h;
  default:
    return 1.0h;
  }
}

// Cooperatively initialize a threadgroup codebook (16 entries).
inline void nf4_init_tg_codebook(
    threadgroup half* cb,
    uint tid,
    uint tgp_size
) {
  for (uint i = tid; i < 16u; i += tgp_size) {
    half v;
    switch (i) {
    case 0:
      v = -1.0h;
      break;
    case 1:
      v = -0.6961928h;
      break;
    case 2:
      v = -0.5250730h;
      break;
    case 3:
      v = -0.39491748h;
      break;
    case 4:
      v = -0.28444138h;
      break;
    case 5:
      v = -0.18477343h;
      break;
    case 6:
      v = -0.09105003h;
      break;
    case 7:
      v = 0.0h;
      break;
    case 8:
      v = 0.07958029h;
      break;
    case 9:
      v = 0.16093750h;
      break;
    case 10:
      v = 0.24611230h;
      break;
    case 11:
      v = 0.33791524h;
      break;
    case 12:
      v = 0.44070983h;
      break;
    case 13:
      v = 0.56261432h;
      break;
    case 14:
      v = 0.72295684h;
      break;
    default:
      v = 1.0h;
      break;
    }
    cb[i] = v;
  }
}

// Decode a 1-byte OCP/NVIDIA E4M3 FP8 scale (1 sign / 4 exp / 3 mantissa,
// bias 7, no infinities, NaN = S.1111.111, max normal magnitude 448,
// subnormals: exp==0 -> value = mantissa/8 * 2^-6) into a `half`.
// Apple GPUs have no native FP8->half convert, so this is a scalar bit
// unpack. It runs once per group (1 per 64 weights) so it is not on the
// hot per-weight path.
inline half e4m3_to_half(uint8_t v) {
  const uint sign = (v >> 7) & 0x1u;
  const uint exp = (v >> 3) & 0xFu;
  const uint mant = v & 0x7u;
  const half hsign = sign ? -1.0h : 1.0h;

  if (exp == 0u) {
    // Subnormal (or zero): value = mantissa / 8 * 2^-6.
    // 2^-6 / 8 == 2^-9 == 1/512.
    return hsign * (half(mant) * (1.0h / 512.0h));
  }
  // NaN encoding: S.1111.111 (exp==15 && mant==7). Return 0 so a corrupt
  // scale does not poison the whole group; correctness tests never feed it.
  if (exp == 0xFu && mant == 0x7u) {
    return 0.0h;
  }
  // Normal: value = (1 + mant/8) * 2^(exp - 7).
  const half mantissa = 1.0h + half(mant) * (1.0h / 8.0h);
  const int e = int(exp) - 7;
  half scale_pow;
  if (e >= 0) {
    scale_pow = half(1u << uint(e));
  } else {
    scale_pow = 1.0h / half(1u << uint(-e));
  }
  return hsign * mantissa * scale_pow;
}

// ─────────────────────────────────────────────────────────────────────────────
// NF4-ZP (asymmetric per-group offset) extension.
//
// A 4-bit per-group zero-point *index* selects one of 16 fixed offsets from
// `nf4_zp_lut`. Dequant becomes:
//     out = scale * Σ (codebook[nibble] + zp_lut[zp_idx_for_group]) · x
//
// The LUT spans a symmetric range, evenly spaced:
//     nf4_zp_lut[i] = (i / 15.0) - 0.5,  i ∈ [0, 15]
//   → index 0 → -0.5, index 7/8 → ≈0, index 15 → +0.5, step = 1/15 ≈ 0.06667
//
// This is roughly ±0.5 of a codebook step (the NF4 codebook spans [-1, 1] in
// 15 intervals, mean step 2/15 ≈ 0.1333), giving sub-step bias correction.
// These exact values MUST be mirrored by the CPU reference and any bench.
constant half nf4_zp_lut[16] = {
    -0.5h,
    -0.43333334h,
    -0.36666667h,
    -0.3h,
    -0.23333333h,
    -0.16666667h,
    -0.1h,
    -0.033333335h,
    0.033333335h,
    0.1h,
    0.16666667h,
    0.23333333h,
    0.3h,
    0.36666667h,
    0.43333334h,
    0.5h,
};

// Look up the per-group zero-point offset for output row `row`, group
// `group_idx`. Zero-point indices are 4-bit, packed two-per-byte, row-major:
// byte = zp[row * zp_stride + (group_idx >> 1)]; even group → low nibble,
// odd group → high nibble. `zp_stride` = ceil(num_groups / 2). This matches
// the AWQ 4-bit zero-point packing convention used elsewhere in the codebase.
inline half nf4_zp_lookup(
    const device uint8_t* zero_points,
    uint row,
    uint zp_stride,
    uint group_idx
) {
  uint8_t byte = zero_points[row * zp_stride + (group_idx >> 1)];
  uint8_t idx =
      ((group_idx & 1u) == 0u) ? (byte & 0x0fu) : ((byte >> 4) & 0x0fu);
  return nf4_zp_lut[idx];
}
