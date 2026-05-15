//! CPU reference for the NF4 quantized matmul with 1-byte E4M3 (FP8)
//! per-group scales. These reference helpers are plain Rust (the NF4 GPU
//! kernels are bench-only and have no PUBLIC `#[kernel]` CPU pair), so they
//! are invoked directly by the correctness test rather than through the
//! kernel dispatcher.

/// NF4 16-entry codebook (QLoRA paper), matching `nf4_common.h`.
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.5250730,
    -0.39491748,
    -0.28444138,
    -0.18477343,
    -0.09105003,
    0.0,
    0.07958029,
    0.16093750,
    0.24611230,
    0.33791524,
    0.44070983,
    0.56261432,
    0.72295684,
    1.0,
];

/// Encode an f32 to the nearest OCP/NVIDIA E4M3 byte (1 sign / 4 exp /
/// 3 mantissa, bias 7, no infinities, NaN = S.1111.111, max normal
/// magnitude 448, subnormals supported). Round-to-nearest-even on the
/// 3-bit mantissa; saturates to +/-448.
pub fn f32_to_e4m3(v: f32) -> u8 {
    if v == 0.0 || !v.is_finite() {
        return 0;
    }
    let sign: u8 = if v < 0.0 {
        0x80
    } else {
        0x00
    };
    let a = v.abs();

    // Max normal magnitude is 448 = (1 + 7/8) * 2^8.
    const MAX_NORMAL: f32 = 448.0;
    if a >= MAX_NORMAL {
        return sign | 0x7E; // exp=15, mant=6 -> 448 (0x7F is NaN)
    }

    // Smallest normal = 2^-6. Below that we encode subnormals:
    // value = mant/8 * 2^-6, mant in 1..=7. Smallest subnormal = 2^-9.
    const MIN_NORMAL: f32 = 0.015625; // 2^-6
    if a < MIN_NORMAL {
        // Round a / 2^-9 to nearest integer mantissa in 0..=7.
        let scaled = a / (2.0f32.powi(-9));
        let m = round_half_even(scaled).clamp(0.0, 7.0) as u32;
        if m == 0 {
            return sign; // +/-0
        }
        return sign | (m as u8);
    }

    // Normal: find exponent e such that 2^e <= a < 2^(e+1), e in -6..=8.
    let mut e = a.log2().floor() as i32;
    e = e.clamp(-6, 8);
    // mantissa = round((a / 2^e - 1) * 8), in 0..=8 (8 carries to next exp).
    let frac = a / 2.0f32.powi(e) - 1.0;
    let mut m = round_half_even(frac * 8.0) as i32;
    if m == 8 {
        m = 0;
        e += 1;
        if e > 8 {
            return sign | 0x7E; // saturate to 448
        }
    }
    let exp_field = (e + 7) as u32; // bias 7, in 1..=15
    sign | ((exp_field as u8) << 3) | (m as u8 & 0x7)
}

/// Decode an E4M3 byte to f32 (mirror of `e4m3_to_half` in `nf4_common.h`).
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte >> 7) & 1 == 1 {
        -1.0f32
    } else {
        1.0f32
    };
    let exp = ((byte >> 3) & 0xF) as i32;
    let mant = (byte & 0x7) as i32;
    if exp == 0 {
        // Subnormal / zero: value = mant/8 * 2^-6 = mant * 2^-9.
        return sign * (mant as f32) * 2.0f32.powi(-9);
    }
    if exp == 0xF && mant == 0x7 {
        // NaN encoding -> 0 (matches the GPU decode's poison-guard).
        return 0.0;
    }
    let mantissa = 1.0 + (mant as f32) / 8.0;
    sign * mantissa * 2.0f32.powi(exp - 7)
}

/// Round half to even (banker's rounding) for f32.
fn round_half_even(x: f32) -> f32 {
    let r = x.round();
    if (x - x.floor() - 0.5).abs() < f32::EPSILON {
        // Exactly halfway: round to even.
        let f = x.floor();
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    } else {
        r
    }
}

/// Round-trip an f32 scale through E4M3 so the CPU reference matches the
/// GPU's decoded scale exactly.
pub fn quantize_scale_e4m3(scale: f32) -> f32 {
    e4m3_to_f32(f32_to_e4m3(scale))
}

fn nf4_nibble(
    weights: *const u32,
    weight_linear_idx: usize,
) -> usize {
    let u32_idx = weight_linear_idx / 8;
    let bit_offset = (weight_linear_idx % 8) * 4;
    unsafe { ((weights.add(u32_idx).read_unaligned() >> bit_offset) & 0xF) as usize }
}

/// CPU reference for `Nf4QmvE4m3`.
///
/// Weights are transposed `[N, K]` packed 4-bit nibbles. `scales` is
/// `[N, num_groups_k]` of E4M3 bytes. Groups run along K.
/// `y[i, j] = sum_l x[i, l] * dequant_scale(j, l/gs) * codebook[nibble]`.
#[allow(clippy::too_many_arguments)]
pub fn nf4_qmv_e4m3_ref(
    weights: *const u32,
    scales: *const u8,
    input: *const f32,
    output: *mut f32,
    in_vec_size: usize,
    out_vec_size: usize,
    batch_size: usize,
    group_size: usize,
) {
    let num_groups_k = in_vec_size.div_ceil(group_size);
    unsafe {
        for i in 0..batch_size {
            for j in 0..out_vec_size {
                let mut acc = 0.0f32;
                for l in 0..in_vec_size {
                    let nibble = nf4_nibble(weights, j * in_vec_size + l);
                    let val_a = *input.add(i * in_vec_size + l);
                    let group_idx = l / group_size;
                    let s_byte = *scales.add(j * num_groups_k + group_idx);
                    let scale = e4m3_to_f32(s_byte);
                    acc += val_a * scale * NF4_CODEBOOK[nibble];
                }
                *output.add(i * out_vec_size + j) = acc;
            }
        }
    }
}

/// CPU reference for `Nf4QmmE4m3`. Same math/layout as the QMV reference;
/// kept as a separate fn so the QMM correctness test is self-contained.
#[allow(clippy::too_many_arguments)]
pub fn nf4_qmm_e4m3_ref(
    weights: *const u32,
    scales: *const u8,
    input: *const f32,
    output: *mut f32,
    in_vec_size: usize,
    out_vec_size: usize,
    batch_size: usize,
    group_size: usize,
) {
    nf4_qmv_e4m3_ref(weights, scales, input, output, in_vec_size, out_vec_size, batch_size, group_size);
}
