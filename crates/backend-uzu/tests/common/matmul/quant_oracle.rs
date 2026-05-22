use backend_uzu::backends::common::gpu_types::{QuantizationMethod, QuantizationMode};

/// Read a 4-bit weight from a u32-packed buffer (LSB-first, 8 weights per u32).
fn unpack_u4(
    w_packed: &[u32],
    index: usize,
) -> u8 {
    let word = w_packed[index / 8];
    ((word >> ((index % 8) * 4)) & 0xF) as u8
}

/// Read an 8-bit weight from a u32-packed buffer (4 weights per u32).
fn unpack_u8(
    w_packed: &[u32],
    index: usize,
) -> u8 {
    let word = w_packed[index / 4];
    ((word >> ((index % 4) * 8)) & 0xFF) as u8
}

fn unpack_zp_u4(
    zp_packed: &[u8],
    row_stride: usize,
    row: usize,
    group: usize,
) -> u8 {
    let byte = zp_packed[row * row_stride + group / 2];
    if group % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }
}

/// CPU reference for the GPU quantized GEMM kernel.
///
/// Mirrors the kernel contract: D[m, n] = sum_k A[m, k] * dequant(B[n, k]),
/// where dequant follows MLX-style ScaleBias or ScaleZeroPoint:
///   ScaleBias:        dq = scale * w + bias
///   ScaleZeroPoint:   dq = scale * (w - zp)
///
/// `weights_raw[n * k + l]` is the unpacked weight value (0..max). The packed
/// u32 buffer fed to the GPU must be produced from this same raw layout.
#[allow(clippy::too_many_arguments)]
pub fn quant_gemm_reference(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    weights_raw: &[u8],
    scales: &[f32],
    biases: Option<&[f32]>,
    zero_points_packed: Option<&[u8]>,
    quant_method: QuantizationMethod,
    mode: QuantizationMode,
    group_size: usize,
) -> Vec<f32> {
    let bits: u32 = match mode {
        QuantizationMode::U4 => 4,
        QuantizationMode::I8 | QuantizationMode::U8 => 8,
    };
    let num_groups_k = k.div_ceil(group_size);
    let zp_stride = if bits == 4 { num_groups_k.div_ceil(2) } else { num_groups_k };

    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for g in 0..num_groups_k {
                let scale = scales[j * num_groups_k + g];
                let l_start = g * group_size;
                let l_end = (l_start + group_size).min(k);
                for l in l_start..l_end {
                    let w_idx = j * k + l;
                    let w = weights_raw[w_idx] as f32;
                    let dq = match quant_method {
                        QuantizationMethod::ScaleBias => {
                            let bias = biases.expect("ScaleBias requires biases")[j * num_groups_k + g];
                            scale * w + bias
                        },
                        QuantizationMethod::ScaleZeroPoint => {
                            let zp_packed = zero_points_packed.expect("ScaleZeroPoint requires zero_points");
                            let zp = if bits == 4 {
                                unpack_zp_u4(zp_packed, zp_stride, j, g) as f32
                            } else {
                                zp_packed[j * zp_stride + g] as f32
                            };
                            scale * (w - zp)
                        },
                    };
                    acc += a[i * k + l] * dq;
                }
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Same as [`quant_gemm_reference`] but unpacking weights from the u32 buffer
/// directly. Useful when only `w_packed` is on hand.
#[allow(clippy::too_many_arguments)]
pub fn quant_gemm_reference_packed(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    w_packed: &[u32],
    scales: &[f32],
    biases: Option<&[f32]>,
    zero_points_packed: Option<&[u8]>,
    quant_method: QuantizationMethod,
    mode: QuantizationMode,
    group_size: usize,
) -> Vec<f32> {
    let bits: u32 = match mode {
        QuantizationMode::U4 => 4,
        QuantizationMode::I8 | QuantizationMode::U8 => 8,
    };
    let mut weights_raw = vec![0u8; n * k];
    for w_idx in 0..(n * k) {
        weights_raw[w_idx] = if bits == 4 { unpack_u4(w_packed, w_idx) } else { unpack_u8(w_packed, w_idx) };
    }
    quant_gemm_reference(m, n, k, a, &weights_raw, scales, biases, zero_points_packed, quant_method, mode, group_size)
}
