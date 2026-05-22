use backend_uzu::backends::common::gpu_types::{QuantizationMethod, QuantizationMode};

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
                let bias = match quant_method {
                    QuantizationMethod::ScaleBias => biases.expect("ScaleBias requires biases")[j * num_groups_k + g],
                    QuantizationMethod::ScaleZeroPoint => {
                        let zp_packed = zero_points_packed.expect("ScaleZeroPoint requires zero_points");
                        let zp = if bits == 4 {
                            unpack_zp_u4(zp_packed, zp_stride, j, g) as f32
                        } else {
                            zp_packed[j * zp_stride + g] as f32
                        };
                        -scale * zp
                    },
                };
                let mut group_acc = 0.0f32;
                let mut group_sum = 0.0f32;
                for l in l_start..l_end {
                    let w_idx = j * k + l;
                    let w = weights_raw[w_idx] as f32;
                    let a_val = a[i * k + l];
                    group_acc += a_val * w;
                    group_sum += a_val;
                }
                acc += scale * group_acc + bias * group_sum;
            }
            out[i * n + j] = acc;
        }
    }
    out
}

