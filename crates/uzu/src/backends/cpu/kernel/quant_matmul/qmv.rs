use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

/// CPU reference implementation of quantized matrix-vector multiply.
///
/// Weight layout is transposed compared to qmm:
///   weights: [N, K], scales: [N, num_groups_k], zero_points: [N, zp_stride_k]
/// where groups are along the K dimension.
///
/// Computes y[i, j] = sum_l( x[i, l] * dequant(w[j, l]) ) for each batch i and output j.
pub fn qmv<T: ArrayElement + Float>(
    weights: *const u32,
    scales: *const T,
    zero_points: Option<*const u8>,
    biases: Option<*const T>,
    input: *const T,
    output: *mut T,
    in_vec_size: usize,
    out_vec_size: usize,
    batch_size: usize,
    use_zero_points: bool,
    use_mlx_quant: bool,
    group_size: usize,
    bits: usize,
) {
    let num_groups_k = in_vec_size.div_ceil(group_size);
    let zp_stride = if bits == 4 {
        (num_groups_k + 1) / 2
    } else {
        num_groups_k
    };

    // Number of weight elements packed per u32
    let pack_factor = if bits == 4 {
        8
    } else {
        4
    };

    unsafe {
        for i in 0..batch_size {
            for j in 0..out_vec_size {
                let mut acc = 0.0f32;

                for l in 0..in_vec_size {
                    // Weights are [N, K]: row j, column l
                    let weight_linear_idx = j * in_vec_size + l;

                    let val_q = if bits == 4 {
                        let u32_idx = weight_linear_idx / pack_factor;
                        let bit_offset = (weight_linear_idx % pack_factor) * 4;
                        ((*weights.add(u32_idx) >> bit_offset) & 0xF) as f32
                    } else {
                        let u32_idx = weight_linear_idx / pack_factor;
                        let byte_offset = (weight_linear_idx % pack_factor) * 8;
                        ((*weights.add(u32_idx) >> byte_offset) & 0xFF) as f32
                    };

                    let val_a = (*input.add(i * in_vec_size + l)).to_f32().unwrap();

                    // Scales are [N, num_groups_k]: groups along K
                    let group_idx = l / group_size;
                    let scale = (*scales.add(j * num_groups_k + group_idx)).to_f32().unwrap();

                    let bias = if use_zero_points {
                        let zp = zero_points.unwrap();
                        // Zero points are [N, zp_stride_k]
                        let zp_val = if bits == 4 {
                            let byte_index = j * zp_stride + (group_idx >> 1);
                            let byte_val = *zp.add(byte_index);
                            if (group_idx & 1) == 0 {
                                (byte_val & 0x0F) as f32
                            } else {
                                ((byte_val >> 4) & 0x0F) as f32
                            }
                        } else {
                            *zp.add(j * zp_stride + group_idx) as f32
                        };
                        -scale * zp_val
                    } else if use_mlx_quant {
                        // Biases are [N, num_groups_k]
                        (*biases.unwrap().add(j * num_groups_k + group_idx)).to_f32().unwrap()
                    } else {
                        0.0f32
                    };

                    acc += val_a * (scale * val_q + bias);
                }

                *output.add(i * out_vec_size + j) = T::from(acc).unwrap();
            }
        }
    }
}

#[kernel(QuantizedMatmulQmv)]
#[variants(T, f32, f16, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmv<T: ArrayElement + Float, const GROUP_SIZE: u32, const BITS: u32>(
    weights: *const u32,
    scales: *const T,
    #[optional(use_zero_points)] zero_points: Option<*const u8>,
    #[optional(use_mlx_quant)] biases: Option<*const T>,
    input: *const T,
    output: *mut T,
    in_vec_size: u32,
    out_vec_size: u32,
    batch_size: u32,
    #[specialize] use_zero_points: bool,
    #[specialize] use_mlx_quant: bool,
) {
    qmv::<T>(
        weights,
        scales,
        zero_points,
        biases,
        input,
        output,
        in_vec_size as usize,
        out_vec_size as usize,
        batch_size as usize,
        use_zero_points,
        use_mlx_quant,
        GROUP_SIZE as usize,
        BITS as usize,
    );
}
