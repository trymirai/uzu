use num_traits::Float;

use crate::{array::ArrayElement, backends::common::gpu_types::QuantizationMethod};

pub fn qmm_transposed<T: ArrayElement + Float>(
    weights: *const u32,
    scales: *const T,
    zero_points: Option<*const u8>,
    biases: Option<*const T>,
    input: *const T,
    output: *mut T,
    in_vec_size: usize,
    out_vec_size: usize,
    batch_size: usize,
    quant_method: QuantizationMethod,
    group_size: usize,
    bits: usize,
) {
    let num_groups_k = in_vec_size.div_ceil(group_size);
    let zp_stride = if bits == 4 {
        (num_groups_k + 1) / 2
    } else {
        num_groups_k
    };

    unsafe {
        for i in 0..batch_size {
            for j in 0..out_vec_size {
                let mut acc = 0.0f32;

                for l in 0..in_vec_size {
                    let group_idx = l / group_size;

                    let weight_linear_idx = j * in_vec_size + l;

                    let val_q = if bits == 4 {
                        let u32_idx = weight_linear_idx / 8;
                        let bit_offset = (weight_linear_idx % 8) * 4;
                        ((weights.add(u32_idx).read_unaligned() >> bit_offset) & 0xF) as f32
                    } else {
                        let u32_idx = weight_linear_idx / 4;
                        let byte_offset = (weight_linear_idx % 4) * 8;
                        ((weights.add(u32_idx).read_unaligned() >> byte_offset) & 0xFF) as f32
                    };

                    let val_a = (*input.add(i * in_vec_size + l)).to_f32().unwrap();
                    let scale_t = *scales.add(j * num_groups_k + group_idx);

                    let w_dequant_f32 = match quant_method {
                        QuantizationMethod::ScaleBias => {
                            let bias_t = *biases.unwrap().add(j * num_groups_k + group_idx);
                            scale_t.to_f32().unwrap() * val_q + bias_t.to_f32().unwrap()
                        },
                        QuantizationMethod::ScaleZeroPoint => {
                            let zp = zero_points.unwrap();
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
                            scale_t.to_f32().unwrap() * (val_q - zp_val)
                        },
                        QuantizationMethod::ScaleSymmetric => {
                            let midpoint = (1 << (bits - 1)) as f32;
                            scale_t.to_f32().unwrap() * (val_q - midpoint)
                        },
                    };

                    acc += val_a * w_dequant_f32;
                }

                *output.add(i * out_vec_size + j) = T::from(acc).unwrap();
            }
        }
    }
}
