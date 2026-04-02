use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(QuantizedMatmulQmm)]
#[variants(T, f32, f16, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmm<T: ArrayElement + Float, const GROUP_SIZE: i32, const BITS: i32>(
    w: *const u32,
    scales: *const T,
    #[optional(use_zero_points)] zero_points: Option<*const u8>,
    #[optional(use_mlx_quant)] biases: Option<*const T>,
    x: *const T,
    y: *mut T,
    k: i32,
    n: i32,
    m: i32,
    #[specialize] use_zero_points: bool,
    #[specialize] use_mlx_quant: bool,
    #[specialize] aligned_k: bool,
) {
    let k = k as usize;
    let n = n as usize;
    let m = m as usize;
    let group_size = GROUP_SIZE as usize;
    let num_groups_n = (n + group_size - 1) / group_size;

    let zp_stride = if BITS == 4 {
        (num_groups_n + 1) / 2
    } else {
        num_groups_n
    };

    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                let group_idx = j / group_size;

                for l in 0..k {
                    let weight_linear_idx = l * n + j;

                    let val_q = if BITS == 4 {
                        let u32_idx = weight_linear_idx / 8;
                        let bit_offset = (weight_linear_idx % 8) * 4;
                        ((*w.add(u32_idx) >> bit_offset) & 0xF) as f32
                    } else {
                        let u32_idx = weight_linear_idx / 4;
                        let byte_offset = (weight_linear_idx % 4) * 8;
                        ((*w.add(u32_idx) >> byte_offset) & 0xFF) as f32
                    };

                    let val_a = (*x.add(i * k + l)).to_f32().unwrap();
                    let scale = (*scales.add(l * num_groups_n + group_idx)).to_f32().unwrap();

                    let bias = if use_zero_points {
                        let zp = zero_points.unwrap();
                        let zp_val = if BITS == 4 {
                            let byte_index = l * zp_stride + (group_idx >> 1);
                            let byte_val = *zp.add(byte_index);
                            if (group_idx & 1) == 0 {
                                (byte_val & 0x0F) as f32
                            } else {
                                ((byte_val >> 4) & 0x0F) as f32
                            }
                        } else {
                            *zp.add(l * zp_stride + group_idx) as f32
                        };
                        -scale * zp_val
                    } else if use_mlx_quant {
                        (*biases.unwrap().add(l * num_groups_n + group_idx)).to_f32().unwrap()
                    } else {
                        0.0f32
                    };

                    acc += val_a * (scale * val_q + bias);
                }

                *y.add(i * n + j) = T::from(acc).unwrap();
            }
        }
    }
}
