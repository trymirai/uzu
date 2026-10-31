use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::{
            gemm::{GemmBPrologueKind, GemmDTransform},
            hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        },
        cpu::kernel::activation_transform::hadamard_transform,
    },
    data_type::DataTypeRepr,
};

#[kernel(MoeGemv)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 0, 32, 64)]
#[variants(BITS, 0, 4, 8)]
#[constraint((BITS == 0) == (GROUP_SIZE == 0))]
pub fn moe_gemv<T: DataTypeRepr + Float, const GROUP_SIZE: u32, const BITS: u32>(
    b: *const u8,
    #[optional(b_prologue != GemmBPrologueKind::FullPrecision)] scales: Option<*const T>,
    #[optional(b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)] zero_points: Option<*const u8>,
    #[optional(b_prologue == GemmBPrologueKind::ScaleBiasDequant)] biases: Option<*const T>,
    a: *const T,
    expert_ids: *const u32,
    d: *mut T,
    #[optional(output_transform.contains(GemmDTransform::BIAS))] output_bias: Option<*const T>,
    #[optional(output_transform.contains(GemmDTransform::RHT))] hadamard_factors: Option<*const i32>,
    in_vec_size: u32,
    out_vec_size: u32,
    num_routes: u32,
    active_experts_per_token: u32,
    #[allow(unused)] group_count_x: u32,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
    zero_point_output_stride: u32,
    zero_point_group_stride: u32,
    #[specialize] b_prologue: GemmBPrologueKind,
    #[specialize] output_transform: GemmDTransform,
    #[allow(unused)]
    #[specialize]
    input_aligned: bool,
    #[specialize] signed_codes: bool,
    #[allow(unused)]
    #[specialize]
    full_tile: bool,
    #[specialize] a_per_token: bool,
) {
    let k_u = in_vec_size as usize;
    let n_u = out_vec_size as usize;
    let routes = num_routes as usize;
    let top_k = active_experts_per_token as usize;
    let bits = BITS as usize;
    let group_size = GROUP_SIZE as usize;

    for route_idx in 0..routes {
        let expert_id = unsafe { *expert_ids.add(route_idx) } as usize;
        let a_row_idx = if a_per_token {
            route_idx / top_k
        } else {
            route_idx
        };

        let expert_b = unsafe { b.add(expert_id * b_expert_stride_bytes as usize) };
        let expert_scales = scales.map(|ptr| unsafe { ptr.add(expert_id * scale_expert_stride as usize) });
        let expert_zero_points =
            zero_points.map(|ptr| unsafe { ptr.add(expert_id * zero_point_expert_stride as usize) });
        let expert_biases = biases.map(|ptr| unsafe { ptr.add(expert_id * scale_expert_stride as usize) });
        let route_a = unsafe { a.add(a_row_idx * k_u) };
        let route_d = unsafe { d.add(route_idx * n_u) };
        let expert_output_bias = output_bias.map(|ptr| unsafe { ptr.add(expert_id * n_u) });
        let expert_hadamard = hadamard_factors.map(|ptr| unsafe { ptr.add(expert_id * n_u) });

        let mut row_out = vec![0.0f32; n_u];
        for (col, out_slot) in row_out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for inner in 0..k_u {
                let a_val = unsafe { *route_a.add(inner) }.to_f32().unwrap();
                let b_val = if BITS == 0 {
                    let b_typed = expert_b as *const T;
                    unsafe { *b_typed.add(col * k_u + inner) }.to_f32().unwrap()
                } else {
                    let pack_factor = 32 / bits;
                    let weight_linear_index = col * k_u + inner;
                    let word_index = weight_linear_index / pack_factor;
                    let code_index_in_word = weight_linear_index % pack_factor;
                    let bit_offset = code_index_in_word * bits;
                    let weights_words = expert_b as *const u32;
                    let word = unsafe { weights_words.add(word_index).read_unaligned() };
                    let code_mask = (1u32 << bits) - 1;
                    let mut weight_code = ((word >> bit_offset) & code_mask) as u8;
                    if signed_codes {
                        weight_code ^= 1u8 << (bits - 1);
                    }
                    let quantized_value = f32::from(weight_code);
                    let group_index = inner / group_size;
                    let metadata_index = col * scale_output_stride as usize + group_index * scale_group_stride as usize;
                    let scale = unsafe { *expert_scales.unwrap().add(metadata_index) }.to_f32().unwrap();
                    let midpoint = (1u32 << (bits - 1)) as f32;
                    let bias_term = match b_prologue {
                        GemmBPrologueKind::ScaleZeroPointDequant => {
                            let zp_ptr = expert_zero_points.unwrap();
                            let zp_index = col * zero_point_output_stride as usize
                                + group_index * zero_point_group_stride as usize;
                            let zp = if bits == 4 {
                                let byte_index = zp_index / 2;
                                let byte_value = unsafe { *zp_ptr.add(byte_index) };
                                if zp_index.is_multiple_of(2) {
                                    (byte_value & 0x0F) as f32
                                } else {
                                    ((byte_value >> 4) & 0x0F) as f32
                                }
                            } else {
                                unsafe { *zp_ptr.add(zp_index) as f32 }
                            };
                            -scale * zp
                        },
                        GemmBPrologueKind::ScaleBiasDequant => {
                            unsafe { *expert_biases.unwrap().add(metadata_index) }.to_f32().unwrap()
                        },
                        GemmBPrologueKind::ScaleSymmetricDequant => -scale * midpoint,
                        GemmBPrologueKind::FullPrecision => unreachable!(),
                    };
                    scale * quantized_value + bias_term
                };
                acc += a_val * b_val;
            }
            *out_slot = acc;
        }

        if output_transform.contains(GemmDTransform::RHT) {
            let factors = expert_hadamard.expect("RHT requires hadamard_factors");
            let block_size = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;
            for block_start in (0..n_u).step_by(block_size) {
                let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize];
                for lane in 0..block_size {
                    stripe[lane] = T::from(row_out[block_start + lane]).unwrap().to_f32().unwrap();
                }
                hadamard_transform(&mut stripe);
                for lane in 0..block_size {
                    let sign = unsafe { *factors.add(block_start + lane) } as f32;
                    let transformed = T::from(stripe[lane] * sign).unwrap().to_f32().unwrap();
                    row_out[block_start + lane] = transformed;
                }
            }
        }

        if output_transform.contains(GemmDTransform::BIAS) {
            let bias_ptr = expert_output_bias.expect("BIAS requires output_bias");
            for (col, val) in row_out.iter_mut().enumerate() {
                let b_val = unsafe { *bias_ptr.add(col) }.to_f32().unwrap();
                *val += b_val;
            }
        }

        for (col, &val) in row_out.iter().enumerate() {
            unsafe {
                *route_d.add(col) = T::from(val).unwrap();
            }
        }
    }
}
