use num_traits::Float;

use crate::{backends::common::gpu_types::gemm::GemmBPrologueKind, data_type::DataTypeRepr};

pub(super) fn compute_route_gemv<T: DataTypeRepr + Float, const GROUP_SIZE: u32, const BITS: u32>(
    expert_b: *const u8,
    expert_scales: Option<*const T>,
    expert_zero_points: Option<*const u8>,
    expert_biases: Option<*const T>,
    route_a: *const T,
    k_u: usize,
    n_u: usize,
    scale_output_stride: usize,
    scale_group_stride: usize,
    zero_point_output_stride: usize,
    zero_point_group_stride: usize,
    signed_codes: bool,
    b_prologue: GemmBPrologueKind,
) -> Vec<f32> {
    let bits = BITS as usize;
    let group_size = if GROUP_SIZE == 0 {
        1
    } else {
        GROUP_SIZE as usize
    };
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
                let bit_offset = (code_index_in_word * bits) as u32;
                let weights_words = expert_b as *const u32;
                let word = unsafe { weights_words.add(word_index).read_unaligned() };
                let code_mask = (1u32 << bits) - 1;
                let mut weight_code = ((word >> bit_offset) & code_mask) as u8;
                if signed_codes {
                    weight_code ^= 1u8 << (bits - 1);
                }
                let quantized_value = f32::from(weight_code);
                let group_index = inner / group_size;
                let metadata_index = col * scale_output_stride + group_index * scale_group_stride;
                let scale = unsafe { *expert_scales.unwrap().add(metadata_index) }.to_f32().unwrap();
                let midpoint = (1u32 << (bits - 1)) as f32;
                let bias_term = match b_prologue {
                    GemmBPrologueKind::ScaleZeroPointDequant => {
                        let zp_ptr = expert_zero_points.unwrap();
                        let zp_index = col * zero_point_output_stride + group_index * zero_point_group_stride;
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
    row_out
}
