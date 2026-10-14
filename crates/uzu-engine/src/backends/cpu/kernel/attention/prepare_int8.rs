use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use super::attention_prepare::apply_rope;
use crate::{
    backends::{
        common::{gpu_types::kv_layout, kernel::kv_layout::Int8KVLayout},
        cpu::kernel::activation_transform::{INT8_SYMMETRIC_QUANTIZATION_MAXIMUM as INT8_MAX, hadamard_transform},
    },
    data_type::DataTypeRepr,
};

#[kernel(AttentionPrepareInt8)]
#[variants(ElementT, bf16)]
#[variants(RopeT, f32)]
#[variants(HEAD_DIM, 256)]
pub fn attention_prepare_int8<ElementT: DataTypeRepr + Float, RopeT: DataTypeRepr + Float, const HEAD_DIM: u32>(
    qkvg: *const ElementT,
    #[optional(!quantize_queries)] queries: Option<*mut ElementT>,
    #[optional(quantize_queries)] q_codes: Option<*mut i8>,
    #[optional(quantize_queries)] q_scales: Option<*mut ElementT>,
    #[optional(append_kv)] k_codes: Option<*mut i8>,
    #[optional(append_kv)] k_scales: Option<*mut ElementT>,
    #[optional(append_kv)] v_codes: Option<*mut u8>,
    #[optional(append_kv)] v_scales: Option<*mut ElementT>,
    #[optional(has_rope)] cosines: Option<*const RopeT>,
    #[optional(has_rope)] sines: Option<*const RopeT>,
    num_q_heads: u32,
    #[optional(has_rope)] rope_dim: Option<u32>,
    kv_token_offset: u32,
    input_row_stride: u32,
    suffix_length: u32,
    query_rows: u32,
    #[specialize] has_rope: bool,
    #[specialize] quantize_queries: bool,
    #[specialize] num_kv_heads: u32,
    #[specialize] append_kv: bool,
) {
    let head_dim = HEAD_DIM as usize;
    let rope_dim = if has_rope {
        rope_dim.unwrap() as usize
    } else {
        0
    };
    let layout = Int8KVLayout::new(HEAD_DIM, num_kv_heads);
    let mut values = vec![0.0f32; head_dim];
    let gqa_factor = num_q_heads / num_kv_heads;
    let v_head_start = num_q_heads + num_kv_heads;
    for head in 0..v_head_start + num_kv_heads {
        let is_query = head < num_q_heads;
        if !append_kv && !is_query {
            continue;
        }
        let is_key = !is_query && head < v_head_start;
        let quantize = !is_query || quantize_queries;
        let (kv_head, scales) = match (is_query, is_key) {
            (true, _) => (0, q_scales),
            (_, true) => (head - num_q_heads, k_scales),
            _ => (head - v_head_start, v_scales),
        };
        for token in 0..suffix_length {
            let query_row = ((head / gqa_factor) * query_rows + (head % gqa_factor) * suffix_length + token) as usize;
            let source = unsafe { qkvg.add((token * input_row_stride + head * HEAD_DIM) as usize) };
            for (channel, value) in values.iter_mut().enumerate() {
                let element = if (is_query || is_key) && channel < rope_dim {
                    apply_rope(source, cosines.unwrap(), sines.unwrap(), token as usize, channel, rope_dim)
                } else {
                    unsafe { *source.add(channel) }
                }
                .to_f32()
                .unwrap();
                if !quantize {
                    let row = (head * suffix_length + token) as usize * head_dim;
                    unsafe { *queries.unwrap().add(row + channel) = ElementT::from(element).unwrap() };
                } else {
                    let sign_word = kv_layout::ROTATION_SIGN_BITS[channel / kv_layout::WORD_BITS as usize];
                    let flip_sign = (sign_word >> (channel % kv_layout::WORD_BITS as usize)) & 1 != 0;
                    *value = if flip_sign {
                        -element
                    } else {
                        element
                    };
                }
            }
            if !quantize {
                continue;
            }
            hadamard_transform(&mut values);
            let maximum = values.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            let scale = ElementT::from(if maximum > 0.0 {
                maximum / INT8_MAX
            } else {
                1.0
            })
            .unwrap();
            let scale_index = if is_query {
                query_row
            } else {
                layout.scale_index(kv_token_offset + token, kv_head)
            };
            for (channel, &value) in values.iter().enumerate() {
                let code = (value / scale.to_f32().unwrap()).round_ties_even().clamp(-INT8_MAX, INT8_MAX) as i8;
                unsafe {
                    if is_query {
                        *q_codes.unwrap().add(scale_index * head_dim + channel) = code;
                    } else if is_key {
                        *k_codes.unwrap().add(layout.k_code_index(kv_token_offset + token, kv_head, channel as u32)) =
                            code;
                    } else {
                        *v_codes.unwrap().add(layout.v_code_index(kv_token_offset + token, kv_head, channel as u32)) =
                            (i16::from(code) + kv_layout::V_CODE_BIAS) as u8;
                    }
                }
            }
            unsafe { *scales.unwrap().add(scale_index) = scale };
        }
    }
}
