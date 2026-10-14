use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use super::attention_prepare::apply_rope;
use crate::{
    array::ArrayElement,
    backends::{
        common::kernel::kv_layout::{Int8KVLayout, KV_ROTATION_SIGN_BITS},
        cpu::kernel::activation_transform::hadamard_transform,
    },
};

#[kernel(AttentionPrepareInt8)]
#[variants(ElementT, bf16)]
#[variants(RopeT, f32)]
#[variants(HEAD_DIM, 256)]
pub fn attention_prepare_int8<ElementT: ArrayElement + Float, RopeT: ArrayElement + Float, const HEAD_DIM: u32>(
    qkvg: *const ElementT,
    #[optional(!quantize_queries)] queries: Option<*mut ElementT>,
    #[optional(quantize_queries)] q_codes: Option<*mut i8>,
    #[optional(quantize_queries)] q_scales: Option<*mut ElementT>,
    k_codes: *mut i8,
    k_scales: *mut ElementT,
    v_codes: *mut u8,
    v_scales: *mut ElementT,
    #[optional(has_rope)] cosines: Option<*const RopeT>,
    #[optional(has_rope)] sines: Option<*const RopeT>,
    num_q_heads: u32,
    #[optional(has_rope)] rope_dim: Option<u32>,
    kv_token_offset: u32,
    input_row_stride: u32,
    suffix_length: u32,
    padded_suffix: u32,
    #[specialize] has_rope: bool,
    #[specialize] quantize_queries: bool,
    #[specialize] num_kv_heads: u32,
) {
    let head_dim = HEAD_DIM as usize;
    let layout = Int8KVLayout::new(HEAD_DIM, num_kv_heads, 0);
    let mut values = vec![0.0f32; head_dim];
    for head in 0..num_q_heads + 2 * num_kv_heads {
        let is_query = head < num_q_heads;
        let is_key = head >= num_q_heads && head < num_q_heads + num_kv_heads;
        let kv_head = if is_query {
            0
        } else if is_key {
            head - num_q_heads
        } else {
            head - num_q_heads - num_kv_heads
        };
        for token in 0..padded_suffix {
            let query_row = (head * padded_suffix + token) as usize;
            if token >= suffix_length {
                if is_query && quantize_queries {
                    unsafe {
                        std::ptr::write_bytes(q_codes.unwrap().add(query_row * head_dim), 0, head_dim);
                        *q_scales.unwrap().add(query_row) = ElementT::from(1.0).unwrap();
                    }
                }
                continue;
            }
            let source = unsafe { qkvg.add((token * input_row_stride + head * HEAD_DIM) as usize) };
            for (channel, value) in values.iter_mut().enumerate() {
                let element = if has_rope && (is_query || is_key) && channel < rope_dim.unwrap() as usize {
                    apply_rope(
                        source,
                        cosines.unwrap(),
                        sines.unwrap(),
                        token as usize,
                        channel,
                        rope_dim.unwrap() as usize,
                    )
                } else {
                    unsafe { *source.add(channel) }
                };
                *value = element.to_f32().unwrap();
            }
            if is_query && !quantize_queries {
                let row = (head * suffix_length + token) as usize * head_dim;
                for (channel, &value) in values.iter().enumerate() {
                    unsafe { *queries.unwrap().add(row + channel) = ElementT::from(value).unwrap() };
                }
                continue;
            }
            for (channel, value) in values.iter_mut().enumerate() {
                if KV_ROTATION_SIGN_BITS[channel / 32] >> (channel % 32) & 1 != 0 {
                    *value = -*value;
                }
            }
            hadamard_transform(&mut values);
            let maximum = values.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            let scale = ElementT::from(if maximum > 0.0 {
                maximum / 127.0
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
                let code = (value / scale.to_f32().unwrap()).round_ties_even().clamp(-127.0, 127.0) as i8;
                unsafe {
                    if is_query {
                        *q_codes.unwrap().add(scale_index * head_dim + channel) = code;
                    } else if is_key {
                        *k_codes.add(layout.k_code_index(kv_token_offset + token, kv_head, channel as u32)) = code;
                    } else {
                        *v_codes.add(layout.v_code_index(kv_token_offset + token, kv_head, channel as u32)) =
                            (i16::from(code) + 128) as u8;
                    }
                }
            }
            unsafe {
                if is_query {
                    *q_scales.unwrap().add(scale_index) = scale;
                } else if is_key {
                    *k_scales.add(scale_index) = scale;
                } else {
                    *v_scales.add(scale_index) = scale;
                }
            }
        }
    }
}
