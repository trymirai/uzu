use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::{kv_layout, qkv_norm::QKVNormParams, trie::TrieNode},
        cpu::kernel::{
            activation_transform::{INT8_SYMMETRIC_QUANTIZATION_MAXIMUM as INT8_MAX, hadamard_transform},
            attention::rope::apply_rope,
        },
    },
    data_type::DataTypeRepr,
};

#[kernel(AttentionPrepare)]
#[variants(ElementT, bf16, f32)]
#[variants(RopeT, f32)]
#[variants(HEAD_DIM, 64, 128, 256, 512)]
pub fn attention_prepare<ElementT: DataTypeRepr + Float, RopeT: DataTypeRepr + Float, const HEAD_DIM: u32>(
    qkvg: *const ElementT,
    q_codes: *mut i8,
    q_scales: *mut bf16,
    #[optional(append_kv)] k_codes: Option<*mut i8>,
    #[optional(append_kv)] k_scales: Option<*mut bf16>,
    #[optional(append_kv)] v_codes: Option<*mut u8>,
    #[optional(append_kv)] v_scales: Option<*mut bf16>,
    #[optional(has_rope)] cosines: Option<*const RopeT>,
    #[optional(has_rope)] sines: Option<*const RopeT>,
    #[optional(has_query_norm_scales)] query_norm_scales: Option<*const f32>,
    #[optional(has_key_norm_scales)] key_norm_scales: Option<*const f32>,
    #[optional(has_value_norm_scales)] value_norm_scales: Option<*const f32>,
    norm_params: QKVNormParams,
    num_q_heads: u32,
    #[optional(has_rope)] rope_dim: Option<u32>,
    kv_token_offset: u32,
    input_row_stride: u32,
    suffix_length: u32,
    query_rows: u32,
    #[optional(has_trie)] trie: Option<*const TrieNode>,
    #[specialize] has_rope: bool,
    #[specialize] num_kv_heads: u32,
    #[specialize] backend_opaque_layout: u32,
    #[specialize] append_kv: bool,
    #[specialize] has_query_norm_scales: bool,
    #[specialize] has_key_norm_scales: bool,
    #[specialize] has_value_norm_scales: bool,
    #[specialize] has_trie: bool,
) {
    let head_dim = HEAD_DIM as usize;
    let rope_dim = if has_rope {
        rope_dim.unwrap() as usize
    } else {
        0
    };
    assert_eq!(backend_opaque_layout, 0);
    let mut values = vec![0.0f32; head_dim];
    let mut normalized = if norm_params.enabled.iter().any(|&enabled| enabled != 0) {
        vec![ElementT::zero(); head_dim]
    } else {
        Vec::new()
    };
    let gqa_factor = num_q_heads / num_kv_heads;
    let v_head_start = num_q_heads + num_kv_heads;
    for head in 0..v_head_start + num_kv_heads {
        let is_query = head < num_q_heads;
        if !append_kv && !is_query {
            continue;
        }
        let is_key = !is_query && head < v_head_start;
        let (kv_head, scales) = match (is_query, is_key) {
            (true, _) => (0, Some(q_scales)),
            (_, true) => (head - num_q_heads, k_scales),
            _ => (head - v_head_start, v_scales),
        };
        let (norm, norm_scales) = if is_query {
            (0, has_query_norm_scales.then(|| query_norm_scales.unwrap()))
        } else if is_key {
            (1, has_key_norm_scales.then(|| key_norm_scales.unwrap()))
        } else {
            (2, has_value_norm_scales.then(|| value_norm_scales.unwrap()))
        };
        for token in 0..suffix_length {
            let rope_row = if has_trie {
                unsafe { (*trie.unwrap().add(token as usize)).height as usize }
            } else {
                token as usize
            };
            let mut source = unsafe { qkvg.add((token * input_row_stride + head * HEAD_DIM) as usize) };
            if norm_params.enabled[norm] != 0 {
                let input = unsafe { std::slice::from_raw_parts(source, head_dim) };
                let sum: f32 = input.iter().map(|v| v.to_f32().unwrap().powi(2)).sum();
                let inverse_rms = (sum / head_dim as f32 + norm_params.epsilon[norm]).sqrt().recip();
                for (channel, output) in normalized.iter_mut().enumerate() {
                    let value = input[channel].to_f32().unwrap() * inverse_rms;
                    *output = if let Some(scales) = norm_scales {
                        let scale = unsafe { *scales.add(channel) } + norm_params.scale_offset[norm];
                        if norm_params.full_layer[norm] != 0 {
                            ElementT::from(value * scale).unwrap()
                        } else {
                            ElementT::from(value).unwrap() * ElementT::from(scale).unwrap()
                        }
                    } else {
                        ElementT::from(value).unwrap()
                    };
                }
                source = normalized.as_ptr();
            }
            for (channel, value) in values.iter_mut().enumerate() {
                let element = if (is_query || is_key) && channel < rope_dim {
                    apply_rope(source, cosines.unwrap(), sines.unwrap(), rope_row, channel, rope_dim)
                } else {
                    unsafe { *source.add(channel) }
                }
                .to_f32()
                .unwrap();
                let sign_word = kv_layout::ROTATION_SIGN_BITS[channel / kv_layout::WORD_BITS as usize % 8];
                let flip_sign = (sign_word >> (channel % kv_layout::WORD_BITS as usize)) & 1 != 0;
                *value = if flip_sign {
                    -element
                } else {
                    element
                };
            }
            hadamard_transform(&mut values);
            let maximum = values.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            let scale = bf16::from_f32(if maximum > 0.0 {
                maximum / INT8_MAX
            } else {
                1.0
            });
            let scale_index = if is_query {
                ((head / gqa_factor) * query_rows + (head % gqa_factor) * suffix_length + token) as usize
            } else {
                ((kv_token_offset + token) * num_kv_heads + kv_head) as usize
            };
            for (channel, &value) in values.iter().enumerate() {
                let code = (value / scale.to_f32()).round_ties_even().clamp(-INT8_MAX, INT8_MAX) as i8;
                unsafe {
                    if is_query {
                        *q_codes.add(scale_index * head_dim + channel) = code;
                    } else if is_key {
                        *k_codes.unwrap().add(scale_index * head_dim + channel) = code;
                    } else {
                        *v_codes.unwrap().add(scale_index * head_dim + channel) =
                            (i16::from(code) + kv_layout::V_CODE_BIAS) as u8;
                    }
                }
            }
            unsafe { *scales.unwrap().add(scale_index) = scale };
        }
    }
}
