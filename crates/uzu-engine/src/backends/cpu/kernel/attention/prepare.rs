use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::{kv_layout, qkv_norm::QKVNormParams, rope::RoPEGridParams, trie::TrieNode},
        cpu::kernel::{
            activation_transform::{INT8_SYMMETRIC_QUANTIZATION_MAXIMUM as INT8_MAX, hadamard_transform},
            attention::rope::{apply_rope, mrope_axis_offset},
        },
    },
    data_type::DataTypeRepr,
};

#[kernel(AttentionPrepare)]
#[variants(ElementT, bf16, f32)]
#[variants(HEAD_DIM, 64, 128, 256, 512)]
pub fn attention_prepare<ElementT: DataTypeRepr + Float, const HEAD_DIM: u32>(
    qkvg: *const ElementT,
    q_codes: *mut i8,
    q_scales: *mut bf16,
    #[optional(append_kv)] k_codes: Option<*mut i8>,
    #[optional(append_kv)] k_scales: Option<*mut bf16>,
    #[optional(append_kv)] v_codes: Option<*mut u8>,
    #[optional(append_kv)] v_scales: Option<*mut bf16>,
    #[optional(has_rope)] rope_inverse_frequencies: Option<*const f32>,
    #[optional(has_query_norm_scales)] query_norm_scales: Option<*const f32>,
    #[optional(has_key_norm_scales)] key_norm_scales: Option<*const f32>,
    #[optional(has_value_norm_scales)] value_norm_scales: Option<*const f32>,
    #[optional(block_mean)] pool_nodes: Option<*const TrieNode>,
    #[optional(block_mean)] pool_windows: Option<*const i32>,
    #[optional(block_mean)] raw_keys: Option<*mut ElementT>,
    norm_params: QKVNormParams,
    num_q_heads: u32,
    #[optional(has_rope)] rope_dim: Option<u32>,
    #[optional(has_rope)] rope_scale: Option<f32>,
    #[optional(has_rope)] rope_position_offset: Option<u32>,
    #[optional(has_mrope)] rope_grid: Option<RoPEGridParams>,
    kv_token_offset: u32,
    input_row_stride: u32,
    suffix_length: u32,
    query_rows: u32,
    #[optional(block_mean)] prefix_tokens: Option<u32>,
    #[optional(has_trie)] trie: Option<*const TrieNode>,
    #[specialize] has_rope: bool,
    #[specialize] has_mrope: bool,
    #[specialize] num_kv_heads: u32,
    #[specialize] backend_opaque_layout: u32,
    #[specialize] append_kv: bool,
    #[specialize] has_query_norm_scales: bool,
    #[specialize] has_key_norm_scales: bool,
    #[specialize] has_value_norm_scales: bool,
    #[specialize] block_mean: bool,
    #[specialize] compression_ratio: u32,
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
    let mut mean = vec![ElementT::zero(); head_dim * usize::from(block_mean)];
    let ratio = compression_ratio as usize;
    let prefix = prefix_tokens.unwrap_or(0) as usize;
    // Raw rows: the tokens after the last complete block, then one per suffix node.
    let block_start = prefix - prefix % ratio.max(1);
    let raw_row = |row: usize| unsafe { raw_keys.unwrap().add(row * head_dim) };
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
                unsafe { (*trie.unwrap().add(token as usize)).height }
            } else {
                token
            };
            let mut rope_position = rope_position_offset.unwrap_or(0) + rope_row;
            let (mut grid_row, mut grid_column) = (0, 0);
            if has_mrope {
                // Rows are cells of grids in raster order: a text token is a 1x1 grid, a trie node sits at its height
                let grid = rope_grid.unwrap();
                let cell = grid.first_cell + rope_row;
                let grid_size = grid.grid_width * grid.grid_height;
                rope_position =
                    rope_position_offset.unwrap() + cell / grid_size * grid.grid_width.max(grid.grid_height);
                grid_row = cell % grid_size / grid.grid_width;
                grid_column = cell % grid_size % grid.grid_width;
            }
            let mut source = unsafe { qkvg.add((token * input_row_stride + head * HEAD_DIM) as usize) };
            if block_mean {
                let token = token as usize;
                unsafe { std::ptr::copy_nonoverlapping(source, raw_row(ratio - 1 + token), head_dim) };
                values.fill(0.0);
                let height = unsafe { (*pool_nodes.unwrap().add(token)).height } as usize;
                if (prefix + height + 1).is_multiple_of(ratio) {
                    for i in 0..ratio {
                        let slot = unsafe { *pool_windows.unwrap().add(token * ratio + i) } as usize;
                        let row = match slot.checked_sub(prefix) {
                            Some(suffix_row) => unsafe { qkvg.add(suffix_row * input_row_stride as usize) },
                            None => raw_row(slot - block_start).cast_const(),
                        };
                        for (channel, sum) in values.iter_mut().enumerate() {
                            *sum += unsafe { *row.add(channel) }.to_f32().unwrap();
                        }
                    }
                }
                for (mean, sum) in mean.iter_mut().zip(&values) {
                    *mean = ElementT::from(sum / ratio as f32).unwrap();
                }
                source = mean.as_ptr();
            }
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
                    let mut position = rope_position;
                    if has_mrope {
                        let grid = rope_grid.unwrap();
                        position += mrope_axis_offset(
                            (channel % (rope_dim / 2)) as u32,
                            grid_row,
                            grid_column,
                            grid.height_pairs,
                            grid.width_pairs,
                        );
                    }
                    apply_rope::<ElementT, _>(
                        source,
                        rope_inverse_frequencies.unwrap(),
                        rope_scale.unwrap(),
                        position,
                        channel,
                        rope_dim,
                    )
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
