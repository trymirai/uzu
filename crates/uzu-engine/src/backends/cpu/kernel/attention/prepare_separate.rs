use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::{
        common::gpu_types::{attention_prepare::SeparatePrepareParams, kv_layout, trie::TrieNode},
        cpu::kernel::attention::compute::quantize,
    },
    data_type::DataTypeRepr,
};

#[kernel(AttentionPrepareSeparate)]
#[variants(ElementT, bf16, f32)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_prepare_separate<ElementT: DataTypeRepr + Float, const HEAD_DIM: u32>(
    queries: *const ElementT,
    keys: *const ElementT,
    values: *const ElementT,
    #[optional(has_key_conv)] key_conv_weights: Option<*const f32>,
    #[optional(has_value_conv)] value_conv_weights: Option<*const f32>,
    #[optional(has_key_conv)] key_conv_state: Option<*mut f32>,
    #[optional(has_value_conv)] value_conv_state: Option<*mut f32>,
    #[optional(has_key_conv && has_trie)] key_node_inputs: Option<*mut f32>,
    #[optional(has_value_conv && has_trie)] value_node_inputs: Option<*mut f32>,
    #[optional(has_query_clip_scales)] query_clip_scales: Option<*const f32>,
    #[optional(has_relative_bias)] relative_params: Option<*const f32>,
    #[optional(has_relative_bias)] relative_logits: Option<*mut bf16>,
    q_codes: *mut i8,
    q_scales: *mut bf16,
    k_codes: *mut i8,
    k_scales: *mut bf16,
    v_codes: *mut u8,
    v_scales: *mut bf16,
    #[optional(has_trie)] trie: Option<*const TrieNode>,
    params: SeparatePrepareParams,
    suffix_length: u32,
    #[specialize] num_q_heads: u32,
    #[specialize] num_kv_heads: u32,
    #[specialize] backend_opaque_layout: u32,
    #[specialize] kernel_size: u32,
    #[specialize] has_key_conv: bool,
    #[specialize] has_value_conv: bool,
    #[specialize] has_conv_taps: bool,
    #[specialize] has_conv_identity: bool,
    #[specialize] updates_conv_state: bool,
    #[specialize] has_query_clip: bool,
    #[specialize] has_query_clip_scales: bool,
    #[specialize] has_key_clip: bool,
    #[specialize] normalizes_values: bool,
    #[specialize] has_log_scaling: bool,
    #[specialize] has_relative_bias: bool,
    #[specialize] relative_dim: u32,
    #[specialize] relative_extent: u32,
    #[specialize] has_trie: bool,
) {
    assert_eq!(backend_opaque_layout, 0);
    let head_dim = HEAD_DIM as usize;
    let (num_q_heads, num_kv_heads) = (num_q_heads as usize, num_kv_heads as usize);
    let (kernel_size, history) = (kernel_size as usize, kernel_size as usize - 1);
    let kv_dim = num_kv_heads * head_dim;
    let suffix = suffix_length as usize;
    let gqa_factor = num_q_heads / num_kv_heads;
    let clip = |values: &mut [f32], threshold: f32| {
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        let factor = (threshold / (norm + 1e-6)).min(1.0);
        values.iter_mut().for_each(|value| *value *= factor);
    };
    let node = |token: usize| unsafe { *trie.unwrap().add(token) };
    let height = |token: usize| if has_trie { node(token).height as usize } else { token };
    let parent = |token: isize| -> isize {
        if token < 0 {
            token
        } else if has_trie {
            node(token as usize).parent as isize
        } else {
            token - 1
        }
    };
    // The conv states are read and written per channel, so updating in place keeps the semantics of one token.
    let mut key_history = key_conv_state
        .map(|state| unsafe { std::slice::from_raw_parts(state, kv_dim * history).to_vec() })
        .unwrap_or_default();
    let mut value_history = value_conv_state
        .map(|state| unsafe { std::slice::from_raw_parts(state, kv_dim * history).to_vec() })
        .unwrap_or_default();
    for token in 0..suffix {
        for head in 0..num_q_heads + 2 * num_kv_heads {
            let is_query = head < num_q_heads;
            let is_key = !is_query && head < num_q_heads + num_kv_heads;
            let kv_head = if is_key {
                head - num_q_heads
            } else {
                head.wrapping_sub(num_q_heads + num_kv_heads)
            };
            let mut values_out;
            if is_query {
                let row = unsafe { queries.add(token * params.query_row_stride as usize + head * head_dim) };
                values_out = (0..head_dim)
                    .map(|channel| {
                        let value = unsafe { *row.add(channel) }.to_f32().unwrap();
                        if has_query_clip_scales {
                            value * unsafe { *query_clip_scales.unwrap().add(head * head_dim + channel) }
                        } else {
                            value
                        }
                    })
                    .collect::<Vec<f32>>();
                if has_query_clip {
                    clip(&mut values_out, params.query_clip_threshold);
                }
                let mut tau = 1.0;
                if has_log_scaling {
                    let ratio =
                        (params.position_offset as usize + height(token) + 1) as f32 / params.log_scaling_n_floor;
                    tau = 1.0 + params.log_scaling_alpha * ratio.max(1.0).ln();
                    values_out.iter_mut().for_each(|value| *value *= tau);
                }
                if has_relative_bias {
                    let (dim, extent) = (relative_dim as usize, relative_extent as usize);
                    let clip_scales = relative_params.unwrap();
                    let profiles = unsafe { clip_scales.add(num_q_heads * dim) };
                    let states = unsafe {
                        queries.add(
                            token * params.query_row_stride as usize + params.relative_offset as usize + head * dim,
                        )
                    };
                    let mut relative = (0..dim)
                        .map(|channel| unsafe {
                            (*states.add(channel)).to_f32().unwrap() * *clip_scales.add(head * dim + channel)
                        })
                        .collect::<Vec<f32>>();
                    clip(&mut relative, params.relative_clip_threshold);
                    relative.iter_mut().for_each(|value| *value *= tau);
                    for distance in 0..extent {
                        let sum = (0..dim)
                            .map(|channel| relative[channel] * unsafe { *profiles.add(channel * extent + distance) })
                            .sum::<f32>();
                        unsafe {
                            *relative_logits.unwrap().add((head * suffix + token) * extent + distance) = bf16::from_f32(sum)
                        };
                    }
                }
            } else {
                let (rows, row_stride) = if is_key {
                    (keys, params.key_row_stride as usize)
                } else {
                    (values, params.value_row_stride as usize)
                };
                let input = |row: usize, channel: usize| unsafe { (*rows.add(row * row_stride + channel)).to_f32().unwrap() };
                values_out = (0..head_dim).map(|channel| input(token, kv_head * head_dim + channel)).collect::<Vec<f32>>();
                let has_conv = if is_key {
                    has_key_conv
                } else {
                    has_value_conv
                };
                if has_conv {
                    let (weights, state, node_inputs, scale) = if is_key {
                        (key_conv_weights.unwrap(), &mut key_history, key_node_inputs, params.key_conv_scale)
                    } else {
                        (value_conv_weights.unwrap(), &mut value_history, value_node_inputs, params.value_conv_scale)
                    };
                    let mut sources = vec![0isize; kernel_size];
                    let mut source = token as isize;
                    for back in 1..kernel_size {
                        source = parent(source);
                        sources[back] = source;
                    }
                    let taps = (0..kernel_size)
                        .map(|tap| if has_conv_taps { input(token, kv_dim + kv_head * kernel_size + tap) } else { 0.0 })
                        .collect::<Vec<f32>>();
                    for (index, value) in values_out.iter_mut().enumerate() {
                        let channel = kv_head * head_dim + index;
                        let mut window = vec![0.0f32; kernel_size];
                        window[history] = *value;
                        for back in 1..kernel_size {
                            window[history - back] = if sources[back] >= 0 {
                                input(sources[back] as usize, channel)
                            } else {
                                state[channel * history + history + height(token) - back]
                            };
                        }
                        let mut accumulator = if has_conv_identity {
                            window[history]
                        } else {
                            0.0
                        };
                        for tap in 0..kernel_size {
                            accumulator += (unsafe { *weights.add(channel * kernel_size + tap) } + taps[tap]) * window[tap];
                        }
                        *value = accumulator * scale;
                        if has_trie {
                            for index in 0..history {
                                unsafe {
                                    *node_inputs.unwrap().add((token * kv_dim + channel) * history + index) = window[index + 1]
                                };
                            }
                        }
                        if updates_conv_state {
                            state[channel * history..(channel + 1) * history].copy_from_slice(&window[1..]);
                        }
                    }
                }
                if is_key && has_key_clip {
                    clip(&mut values_out, params.key_clip_threshold);
                }
                if !is_key && normalizes_values {
                    let mean_square = values_out.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
                    let inverse_rms = (mean_square + params.value_norm_epsilon).sqrt().recip();
                    values_out.iter_mut().for_each(|value| *value *= inverse_rms);
                }
            }
            let (codes, scale) = quantize(values_out);
            let scale_index = if is_query {
                (head / gqa_factor) * params.query_rows as usize + (head % gqa_factor) * suffix + token
            } else {
                (params.kv_token_offset as usize + token) * num_kv_heads + kv_head
            };
            unsafe {
                for (channel, &code) in codes.iter().enumerate() {
                    if is_query {
                        *q_codes.add(scale_index * head_dim + channel) = code;
                    } else if is_key {
                        *k_codes.add(scale_index * head_dim + channel) = code;
                    } else {
                        *v_codes.add(scale_index * head_dim + channel) = (i16::from(code) + kv_layout::V_CODE_BIAS) as u8;
                    }
                }
                *(if is_query {
                    q_scales
                } else if is_key {
                    k_scales
                } else {
                    v_scales
                })
                .add(scale_index) = scale;
            }
        }
    }
    // Updates happen once all tokens read the state: a batch that updates in place holds a single token.
    if updates_conv_state {
        for (state, history_values) in [(key_conv_state, &key_history), (value_conv_state, &value_history)] {
            if let Some(state) = state {
                unsafe { std::slice::from_raw_parts_mut(state, kv_dim * history) }.copy_from_slice(history_values);
            }
        }
    }
}
