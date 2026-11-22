use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{backends::common::gpu_types::trie::TrieNode, data_type::DataTypeRepr};

#[kernel(CausalConv)]
#[variants(T, f32, f16, bf16)]
pub fn causal_conv<T: DataTypeRepr + Float>(
    input: *const T,
    #[optional(has_taps)] taps: Option<*const T>,
    weights: *const f32,
    state: *mut f32,
    #[optional(is_trie)] trie: Option<*const TrieNode>,
    #[optional(is_trie)] node_states: Option<*mut f32>,
    output: *mut T,
    token_count: u32,
    input_stride: u32,
    taps_stride: u32,
    output_scale: f32,
    accumulate: u32,
    #[specialize] channels: u32,
    #[specialize] kernel_size: u32,
    #[specialize] group_size: u32,
    #[specialize] has_identity: bool,
    #[specialize] has_taps: bool,
    #[specialize] is_trie: bool,
) {
    let token_count = token_count as usize;
    let input_stride = input_stride as usize;
    let taps_stride = taps_stride as usize;
    let channels = channels as usize;
    let kernel_size = kernel_size as usize;
    let group_size = group_size as usize;
    let history = kernel_size - 1;

    unsafe {
        for channel in 0..channels {
            let channel_weights = weights.add(channel * kernel_size);
            let tap_offset = (channel / group_size) * kernel_size;
            let channel_state = state.add(channel * history);
            let mut window = vec![0.0f32; kernel_size];
            if !is_trie {
                for tap in 0..history {
                    window[tap] = *channel_state.add(tap);
                }
            }

            for token in 0..token_count {
                if is_trie {
                    let parent = (*trie.unwrap().add(token)).parent;
                    let parent_inputs = if parent < 0 {
                        channel_state as *const f32
                    } else {
                        node_states.unwrap().add((parent as usize * channels + channel) * history) as *const f32
                    };
                    for tap in 0..history {
                        window[tap] = *parent_inputs.add(tap);
                    }
                }

                let value = (*input.add(token * input_stride + channel)).to_f32().unwrap();
                window[history] = value;
                let mut accumulator = 0.0f32;
                for tap in 0..kernel_size {
                    let mut weight = *channel_weights.add(tap);
                    if has_taps {
                        weight += (*taps.unwrap().add(token * taps_stride + tap_offset + tap)).to_f32().unwrap();
                    }
                    accumulator += weight * window[tap];
                }
                if has_identity {
                    accumulator += value;
                }
                let output = output.add(token * channels + channel);
                let base = if accumulate != 0 {
                    (*output).to_f32().unwrap()
                } else {
                    0.0
                };
                *output = T::from(base + accumulator * output_scale).unwrap();

                window.copy_within(1..kernel_size, 0);
                if is_trie {
                    let inputs_left = node_states.unwrap().add((token * channels + channel) * history);
                    for tap in 0..history {
                        *inputs_left.add(tap) = window[tap];
                    }
                }
            }

            if !is_trie {
                for tap in 0..history {
                    *channel_state.add(tap) = window[tap];
                }
            }
        }
    }
}

#[kernel(CausalConvTokens)]
#[variants(T, f32, f16, bf16)]
pub fn causal_conv_tokens<T: DataTypeRepr + Float>(
    input: *const T,
    #[optional(has_taps)] taps: Option<*const T>,
    weights: *const f32,
    state: *const f32,
    output: *mut T,
    token_count: u32,
    input_stride: u32,
    taps_stride: u32,
    output_scale: f32,
    accumulate: u32,
    #[specialize] channels: u32,
    #[specialize] kernel_size: u32,
    #[specialize] group_size: u32,
    #[specialize] has_identity: bool,
    #[specialize] has_taps: bool,
) {
    let (channels, kernel_size, group_size) = (channels as usize, kernel_size as usize, group_size as usize);
    let history = kernel_size - 1;
    unsafe {
        for token in 0..token_count as usize {
            for channel in 0..channels {
                let tap_offset = token * taps_stride as usize + (channel / group_size) * kernel_size;
                let mut accumulator = 0.0f32;
                let mut value = 0.0f32;
                for tap in 0..kernel_size {
                    value = match (token + tap).checked_sub(history) {
                        Some(source) => (*input.add(source * input_stride as usize + channel)).to_f32().unwrap(),
                        None => *state.add(channel * history + token + tap),
                    };
                    let mut weight = *weights.add(channel * kernel_size + tap);
                    if has_taps {
                        weight += (*taps.unwrap().add(tap_offset + tap)).to_f32().unwrap();
                    }
                    accumulator += weight * value;
                }
                if has_identity {
                    accumulator += value;
                }
                let output = output.add(token * channels + channel);
                let base = if accumulate != 0 {
                    (*output).to_f32().unwrap()
                } else {
                    0.0
                };
                *output = T::from(base + accumulator * output_scale).unwrap();
            }
        }
    }
}

#[kernel(CausalConvShift)]
#[variants(T, f32, f16, bf16)]
pub fn causal_conv_shift<T: DataTypeRepr + Float>(
    input: *const T,
    state: *mut f32,
    token_count: u32,
    input_stride: u32,
    #[specialize] channels: u32,
    #[specialize] kernel_size: u32,
) {
    let history = kernel_size as usize - 1;
    let token_count = token_count as usize;
    unsafe {
        for channel in 0..channels as usize {
            let channel_state = state.add(channel * history);
            let window = (0..history)
                .map(|index| match (token_count + index).checked_sub(history) {
                    Some(source) => (*input.add(source * input_stride as usize + channel)).to_f32().unwrap(),
                    None => *channel_state.add(token_count + index),
                })
                .collect::<Vec<f32>>();
            for (index, value) in window.into_iter().enumerate() {
                *channel_state.add(index) = value;
            }
        }
    }
}
