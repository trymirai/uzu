use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{backends::common::gpu_types::trie::TrieNode, data_type::DataTypeRepr};

#[kernel(NGramGate)]
#[variants(T, f32, f16, bf16)]
pub fn ngram_gate<T: DataTypeRepr + Float>(
    keys: *const T,
    stream: *const T,
    values: *const T,
    key_scales: *const f32,
    query_scales: *const f32,
    conv_scales: *const f32,
    gated: *mut T,
    conv_input: *mut T,
    num_tokens: u32,
    stream_count: u32,
    model_dim: u32,
    epsilon: f32,
    scale_offset: f32,
) {
    let stream_count = stream_count as usize;
    let model_dim = model_dim as usize;
    let dim = model_dim as f32;

    for row in 0..num_tokens as usize * stream_count {
        let token = row / stream_count;
        let stream_index = row % stream_count;
        let row_offset = row * model_dim;
        let value_offset = token * model_dim;
        let scale_offset_index = stream_index * model_dim;

        let mut key_square_sum = 0.0f32;
        let mut query_square_sum = 0.0f32;
        let mut scaled_dot = 0.0f32;
        let mut value_square_sum = 0.0f32;
        for i in 0..model_dim {
            unsafe {
                let key = (*keys.add(row_offset + i)).to_f32().unwrap();
                let query = (*stream.add(row_offset + i)).to_f32().unwrap();
                let value = (*values.add(value_offset + i)).to_f32().unwrap();
                let key_scale = *key_scales.add(scale_offset_index + i) + scale_offset;
                let query_scale = *query_scales.add(scale_offset_index + i) + scale_offset;
                key_square_sum += key * key;
                query_square_sum += query * query;
                scaled_dot += key * key_scale * query * query_scale;
                value_square_sum += value * value;
            }
        }

        let key_rms_inv = (key_square_sum / dim + epsilon).sqrt().recip();
        let query_rms_inv = (query_square_sum / dim + epsilon).sqrt().recip();
        let logit = key_rms_inv * query_rms_inv * scaled_dot / dim.sqrt();
        let signed_root = if logit == 0.0 {
            0.0
        } else {
            logit.signum() * logit.abs().max(1e-6).sqrt()
        };
        let gate = 1.0 / (1.0 + (-signed_root).exp());
        let conv_rms_inv = (gate * gate * value_square_sum / dim + epsilon).sqrt().recip();

        for i in 0..model_dim {
            unsafe {
                let gated_value = gate * (*values.add(value_offset + i)).to_f32().unwrap();
                let conv_scale = *conv_scales.add(scale_offset_index + i) + scale_offset;
                *gated.add(row_offset + i) = T::from(gated_value).unwrap();
                *conv_input.add(row_offset + i) = T::from(gated_value * conv_rms_inv * conv_scale).unwrap();
            }
        }
    }
}

#[kernel(NGramConvInject)]
#[variants(T, f32, f16, bf16)]
pub fn ngram_conv_inject<T: DataTypeRepr + Float>(
    conv_input: *const T,
    state: *const T,
    weights: *const f32,
    gated: *const T,
    stream: *mut T,
    next_state: *mut T,
    num_tokens: u32,
    channels: u32,
    kernel_size: u32,
) {
    let num_tokens = num_tokens as usize;
    let channels = channels as usize;
    let kernel_size = kernel_size as usize;
    let state_rows = kernel_size - 1;

    let sample = |row: usize, channel: usize| -> f32 {
        unsafe {
            match row < state_rows {
                true => (*state.add(row * channels + channel)).to_f32().unwrap(),
                false => (*conv_input.add((row - state_rows) * channels + channel)).to_f32().unwrap(),
            }
        }
    };

    for channel in 0..channels {
        for token in 0..num_tokens {
            let mut accumulator = 0.0f32;
            for tap in 0..kernel_size {
                accumulator += unsafe { *weights.add(channel * kernel_size + tap) } * sample(token + tap, channel);
            }
            let activated = accumulator / (1.0 + (-accumulator).exp());
            let index = token * channels + channel;
            unsafe {
                let updated =
                    (*stream.add(index)).to_f32().unwrap() + (*gated.add(index)).to_f32().unwrap() + activated;
                *stream.add(index) = T::from(updated).unwrap();
            }
        }
        for row in 0..state_rows {
            unsafe { *next_state.add(row * channels + channel) = T::from(sample(num_tokens + row, channel)).unwrap() };
        }
    }
}

// Note that the most natural layout for `state` (which is used here) is not the
// same as in short-conv. relevant for when we unify them.
#[kernel(NGramConvInjectTrie)]
#[variants(T, f32, f16, bf16)]
pub fn ngram_conv_inject_trie<T: DataTypeRepr + Float>(
    conv_input: *const T,
    state: *const T,
    trie: *const TrieNode,
    weights: *const f32,
    gated: *const T,
    stream: *mut T,
    node_states: *mut T,
    num_tokens: u32,
    channels: u32,
    kernel_size: u32,
) {
    let num_tokens = num_tokens as usize;
    let channels = channels as usize;
    let kernel_size = kernel_size as usize;
    let state_rows = kernel_size - 1;

    // Nodes come in DFS order, so a parent's slot is complete before any child reads it.
    for channel in 0..channels {
        for node in 0..num_tokens {
            let parent = unsafe { (*trie.add(node)).parent };
            let parent_state = if parent < 0 {
                state
            } else {
                unsafe { node_states.add(parent as usize * state_rows * channels) as *const T }
            };
            let parent_row =
                |row: usize| -> f32 { unsafe { (*parent_state.add(row * channels + channel)).to_f32().unwrap() } };
            let own = unsafe { (*conv_input.add(node * channels + channel)).to_f32().unwrap() };

            let weight = |tap: usize| -> f32 { unsafe { *weights.add(channel * kernel_size + tap) } };
            let mut accumulator = weight(state_rows) * own;
            for tap in 0..state_rows {
                accumulator += weight(tap) * parent_row(tap);
            }
            let activated = accumulator / (1.0 + (-accumulator).exp());
            let index = node * channels + channel;
            unsafe {
                let updated =
                    (*stream.add(index)).to_f32().unwrap() + (*gated.add(index)).to_f32().unwrap() + activated;
                *stream.add(index) = T::from(updated).unwrap();
            }

            let own_state = unsafe { node_states.add(node * state_rows * channels) };
            for row in 0..state_rows {
                let value = if row + 1 < state_rows {
                    parent_row(row + 1)
                } else {
                    own
                };
                unsafe { *own_state.add(row * channels + channel) = T::from(value).unwrap() };
            }
        }
    }
}
