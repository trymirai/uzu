use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(NGramHash)]
pub fn ngram_hash(
    tokens_with_history: *const u32,
    multipliers: &[u64],
    head_vocab_sizes: &[u32],
    head_offsets: &[u32],
    ngram_ids: *mut u32,
    num_tokens: u32,
    ngram_size: u32,
    heads_per_ngram: u32,
    eos_token_id: u32,
) {
    let history_length = ngram_size as usize - 1;
    let num_heads = history_length * heads_per_ngram as usize;
    let eos_token_id = eos_token_id as u64;

    for token in 0..num_tokens as usize {
        let position = history_length + token;
        for head in 0..num_heads {
            let order = head / heads_per_ngram as usize + 2;
            unsafe {
                let mut hash = *tokens_with_history.add(position) as u64 * multipliers[0];
                let mut crossed_eos = false;
                for shift in 1..order {
                    let shifted_token = match crossed_eos {
                        true => eos_token_id,
                        false => *tokens_with_history.add(position - shift) as u64,
                    };
                    crossed_eos |= shifted_token == eos_token_id;
                    hash ^= shifted_token * multipliers[shift];
                }
                let row = hash % head_vocab_sizes[head] as u64 + head_offsets[head] as u64;
                *ngram_ids.add(token * num_heads + head) = row as u32;
            }
        }
    }
}

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
