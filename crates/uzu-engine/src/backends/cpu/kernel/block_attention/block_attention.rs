use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(DepthAttention)]
#[variants(T, f32, f16, bf16)]
pub fn depth_attention<T: DataTypeRepr + Float>(
    stream: *const T,
    source_slots: *const u32,
    queries: *const f32,
    output: *mut T,
    token_count: u32,
    stream_token_count: u32,
    token_offset: u32,
    source_count: u32,
    epsilon: f32,
    #[specialize] model_dim: u32,
    #[specialize] query_count: u32,
) {
    let token_count = token_count as usize;
    let token_offset = token_offset as usize;
    let source_count = source_count as usize;
    let model_dim = model_dim as usize;
    let query_count = query_count as usize;
    let slot_stride = stream_token_count as usize * model_dim;
    let logit_scale = (model_dim as f32).sqrt().recip();

    unsafe {
        for token in 0..token_count {
            let sources = (0..source_count)
                .map(|source| {
                    let slot = *source_slots.add(source) as usize;
                    let row = stream.add(slot * slot_stride + (token_offset + token) * model_dim);
                    (0..model_dim).map(|channel| (*row.add(channel)).to_f32().unwrap()).collect::<Vec<f32>>()
                })
                .collect::<Vec<_>>();
            let inverse_rms = sources
                .iter()
                .map(|values| {
                    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / model_dim as f32;
                    (mean_square + epsilon).sqrt().recip()
                })
                .collect::<Vec<f32>>();
            for query in 0..query_count {
                let query_row = queries.add(query * model_dim);
                let logits = sources
                    .iter()
                    .zip(&inverse_rms)
                    .map(|(values, inverse_rms)| {
                        let dot = (0..model_dim).map(|channel| values[channel] * *query_row.add(channel)).sum::<f32>();
                        dot * inverse_rms * logit_scale
                    })
                    .collect::<Vec<f32>>();
                let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exponentials = logits.iter().map(|logit| (logit - maximum).exp()).collect::<Vec<f32>>();
                let denominator = exponentials.iter().sum::<f32>();
                for channel in 0..model_dim {
                    let mixed = (0..source_count)
                        .map(|source| exponentials[source] / denominator * inverse_rms[source] * sources[source][channel])
                        .sum::<f32>();
                    *output.add((query * token_count + token) * model_dim + channel) = T::from(mixed).unwrap();
                }
            }
        }
    }
}

#[kernel(MessageGateDown)]
#[variants(T, f32, f16, bf16)]
pub fn message_gate_down<T: DataTypeRepr + Float>(
    inputs: *const T,
    messages: *const T,
    down_weights: *const bf16,
    low_rank: *mut f32,
    token_count: u32,
    input_slot_stride: u32,
    epsilon: f32,
    #[specialize] model_dim: u32,
    #[specialize] rank: u32,
    #[specialize] slot_count: u32,
) {
    let (model_dim, rank, token_count) = (model_dim as usize, rank as usize, token_count as usize);
    let unit_rms_scale = |row: *const T| unsafe {
        let mean_square =
            (0..model_dim).map(|channel| (*row.add(channel)).to_f32().unwrap().powi(2)).sum::<f32>() / model_dim as f32;
        (mean_square + epsilon).sqrt().recip()
    };
    unsafe {
        for slot in 0..slot_count as usize {
            for token in 0..token_count {
                let input_row = inputs.add(slot * input_slot_stride as usize + token * model_dim);
                let message_row = messages.add((slot * token_count + token) * model_dim);
                let (input_scale, message_scale) = (unit_rms_scale(input_row), unit_rms_scale(message_row));
                for row in 0..rank {
                    let weights = down_weights.add((slot * rank + row) * 2 * model_dim);
                    let (mut input_sum, mut message_sum) = (0.0f32, 0.0f32);
                    for channel in 0..model_dim {
                        input_sum += (*weights.add(channel)).to_f32() * (*input_row.add(channel)).to_f32().unwrap();
                        message_sum +=
                            (*weights.add(model_dim + channel)).to_f32() * (*message_row.add(channel)).to_f32().unwrap();
                    }
                    *low_rank.add((slot * token_count + token) * rank + row) =
                        input_sum * input_scale + message_sum * message_scale;
                }
            }
        }
    }
}

#[kernel(MessageGateApply)]
#[variants(T, f32, f16, bf16)]
pub fn message_gate_apply<T: DataTypeRepr + Float>(
    inputs: *const T,
    messages: *const T,
    low_rank: *const f32,
    forget_weights: *const f32,
    gain_weights: *const f32,
    output: *mut T,
    token_count: u32,
    input_slot_stride: u32,
    #[specialize] model_dim: u32,
    #[specialize] rank: u32,
    #[specialize] slot_count: u32,
) {
    let (model_dim, rank, token_count) = (model_dim as usize, rank as usize, token_count as usize);
    unsafe {
        for slot in 0..slot_count as usize {
            let slot_forget = forget_weights.add(slot * model_dim * rank);
            let slot_gain = gain_weights.add(slot * model_dim * rank);
            for token in 0..token_count {
                let row = slot * token_count + token;
                let input_row = inputs.add(slot * input_slot_stride as usize + token * model_dim);
                for channel in 0..model_dim {
                    let (mut forget, mut gain) = (0.0f32, 0.0f32);
                    for index in 0..rank {
                        let z = *low_rank.add(row * rank + index);
                        forget += *slot_forget.add(channel * rank + index) * z;
                        gain += *slot_gain.add(channel * rank + index) * z;
                    }
                    let offset = row * model_dim + channel;
                    let mixed = (1.0 - forget.tanh()) * (*input_row.add(channel)).to_f32().unwrap()
                        + gain.tanh() * (*messages.add(offset)).to_f32().unwrap();
                    *output.add(offset) = T::from(mixed).unwrap();
                }
            }
        }
    }
}
