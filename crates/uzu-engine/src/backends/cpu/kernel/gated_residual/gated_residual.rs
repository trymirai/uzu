use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[kernel(GatedResidualExpand)]
#[variants(T, f32, f16, bf16)]
pub fn gated_residual_expand<T: DataTypeRepr + Float>(
    input: *const T,
    output: *mut T,
    model_dim: u32,
    stream_count: u32,
    length: u32,
) {
    let model_dim = model_dim as usize;
    let row_size = stream_count as usize * model_dim;
    for position in 0..length as usize {
        let row = position / row_size;
        let channel = position % model_dim;
        unsafe { *output.add(position) = *input.add(row * model_dim + channel) };
    }
}

#[kernel(GatedResidualLowRankActivation)]
#[variants(T, f32, f16, bf16)]
pub fn gated_residual_low_rank_activation<T: DataTypeRepr + Float>(
    values: *mut T,
    stream_count: u32,
    length: u32,
) {
    for position in 0..length as usize {
        unsafe {
            let x = (*values.add(position)).to_f32().unwrap() / stream_count as f32;
            *values.add(position) = T::from(x * sigmoid(x)).unwrap();
        }
    }
}

#[kernel(GatedResidualMix)]
#[variants(T, f32, f16, bf16)]
pub fn gated_residual_mix<T: DataTypeRepr + Float>(
    streams: *const T,
    logits: *const T,
    output: *mut T,
    model_dim: u32,
    stream_count: u32,
    length: u32,
) {
    let model_dim = model_dim as usize;
    let stream_count = stream_count as usize;
    for position in 0..length as usize {
        let row = position / model_dim;
        let channel = position % model_dim;
        let row_offset = row * stream_count * model_dim + channel;
        let sum: f32 = (0..stream_count)
            .map(|stream| {
                let index = row_offset + stream * model_dim;
                let (logit, value) = unsafe { (*logits.add(index), *streams.add(index)) };
                sigmoid(logit.to_f32().unwrap()) * value.to_f32().unwrap()
            })
            .sum();
        unsafe { *output.add(position) = T::from(sum / stream_count as f32).unwrap() };
    }
}

#[kernel(GatedResidualUpdate)]
#[variants(T, f32, f16, bf16)]
pub fn gated_residual_update<T: DataTypeRepr + Float>(
    streams: *const T,
    injection_logits: *const T,
    update: *const T,
    output: *mut T,
    model_dim: u32,
    stream_count: u32,
    length: u32,
) {
    let model_dim = model_dim as usize;
    for position in 0..length as usize {
        let row_stream = position / model_dim;
        let row = row_stream / stream_count as usize;
        let channel = position % model_dim;
        unsafe {
            let logit = (*injection_logits.add(row_stream)).to_f32().unwrap();
            let weight = 2.0 * sigmoid(logit / stream_count as f32);
            let value = (*streams.add(position)).to_f32().unwrap()
                + weight * (*update.add(row * model_dim + channel)).to_f32().unwrap();
            *output.add(position) = T::from(value).unwrap();
        }
    }
}
