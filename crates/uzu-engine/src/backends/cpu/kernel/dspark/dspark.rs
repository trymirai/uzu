use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

const TAP_COUNT: usize = 4;

/// Rounds through `T`, as the GPU kernels store their intermediates.
fn round<T: DataTypeRepr + Float>(value: f32) -> f32 {
    T::from(value).unwrap().to_f32().unwrap()
}

#[kernel(TapMixNorm)]
#[variants(T, f32, f16, bf16)]
pub fn tap_mix_norm<T: DataTypeRepr + Float>(
    tap0: *const T,
    tap1: *const T,
    tap2: *const T,
    tap3: *const T,
    row_indices: *const u32,
    mix_weights: *const f32,
    norm_scales: *const f32,
    output: *mut T,
    row_count: u32,
    epsilon: f32,
    #[specialize] model_dim: u32,
    #[specialize] mix_count: u32,
) {
    let (model_dim, row_count, mix_count) = (model_dim as usize, row_count as usize, mix_count as usize);
    let taps = [tap0, tap1, tap2, tap3];
    unsafe {
        for row in 0..row_count {
            let source = *row_indices.add(row) as usize * model_dim;
            for mix in 0..mix_count {
                let mixed = (0..model_dim)
                    .map(|channel| {
                        (0..TAP_COUNT)
                            .map(|tap| {
                                *mix_weights.add(mix * TAP_COUNT + tap) * (*taps[tap].add(source + channel)).to_f32().unwrap()
                            })
                            .sum::<f32>()
                    })
                    .collect::<Vec<f32>>();
                let mean_square = mixed.iter().map(|value| value * value).sum::<f32>() / model_dim as f32;
                let inverse_rms = (mean_square + epsilon).sqrt().recip();
                for (channel, value) in mixed.iter().enumerate() {
                    *output.add((mix * row_count + row) * model_dim + channel) =
                        T::from(value * inverse_rms * *norm_scales.add(mix * model_dim + channel)).unwrap();
                }
            }
        }
    }
}

#[kernel(MarkovStart)]
#[variants(T, f32, f16, bf16)]
pub fn markov_start<T: DataTypeRepr + Float>(
    anchor: *const u32,
    token_embeddings: *const T,
    draft_hidden: *const T,
    markov_context: *const T,
    inputs: *mut T,
    chain_length: u32,
    #[specialize] rank: u32,
    #[specialize] model_dim: u32,
) {
    let (rank, model_dim) = (rank as usize, model_dim as usize);
    let width = 2 * rank + 2 * model_dim;
    unsafe {
        for step in 0..chain_length as usize {
            let row = inputs.add(step * width);
            if step == 0 {
                let embedding = *anchor as usize * rank;
                for index in 0..rank {
                    *row.add(index) = T::zero();
                    *row.add(rank + index) = *token_embeddings.add(embedding + index);
                }
            }
            for index in 0..model_dim {
                *row.add(2 * rank + index) = *draft_hidden.add((step + 1) * model_dim + index);
                *row.add(2 * rank + model_dim + index) = *markov_context.add(index);
            }
        }
    }
}

#[kernel(MarkovStep)]
#[variants(T, f32, f16, bf16)]
pub fn markov_step<T: DataTypeRepr + Float>(
    projection: *const T,
    state: *const T,
    #[optional(has_next)] next_state: Option<*mut T>,
    latent: *mut T,
    #[specialize] rank: u32,
    #[specialize] has_next: bool,
) {
    let rank = rank as usize;
    unsafe {
        for index in 0..rank {
            let value = |offset: usize| (*projection.add(offset + index)).to_f32().unwrap();
            let gate = round::<T>(1.0 / (1.0 + (-value(0)).exp()));
            let kept = round::<T>(gate * (*state.add(index)).to_f32().unwrap());
            let blend = round::<T>((1.0 - gate) * round::<T>(value(rank).tanh()));
            if has_next {
                *next_state.unwrap().add(index) = T::from(kept + blend).unwrap();
            }
            *latent.add(index) = T::from(value(2 * rank).tanh()).unwrap();
        }
    }
}

#[kernel(MarkovLogits)]
#[variants(T, f32, f16, bf16)]
pub fn markov_logits<T: DataTypeRepr + Float>(
    base_logits: *const T,
    bias: *const T,
    bias_offset: *const T,
    logits: *mut T,
    vocab_size: u32,
    #[optional(has_soft_cap)] soft_cap: Option<f32>,
    #[specialize] has_soft_cap: bool,
) {
    unsafe {
        for index in 0..vocab_size as usize {
            let read = |values: *const T| (*values.add(index)).to_f32().unwrap();
            let mut value = read(base_logits) + round::<T>(read(bias) + read(bias_offset));
            if has_soft_cap {
                let soft_cap = soft_cap.unwrap();
                value = (value / soft_cap).tanh() * soft_cap;
            }
            *logits.add(index) = T::from(value).unwrap();
        }
    }
}

#[kernel(MarkovEmbed)]
#[variants(T, f32, f16, bf16)]
pub fn markov_embed<T: DataTypeRepr + Float>(
    token: *const u32,
    token_embeddings: *const T,
    embedding: *mut T,
    #[specialize] rank: u32,
) {
    let rank = rank as usize;
    unsafe {
        let row = *token as usize * rank;
        for index in 0..rank {
            *embedding.add(index) = *token_embeddings.add(row + index);
        }
    }
}
