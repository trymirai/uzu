use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::common::gpu_types::{log_scaling::LogScalingParams, trie::TrieNode},
    data_type::DataTypeRepr,
};

#[kernel(AttentionRelativeLogits)]
#[variants(T, f32, f16, bf16)]
pub fn attention_relative_logits<T: DataTypeRepr + Float>(
    states: *const T,
    clip_scales: *const f32,
    profiles: *const f32,
    logits: *mut bf16,
    #[optional(has_trie)] trie: Option<*const TrieNode>,
    token_count: u32,
    state_row_stride: u32,
    clip_threshold: f32,
    #[optional(has_log_scaling)] log_scaling: Option<LogScalingParams>,
    #[specialize] num_heads: u32,
    #[specialize] dim: u32,
    #[specialize] extent: u32,
    #[specialize] has_log_scaling: bool,
    #[specialize] has_trie: bool,
) {
    let (dim, extent) = (dim as usize, extent as usize);
    for token in 0..token_count as usize {
        let position = if has_trie {
            unsafe { (*trie.unwrap().add(token)).height }
        } else {
            token as u32
        };
        for head in 0..num_heads as usize {
            let mut relative = (0..dim)
                .map(|channel| unsafe {
                    (*states.add(token * state_row_stride as usize + head * dim + channel)).to_f32().unwrap()
                        * *clip_scales.add(head * dim + channel)
                })
                .collect::<Vec<f32>>();
            let norm = relative.iter().map(|value| value * value).sum::<f32>().sqrt();
            let mut scale = (clip_threshold / (norm + 1e-6)).min(1.0);
            if has_log_scaling {
                let log_scaling = log_scaling.unwrap();
                let ratio = (log_scaling.position_offset + position + 1) as f32 / log_scaling.n_floor;
                scale *= 1.0 + log_scaling.alpha * ratio.max(1.0).ln();
            }
            relative.iter_mut().for_each(|value| *value *= scale);
            let row = (head * token_count as usize + token) * extent;
            for distance in 0..extent {
                let sum = (0..dim)
                    .map(|channel| relative[channel] * unsafe { *profiles.add(channel * extent + distance) })
                    .sum::<f32>();
                unsafe { *logits.add(row + distance) = bf16::from_f32(sum) };
            }
        }
    }
}
