use half::bf16;
use uzu_engine_macros::kernel;

use super::head::{LOG_PROB_FLOOR, dense, log_sum, weights};
use crate::backends::common::gpu_types::{
    activation_type::ActivationType,
    confidence_head::{ConfidenceHeadParams, POOL_FEATURES, POOL_INPUTS},
    weaver::MetadataIdx,
};

#[kernel(ConfidenceHeadNodes)]
pub fn confidence_head_nodes(
    residual_logits: *const bf16,
    candidate_logits: *const f32,
    pool_features: *const f32,
    round_metadata: *const u32,
    round_valid: *const u32,
    draft: *const bf16,
    pool_entry_weights: *const bf16,
    pool_entry_biases: *const bf16,
    pool_hidden_weights: *const bf16,
    pool_hidden_biases: *const bf16,
    pool_out_weights: *const bf16,
    pool_out_biases: *const bf16,
    scores: *mut f32,
    nodes: *mut bf16,
    node_inputs: *mut bf16,
    params: ConfidenceHeadParams,
    rows: u32,
    #[specialize] model_dim: u32,
    #[specialize] pool_hidden: u32,
) {
    let (rows, slots, candidates) = (rows as usize, params.tree_slot_count as usize, params.candidates as usize);
    let (width, pool) = (model_dim as usize, pool_hidden as usize);
    let features_count = POOL_FEATURES as usize * candidates;
    let entry_weights = weights(pool_entry_weights, POOL_INPUTS as usize * pool);
    let entry_biases = weights(pool_entry_biases, pool);
    let hidden_weights = weights(pool_hidden_weights, pool * pool);
    let hidden_biases = weights(pool_hidden_biases, pool);
    let out_weights = weights(pool_out_weights, width * 2 * pool);
    let out_biases = weights(pool_out_biases, width);
    let read = |buffer: *const u32, index: usize| unsafe { *buffer.add(index) };
    for row in 0..rows {
        let slot = read(round_metadata, MetadataIdx::TreeSlot as usize * rows + row) as usize;
        if read(round_valid, row) == 0 || slot >= slots {
            continue;
        }
        let depth = read(round_metadata, MetadataIdx::Depth as usize * rows + row) as usize;
        let features = unsafe { std::slice::from_raw_parts(pool_features.add(depth * features_count), features_count) };
        let logits = (0..candidates)
            .map(|index| unsafe {
                *candidate_logits.add(depth * candidates + index)
                    + (*residual_logits.add(row * candidates + index)).to_f32()
            })
            .collect::<Vec<_>>();
        let log_sum = log_sum(&logits);
        let mut mean = vec![0.0f32; pool];
        let mut peak = vec![f32::NEG_INFINITY; pool];
        for (index, logit) in logits.iter().enumerate() {
            let (log_draft, noise) = (features[index], features[candidates + index]);
            let log_logit = logit - log_sum;
            let floored = log_logit.max(LOG_PROB_FLOOR);
            let hidden = dense(&entry_weights, &entry_biases, &[log_draft, floored - log_draft, noise])
                .into_iter()
                .map(|value| ActivationType::SILU.activate(value, 1.0))
                .collect::<Vec<_>>();
            let encoded = dense(&hidden_weights, &hidden_biases, &hidden);
            for (unit, value) in encoded.into_iter().enumerate() {
                mean[unit] += log_logit.exp() * value;
                peak[unit] = peak[unit].max(value);
            }
            unsafe {
                *scores.add(2 * slot * candidates + index) = log_logit;
                *scores.add((2 * slot + 1) * candidates + index) = noise;
            }
        }
        let summary = mean.into_iter().chain(peak).collect::<Vec<_>>();
        let node = dense(&out_weights, &out_biases, &summary);
        for (unit, value) in node.into_iter().enumerate() {
            let value = bf16::from_f32(value + unsafe { (*draft.add(depth * width + unit)).to_f32() });
            unsafe {
                *nodes.add(slot * width + unit) = value;
                *node_inputs.add(slot * width + unit) = value;
            }
        }
    }
}
