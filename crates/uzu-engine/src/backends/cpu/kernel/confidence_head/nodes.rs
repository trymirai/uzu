use half::bf16;
use uzu_engine_macros::kernel;

use super::head::{LOG_PROB_FLOOR, SLOT_NUMBERS, dense, log_sum, silu, tensor, weights};
use crate::backends::common::gpu_types::{
    confidence_head::ConfidenceHeadParams,
    weaver::{FrontierIdx, MetadataIdx, TreeIdx},
};

#[kernel(ConfidenceHeadNodes)]
pub fn confidence_head_nodes(
    residual_logits: *const bf16,
    candidate_logits: *const f32,
    candidate_ids: *const u32,
    pool_features: *const f32,
    node_metadata: *const u32,
    node_valid: *const u32,
    packed_tree: *const u32,
    frontier: *const u32,
    draft: *const bf16,
    pool_entry_weights: *const bf16,
    pool_entry_biases: *const bf16,
    pool_hidden_weights: *const bf16,
    pool_hidden_biases: *const bf16,
    pool_out_weights: *const bf16,
    pool_out_biases: *const bf16,
    slot_mean: *const bf16,
    slot_std: *const bf16,
    scores: *mut f32,
    nodes: *mut bf16,
    node_inputs: *mut bf16,
    child_pools: *mut u32,
    child_numbers: *mut f32,
    slot_rows: *mut u32,
    row_slots: *mut u32,
    params: ConfidenceHeadParams,
    rows: u32,
    #[specialize] model_dim: u32,
    #[specialize] pool_hidden: u32,
) {
    let (rows, slots, candidates, children) =
        (rows as usize, params.tree_slot_count as usize, params.candidates as usize, params.children as usize);
    let (width, pool) = (model_dim as usize, pool_hidden as usize);
    let capacity = slots * children;
    let read = |buffer: *const u32, index: usize| unsafe { *buffer.add(index) };
    for row in 0..rows {
        let slot = read(node_metadata, MetadataIdx::TreeSlot as usize * rows + row) as usize;
        if read(node_valid, row) == 0 || slot >= slots {
            continue;
        }
        let depth = read(node_metadata, MetadataIdx::Depth as usize * rows + row) as usize;
        let features = tensor(unsafe { pool_features.add(depth * 3 * candidates) }, 3 * candidates);
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
            let floored = (logit - log_sum).max(LOG_PROB_FLOOR);
            let hidden = dense(
                &weights(pool_entry_weights, 3 * pool),
                &weights(pool_entry_biases, pool),
                &[log_draft, floored - log_draft, noise],
            )
            .into_iter()
            .map(silu)
            .collect::<Vec<_>>();
            let encoded =
                dense(&weights(pool_hidden_weights, pool * pool), &weights(pool_hidden_biases, pool), &hidden);
            for (unit, value) in encoded.into_iter().enumerate() {
                mean[unit] += (logit - log_sum).exp() * value;
                peak[unit] = peak[unit].max(value);
            }
            unsafe {
                *scores.add(2 * slot * candidates + index) = *logit;
                *scores.add((2 * slot + 1) * candidates + index) = noise;
            }
            for child in 0..children {
                let entry = slot * children + child;
                if read(candidate_ids, depth * candidates + index)
                    == read(frontier, FrontierIdx::TokenId as usize * capacity + entry)
                {
                    let numbers = [log_draft, floored, noise, features[2 * candidates + index]];
                    unsafe {
                        *child_pools.add(entry) = index as u32;
                        for (number, value) in numbers.into_iter().enumerate() {
                            *child_numbers.add(entry * SLOT_NUMBERS + number) =
                                (value - (*slot_mean.add(number)).to_f32()) / (*slot_std.add(number)).to_f32();
                        }
                    }
                }
            }
        }
        let summary = mean.into_iter().chain(peak).collect::<Vec<_>>();
        let node = dense(&weights(pool_out_weights, width * 2 * pool), &weights(pool_out_biases, width), &summary);
        for (unit, value) in node.into_iter().enumerate() {
            let value = bf16::from_f32(value + unsafe { (*draft.add(depth * width + unit)).to_f32() });
            unsafe {
                *nodes.add(slot * width + unit) = value;
                *node_inputs.add(slot * width + unit) = value;
            }
        }
        let own_row = if slot == 0 {
            0
        } else {
            let parent = read(packed_tree, TreeIdx::ParentSlot as usize * slots + slot) as usize;
            let own_token = read(packed_tree, TreeIdx::TokenId as usize * slots + slot);
            (0..children)
                .find(|&child| {
                    read(frontier, FrontierIdx::TokenId as usize * capacity + parent * children + child) == own_token
                })
                .map_or(0, |child| 1 + parent * children + child)
        };
        unsafe {
            *slot_rows.add(slot) = own_row as u32;
            *row_slots.add(own_row) = slot as u32;
        }
    }
}
