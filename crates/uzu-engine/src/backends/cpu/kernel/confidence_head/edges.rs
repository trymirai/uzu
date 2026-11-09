use half::bf16;
use uzu_engine_macros::kernel;

use super::head::{LOG_PROB_FLOOR, NONE, dense, layer_norm, log_sum, weights};
use crate::backends::{
    common::gpu_types::{
        activation_type::ActivationType,
        confidence_head::{ConfidenceHeadParams, POOL_FEATURES, SLOT_NUMBERS},
        weaver::{FrontierIdx, MetadataIdx, TreeIdx},
    },
    cpu::kernel::weaver::weaver_selection::top_k_score_key,
};

#[kernel(ConfidenceHeadEdges)]
pub fn confidence_head_edges(
    #[optional(!refine)] node_metadata: Option<*const u32>,
    #[optional(!refine)] node_valid: Option<*const u32>,
    #[optional(!refine)] candidate_ids: Option<*const u32>,
    #[optional(!refine)] pool_features: Option<*const f32>,
    #[optional(!refine)] slot_stats: Option<*const bf16>,
    slot_rows: *mut u32,
    #[optional(!refine)] row_slots: Option<*mut u32>,
    nodes: *const bf16,
    #[optional(refine)] node_shifts: Option<*const bf16>,
    #[optional(refine)] child_shifts: Option<*const bf16>,
    token_features: *const bf16,
    child_numbers: *mut f32,
    child_pools: *mut u32,
    scores: *const f32,
    frontier: *mut u32,
    packed_tree: *mut u32,
    #[optional(!refine)] child_inputs: Option<*mut bf16>,
    input_norm: *const bf16,
    entry_weights: *const bf16,
    entry_biases: *const bf16,
    block_norm: *const bf16,
    up_weights: *const bf16,
    up_biases: *const bf16,
    down_weights: *const bf16,
    down_biases: *const bf16,
    final_norm: *const bf16,
    readout_weights: *const bf16,
    readout_biases: *const bf16,
    params: ConfidenceHeadParams,
    groups: u32,
    #[specialize] model_dim: u32,
    #[specialize] edge_hidden: u32,
    #[specialize] refine: bool,
) {
    let (slots, candidates, children) =
        (params.tree_slot_count as usize, params.candidates as usize, params.children as usize);
    let (width, hidden, groups) = (model_dim as usize, edge_hidden as usize, groups as usize);
    let slot_numbers = SLOT_NUMBERS as usize;
    let inputs = 3 * width + slot_numbers;
    let norm = |buffer: *const bf16, count: usize| {
        let values = weights(buffer, 2 * count);
        (values[..count].to_vec(), values[count..].to_vec())
    };
    let readout = (weights(readout_weights, 3 * width), weights(readout_biases, 3));
    let input_norm = norm(input_norm, inputs);
    let entry_layer = (weights(entry_weights, width * inputs), weights(entry_biases, width));
    let block_norm = norm(block_norm, width);
    let up = (weights(up_weights, hidden * width), weights(up_biases, hidden));
    let down = (weights(down_weights, width * hidden), weights(down_biases, width));
    let final_norm = norm(final_norm, width);
    let capacity = slots * children;
    let value = |buffer: *const bf16, index: usize| unsafe { (*buffer.add(index)).to_f32() };
    let shift = |buffer: Option<*const bf16>, index: usize| buffer.map_or(0.0, |buffer| value(buffer, index));
    let token_of = |entry: usize| unsafe { *frontier.add(FrontierIdx::TokenId as usize * capacity + entry) };
    for group in 0..groups {
        let slot = match node_metadata {
            Some(metadata) => (unsafe { *metadata.add(MetadataIdx::TreeSlot as usize * groups + group) }) as usize,
            None => group,
        };
        let skip = slot >= slots
            || match node_valid {
                Some(valid) => (unsafe { *valid.add(group) }) == 0,
                None => (unsafe { *slot_rows.add(slot) }) == NONE,
            };
        if skip {
            continue;
        }
        let entries = (slot * children..(slot + 1) * children).collect::<Vec<_>>();
        if let (Some(metadata), Some(ids), Some(features), Some(stats), Some(row_slots)) =
            (node_metadata, candidate_ids, pool_features, slot_stats, row_slots)
        {
            let depth = (unsafe { *metadata.add(MetadataIdx::Depth as usize * groups + group) }) as usize;
            let stats = weights(stats, 2 * slot_numbers);
            let feature = |index: usize| unsafe { *features.add(depth * POOL_FEATURES as usize * candidates + index) };
            for &entry in &entries {
                let token = token_of(entry);
                if let Some(index) =
                    (0..candidates).find(|&index| (unsafe { *ids.add(depth * candidates + index) }) == token)
                {
                    let numbers = [
                        feature(index),
                        (unsafe { *scores.add(2 * slot * candidates + index) }).max(LOG_PROB_FLOOR),
                        unsafe { *scores.add((2 * slot + 1) * candidates + index) },
                        feature(2 * candidates + index),
                    ];
                    unsafe {
                        *child_pools.add(entry) = index as u32;
                        for (number, value) in numbers.into_iter().enumerate() {
                            *child_numbers.add(entry * slot_numbers + number) =
                                (value - stats[number]) / stats[slot_numbers + number];
                        }
                    }
                }
            }
            let own_row = if slot == 0 {
                0
            } else {
                let tree = |index: TreeIdx| unsafe { *packed_tree.add(index as usize * slots + slot) };
                let parent = tree(TreeIdx::ParentSlot) as usize;
                let own_token = tree(TreeIdx::TokenId);
                (0..children)
                    .rev()
                    .find(|&child| token_of(parent * children + child) == own_token)
                    .map_or(0, |child| 1 + parent * children + child)
            };
            unsafe {
                *slot_rows.add(slot) = own_row as u32;
                *row_slots.add(own_row) = slot as u32;
            }
        }
        let row = (unsafe { *slot_rows.add(slot) }) as usize;
        let node = (0..width)
            .map(|unit| value(nodes, slot * width + unit) + shift(node_shifts, row * width + unit))
            .collect::<Vec<_>>();
        let raw = dense(&readout.0[width..], &readout.1[1..], &node);
        let [alpha, beta] =
            [raw[0], raw[1]].map(|value| ActivationType::SOFTPLUS.activate(value).min(params.logit_cap));
        let mut logits = (0..candidates)
            .map(|index| unsafe {
                alpha * *scores.add(2 * slot * candidates + index)
                    + beta * *scores.add((2 * slot + 1) * candidates + index)
            })
            .collect::<Vec<_>>();
        let mut child_rows = Vec::new();
        for &entry in &entries {
            let token = token_of(entry) as usize;
            let tokens = (0..width)
                .map(|unit| {
                    value(token_features, token * width + unit) + shift(child_shifts, (1 + entry) * width + unit)
                })
                .collect::<Vec<_>>();
            let numbers = unsafe { std::slice::from_raw_parts(child_numbers.add(entry * slot_numbers), slot_numbers) };
            let input = node
                .iter()
                .chain(&tokens)
                .copied()
                .chain(node.iter().zip(&tokens).map(|(node, token)| node * token))
                .chain(numbers.iter().copied())
                .collect::<Vec<_>>();
            let input = layer_norm(&input, &input_norm.0, &input_norm.1, params.epsilon);
            let state = dense(&entry_layer.0, &entry_layer.1, &input);
            let block = layer_norm(&state, &block_norm.0, &block_norm.1, params.epsilon);
            let raised = dense(&up.0, &up.1, &block)
                .into_iter()
                .map(|value| ActivationType::SILU.activate(value))
                .collect::<Vec<_>>();
            let lowered = dense(&down.0, &down.1, &raised);
            let state = state.iter().zip(lowered).map(|(state, lowered)| state + lowered).collect::<Vec<_>>();
            let state = layer_norm(&state, &final_norm.0, &final_norm.1, params.epsilon);
            let pool = unsafe { *child_pools.add(entry) } as usize;
            logits[pool] += dense(&readout.0[..width], &readout.1[..1], &state)[0];
            child_rows.push((pool, tokens.into_iter().chain(numbers.iter().copied()).collect::<Vec<_>>()));
        }
        let log_sum = log_sum(&logits);
        let parent_path = f32::from_bits(unsafe { *packed_tree.add(TreeIdx::PathLogprobBits as usize * slots + slot) });
        for (&entry, (pool, row_values)) in entries.iter().zip(&child_rows) {
            let edge = (logits[*pool] - log_sum).min(0.0);
            let field = |index: FrontierIdx| unsafe { frontier.add(index as usize * capacity + entry) };
            unsafe { *field(FrontierIdx::EdgeLogprobBits) = edge.to_bits() };
            if refine {
                let token = unsafe { *field(FrontierIdx::TokenId) };
                for other in 0..slots {
                    let tree = |index: TreeIdx| unsafe { packed_tree.add(index as usize * slots + other) };
                    if unsafe { *tree(TreeIdx::ParentSlot) } == slot as u32
                        && unsafe { *tree(TreeIdx::TokenId) } == token
                    {
                        unsafe { *tree(TreeIdx::EdgeLogprobBits) = edge.to_bits() };
                    }
                }
            } else if let Some(child_inputs) = child_inputs {
                unsafe {
                    *field(FrontierIdx::PathLogprobBits) = (parent_path + edge).to_bits();
                    *field(FrontierIdx::PathScoreKey) = top_k_score_key(parent_path + edge);
                    for (column, value) in row_values.iter().enumerate() {
                        *child_inputs.add(entry * (width + slot_numbers) + column) = bf16::from_f32(*value);
                    }
                }
            }
        }
    }
}
