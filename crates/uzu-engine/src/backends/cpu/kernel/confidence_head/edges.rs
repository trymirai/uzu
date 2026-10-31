use half::bf16;
use uzu_engine_macros::kernel;

use super::head::{NONE, dense, layer_norm, log_sum, weights};
use crate::backends::{
    common::gpu_types::{
        activation_type::ActivationType,
        confidence_head::{ConfidenceHeadParams, SLOT_NUMBERS},
        weaver::{FrontierIdx, MetadataIdx, TreeIdx},
    },
    cpu::kernel::weaver::weaver_selection::top_k_score_key,
};

#[kernel(ConfidenceHeadEdges)]
pub fn confidence_head_edges(
    #[optional(!refine)] node_metadata: Option<*const u32>,
    slot_rows: *const u32,
    nodes: *const bf16,
    #[optional(refine)] node_shifts: Option<*const bf16>,
    #[optional(refine)] child_shifts: Option<*const bf16>,
    token_features: *const bf16,
    child_numbers: *const f32,
    child_pools: *const u32,
    scores: *const f32,
    frontier: *mut u32,
    packed_tree: *mut u32,
    #[optional(!refine)] child_inputs: Option<*mut bf16>,
    input_norm_scales: *const bf16,
    input_norm_biases: *const bf16,
    entry_weights: *const bf16,
    entry_biases: *const bf16,
    block_norm_scales: *const bf16,
    block_norm_biases: *const bf16,
    up_weights: *const bf16,
    up_biases: *const bf16,
    down_weights: *const bf16,
    down_biases: *const bf16,
    final_norm_scales: *const bf16,
    final_norm_biases: *const bf16,
    output_weights: *const bf16,
    output_biases: *const bf16,
    alpha_beta_weights: *const bf16,
    alpha_beta_biases: *const bf16,
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
    let alpha_beta = (weights(alpha_beta_weights, 2 * width), weights(alpha_beta_biases, 2));
    let input_norm = (weights(input_norm_scales, inputs), weights(input_norm_biases, inputs));
    let entry_layer = (weights(entry_weights, width * inputs), weights(entry_biases, width));
    let block_norm = (weights(block_norm_scales, width), weights(block_norm_biases, width));
    let up = (weights(up_weights, hidden * width), weights(up_biases, hidden));
    let down = (weights(down_weights, width * hidden), weights(down_biases, width));
    let final_norm = (weights(final_norm_scales, width), weights(final_norm_biases, width));
    let output = (weights(output_weights, width), weights(output_biases, 1));
    let capacity = slots * children;
    let value = |buffer: *const bf16, index: usize| unsafe { (*buffer.add(index)).to_f32() };
    let shift = |buffer: Option<*const bf16>, index: usize| buffer.map_or(0.0, |buffer| value(buffer, index));
    for group in 0..groups {
        let slot = match node_metadata {
            Some(metadata) => (unsafe { *metadata.add(MetadataIdx::TreeSlot as usize * groups + group) }) as usize,
            None => group,
        };
        let row = if slot < slots {
            unsafe { *slot_rows.add(slot) }
        } else {
            NONE
        };
        if row == NONE {
            continue;
        }
        let row = row as usize;
        let node = (0..width)
            .map(|unit| value(nodes, slot * width + unit) + shift(node_shifts, row * width + unit))
            .collect::<Vec<_>>();
        let raw = dense(&alpha_beta.0, &alpha_beta.1, &node);
        let [alpha, beta] =
            [raw[0], raw[1]].map(|value| ActivationType::SOFTPLUS.activate(value).min(params.logit_cap));
        let mut logits = (0..candidates)
            .map(|index| unsafe {
                alpha * *scores.add(2 * slot * candidates + index)
                    + beta * *scores.add((2 * slot + 1) * candidates + index)
            })
            .collect::<Vec<_>>();
        let entries = (slot * children..(slot + 1) * children).collect::<Vec<_>>();
        let mut child_rows = Vec::new();
        for &entry in &entries {
            let token = unsafe { *frontier.add(FrontierIdx::TokenId as usize * capacity + entry) } as usize;
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
            logits[pool] += dense(&output.0, &output.1, &state)[0];
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
