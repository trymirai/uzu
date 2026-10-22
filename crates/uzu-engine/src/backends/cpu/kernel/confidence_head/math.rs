//! Arithmetic shared by the confidence head's CPU kernels: the local part of a slot from its pool row, its node
//! vector and its children. Everything is f32, in the order the Metal kernels use.

use crate::encodable_block::conf_head::{CONF_HEAD_SLOT_NUMBERS as SLOT_NUMBERS, layout::LocalLayout};

/// The floor of the head's log-probabilities of pool candidates.
pub const LOG_PROB_FLOOR: f32 = -30.0;

/// Layer norm: mean, then the mean of squared deviations.
pub fn layer_norm(
    values: &[f32],
    scales: &[f32],
    biases: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let count = values.len() as f32;
    let mean = values.iter().sum::<f32>() / count;
    let variance = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f32>() / count;
    let inverse = 1.0 / (variance + epsilon).sqrt();
    values
        .iter()
        .zip(scales)
        .zip(biases)
        .map(|((value, scale), bias)| (value - mean) * inverse * scale + bias)
        .collect()
}

/// log_softmax as (x - max) - ln(sum exp(x - max)), clamped at LOG_PROB_FLOOR.
pub fn log_softmax_floor(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = values.iter().map(|value| (value - max).exp()).sum::<f32>();
    let log_sum = sum.ln();
    values.iter().map(|value| ((value - max) - log_sum).max(LOG_PROB_FLOOR)).collect()
}

pub fn softplus(value: f32) -> f32 {
    value.max(0.0) + (1.0 + (-value.abs()).exp()).ln()
}

pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// `weights` is [outputs, inputs] row-major.
pub fn dense(
    weights: &[f32],
    biases: &[f32],
    input: &[f32],
) -> Vec<f32> {
    let inputs = input.len();
    biases
        .iter()
        .enumerate()
        .map(|(output, bias)| {
            weights[output * inputs..(output + 1) * inputs].iter().zip(input).map(|(w, x)| w * x).sum::<f32>() + bias
        })
        .collect()
}

pub struct Dims {
    pub hidden: usize,
    pub pool_hidden: usize,
    pub edge_hidden: usize,
}

/// z_u's pool term: the set encoder over every candidate (dense + SiLU, dense), the pi_l-weighted mean and the
/// maximum, then the output layer.
pub fn pool_summary(
    weights: &[f32],
    layout: &LocalLayout,
    dims: &Dims,
    log_draft: &[f32],
    log_logits: &[f32],
    noise: &[f32],
    weight: &[f32],
) -> Vec<f32> {
    let ph = dims.pool_hidden;
    let entry = &weights[layout.pool_entry_weights..];
    let entry_biases = &weights[layout.pool_entry_biases..layout.pool_entry_biases + ph];
    let hidden_weights = &weights[layout.pool_hidden_weights..layout.pool_hidden_weights + ph * ph];
    let hidden_biases = &weights[layout.pool_hidden_biases..layout.pool_hidden_biases + ph];
    let mut mean = vec![0.0f32; ph];
    let mut peak = vec![f32::NEG_INFINITY; ph];
    for candidate in 0..log_draft.len() {
        let numbers = [log_draft[candidate], log_logits[candidate] - log_draft[candidate], noise[candidate]];
        let first = (0..ph)
            .map(|unit| {
                silu(
                    entry[unit * 3] * numbers[0]
                        + entry[unit * 3 + 1] * numbers[1]
                        + entry[unit * 3 + 2] * numbers[2]
                        + entry_biases[unit],
                )
            })
            .collect::<Vec<_>>();
        let second = dense(hidden_weights, hidden_biases, &first);
        for unit in 0..ph {
            mean[unit] += weight[candidate] * second[unit];
            peak[unit] = peak[unit].max(second[unit]);
        }
    }
    let summary = [mean, peak].concat();
    dense(
        &weights[layout.pool_out_weights..layout.pool_out_weights + dims.hidden * 2 * ph],
        &weights[layout.pool_out_biases..layout.pool_out_biases + dims.hidden],
        &summary,
    )
}

/// alpha and beta: softplus of the output layer's first two rows, capped.
pub fn alpha_beta(
    weights: &[f32],
    layout: &LocalLayout,
    node: &[f32],
    cap: f32,
) -> (f32, f32) {
    let hidden = node.len();
    let row = |index: usize| {
        weights[layout.output_weights + index * hidden..layout.output_weights + (index + 1) * hidden]
            .iter()
            .zip(node)
            .map(|(w, z)| w * z)
            .sum::<f32>()
            + weights[layout.output_biases + index]
    };
    (softplus(row(0)).min(cap), softplus(row(1)).min(cap))
}

/// The standardised slot numbers of a child.
pub fn slot_numbers(
    weights: &[f32],
    layout: &LocalLayout,
    raw: [f32; SLOT_NUMBERS],
) -> [f32; SLOT_NUMBERS] {
    std::array::from_fn(|index| {
        (raw[index] - weights[layout.slot_means + index]) / weights[layout.slot_stds + index]
            * weights[layout.slot_keep + index]
    })
}

/// delta of one child: the residual edge MLP of depth one over [z_u; e_c; z_u * e_c; slot].
pub fn edge_delta(
    weights: &[f32],
    layout: &LocalLayout,
    dims: &Dims,
    node: &[f32],
    vector: &[f32],
    numbers: &[f32],
    epsilon: f32,
) -> f32 {
    let hidden = dims.hidden;
    let width = LocalLayout::edge_input(hidden);
    let input = [node, vector, &node.iter().zip(vector).map(|(z, e)| z * e).collect::<Vec<_>>(), numbers].concat();
    let normed = layer_norm(
        &input,
        &weights[layout.edge_input_scales..layout.edge_input_scales + width],
        &weights[layout.edge_input_biases..layout.edge_input_biases + width],
        epsilon,
    );
    let mut state = dense(
        &weights[layout.edge_entry_weights..layout.edge_entry_weights + hidden * width],
        &weights[layout.edge_entry_biases..layout.edge_entry_biases + hidden],
        &normed,
    );
    let block = layer_norm(
        &state,
        &weights[layout.edge_block_scales..layout.edge_block_scales + hidden],
        &weights[layout.edge_block_biases..layout.edge_block_biases + hidden],
        epsilon,
    );
    let raised = dense(
        &weights[layout.edge_up_weights..layout.edge_up_weights + dims.edge_hidden * hidden],
        &weights[layout.edge_up_biases..layout.edge_up_biases + dims.edge_hidden],
        &block,
    )
    .into_iter()
    .map(silu)
    .collect::<Vec<_>>();
    let lowered = dense(
        &weights[layout.edge_down_weights..layout.edge_down_weights + hidden * dims.edge_hidden],
        &weights[layout.edge_down_biases..layout.edge_down_biases + hidden],
        &raised,
    );
    for (value, update) in state.iter_mut().zip(lowered) {
        *value += update;
    }
    let output = layer_norm(
        &state,
        &weights[layout.edge_final_scales..layout.edge_final_scales + hidden],
        &weights[layout.edge_final_biases..layout.edge_final_biases + hidden],
        epsilon,
    );
    output.iter().zip(&weights[layout.delta_weights..layout.delta_weights + hidden]).map(|(y, w)| y * w).sum::<f32>()
        + weights[layout.delta_bias]
}

/// min(log q_u(c), 0) of every child: scores alpha * l + beta * G over the whole pool, delta added on the real
/// children's indices, minus their log-sum-exp.
pub fn child_edges(
    alpha: f32,
    beta: f32,
    logits: &[f32],
    noise: &[f32],
    indices: &[usize],
    deltas: &[f32],
    real: &[bool],
) -> Vec<f32> {
    let mut scores = logits.iter().zip(noise).map(|(l, g)| alpha * l + beta * g).collect::<Vec<_>>();
    for ((&index, &delta), &is_real) in indices.iter().zip(deltas).zip(real) {
        if is_real {
            scores[index] += delta;
        }
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normaliser = max + scores.iter().map(|score| (score - max).exp()).sum::<f32>().ln();
    indices.iter().map(|&index| (scores[index] - normaliser).min(0.0)).collect()
}
