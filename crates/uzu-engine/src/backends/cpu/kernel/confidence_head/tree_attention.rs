use half::bf16;
use uzu_engine_macros::kernel;

use super::tree_relations::ABSENT;
use crate::backends::common::gpu_types::confidence_head::RELATIONS;

#[kernel(ConfidenceHeadTreeAttention)]
pub fn confidence_head_tree_attention(
    qkv: *const bf16,
    relations: *const u32,
    relation_bias: *const bf16,
    output: *mut bf16,
    rows: u32,
    #[specialize] num_heads: u32,
    #[specialize] head_dim: u32,
) {
    let (rows, heads, head_dim) = (rows as usize, num_heads as usize, head_dim as usize);
    let width = heads * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    let element = |row: usize, part: usize, head: usize, unit: usize| unsafe {
        (*qkv.add(row * 3 * width + part * width + head * head_dim + unit)).to_f32()
    };
    for query in 0..rows {
        let relation = |key: usize| unsafe { *relations.add(query * rows + key) };
        let keys = if relation(query) == ABSENT {
            Vec::new()
        } else {
            (0..rows).filter(|&key| relation(key) != ABSENT).collect::<Vec<_>>()
        };
        for head in 0..heads {
            let scores = keys
                .iter()
                .map(|&key| {
                    let bias = match relation(key) {
                        0 => 0.0,
                        code => unsafe { (*relation_bias.add(head * RELATIONS as usize + code as usize - 1)).to_f32() },
                    };
                    (0..head_dim).map(|unit| element(query, 0, head, unit) * element(key, 1, head, unit)).sum::<f32>()
                        * scale
                        + bias
                })
                .collect::<Vec<_>>();
            let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores.iter().map(|score| (score - top).exp()).collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            for unit in 0..head_dim {
                let value = if keys.is_empty() {
                    0.0
                } else {
                    keys.iter().zip(&weights).map(|(&key, weight)| weight * element(key, 2, head, unit)).sum::<f32>()
                        / total
                };
                unsafe { *output.add(query * width + head * head_dim + unit) = bf16::from_f32(value) };
            }
        }
    }
}
