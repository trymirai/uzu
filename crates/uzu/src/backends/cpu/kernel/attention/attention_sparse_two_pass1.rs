use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

const TOTAL_BLOCKS_COUNT: u32 = 32;

#[kernel(AttentionSparseTwoPass1)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_sparse_two_pass1<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    keys: *const T,
    values: *const T,
    selected_pages: *const i32,
    out: *mut f32,
    sums: *mut f32,
    maxs: *mut f32,
    gqa_factor: u32,
    selected_page_count: u32,
    page_size: u32,
    recent_start: u32,
    prefix_length: u32,
    k_head_stride: u32,
    k_seq_stride: u32,
    v_head_stride: u32,
    v_seq_stride: u32,
    scale: f32,
    #[optional(has_sinks)] sinks: Option<*const f32>,
    num_heads: u32,
    #[specialize] has_sinks: bool,
) {
    let mut query = vec![0.0f32; HEAD_DIM as usize];
    let mut partial = vec![0.0f32; HEAD_DIM as usize];

    for head_idx in 0..num_heads {
        let kv_head_idx = head_idx / gqa_factor;
        let query_base = unsafe { queries.add((head_idx * HEAD_DIM) as usize) };
        let key_base = unsafe { keys.add((kv_head_idx * k_head_stride) as usize) };
        let value_base = unsafe { values.add((kv_head_idx * v_head_stride) as usize) };

        for dim in 0..HEAD_DIM as usize {
            query[dim] = scale * unsafe { *query_base.add(dim) }.to_f32().unwrap();
        }

        for block_idx in 0..TOTAL_BLOCKS_COUNT {
            partial.fill(0.0);
            let mut max_score = if has_sinks && block_idx == 0 {
                unsafe { *sinks.unwrap().add(head_idx as usize) }
            } else {
                -1e9f32
            };
            let mut sum_exp = if has_sinks && block_idx == 0 {
                1.0
            } else {
                0.0
            };

            for page_slot in 0..selected_page_count as usize {
                let page_start = unsafe { *selected_pages.add(page_slot) as usize } * page_size as usize;
                let page_end = page_start + page_size as usize;
                let mut token = page_start
                    + (block_idx as usize + TOTAL_BLOCKS_COUNT as usize - page_start % TOTAL_BLOCKS_COUNT as usize)
                        % TOTAL_BLOCKS_COUNT as usize;
                while token < page_end {
                    update_sparse_block(
                        &query,
                        key_base,
                        value_base,
                        token,
                        k_seq_stride as usize,
                        v_seq_stride as usize,
                        &mut max_score,
                        &mut sum_exp,
                        &mut partial,
                    );
                    token += TOTAL_BLOCKS_COUNT as usize;
                }
            }

            let mut token = recent_start as usize
                + (block_idx as usize + TOTAL_BLOCKS_COUNT as usize
                    - recent_start as usize % TOTAL_BLOCKS_COUNT as usize)
                    % TOTAL_BLOCKS_COUNT as usize;
            while token < prefix_length as usize {
                update_sparse_block(
                    &query,
                    key_base,
                    value_base,
                    token,
                    k_seq_stride as usize,
                    v_seq_stride as usize,
                    &mut max_score,
                    &mut sum_exp,
                    &mut partial,
                );
                token += TOTAL_BLOCKS_COUNT as usize;
            }

            if prefix_length % TOTAL_BLOCKS_COUNT == block_idx {
                update_sparse_block(
                    &query,
                    key_base,
                    value_base,
                    prefix_length as usize,
                    k_seq_stride as usize,
                    v_seq_stride as usize,
                    &mut max_score,
                    &mut sum_exp,
                    &mut partial,
                );
            }

            let output_base = ((head_idx * TOTAL_BLOCKS_COUNT + block_idx) * HEAD_DIM) as usize;
            for dim in 0..HEAD_DIM as usize {
                unsafe {
                    *out.add(output_base + dim) = partial[dim];
                }
            }
            unsafe {
                *sums.add((head_idx * TOTAL_BLOCKS_COUNT + block_idx) as usize) = sum_exp;
                *maxs.add((head_idx * TOTAL_BLOCKS_COUNT + block_idx) as usize) = max_score;
            }
        }
    }
}

fn update_sparse_block<T: ArrayElement + Float>(
    query: &[f32],
    key_base: *const T,
    value_base: *const T,
    token: usize,
    k_seq_stride: usize,
    v_seq_stride: usize,
    max_score: &mut f32,
    sum_exp: &mut f32,
    partial: &mut [f32],
) {
    let key = unsafe { key_base.add(token * k_seq_stride) };
    let mut score = 0.0f32;
    for dim in 0..query.len() {
        score += query[dim] * unsafe { *key.add(dim) }.to_f32().unwrap();
    }

    let new_max = f32::max(*max_score, score);
    let factor = (*max_score - new_max).exp();
    let exp_score = (score - new_max).exp();
    *max_score = new_max;
    *sum_exp = *sum_exp * factor + exp_score;

    let value = unsafe { value_base.add(token * v_seq_stride) };
    for dim in 0..partial.len() {
        partial[dim] = partial[dim] * factor + exp_score * unsafe { *value.add(dim) }.to_f32().unwrap();
    }
}
