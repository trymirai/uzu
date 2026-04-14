use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AttentionSparseSingleDecode)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_sparse_single_decode<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    keys: *const T,
    values: *const T,
    selected_pages: *const i32,
    out: *mut T,
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
    for head_idx in 0..num_heads {
        let kv_head_idx = head_idx / gqa_factor;
        let query = unsafe { queries.add((head_idx * HEAD_DIM) as usize) };
        let keys = unsafe { keys.add((kv_head_idx * k_head_stride) as usize) };
        let values = unsafe { values.add((kv_head_idx * v_head_stride) as usize) };
        let output = unsafe { out.add((head_idx * HEAD_DIM) as usize) };

        let mut scaled_query: Vec<f32> = vec![0.0; HEAD_DIM as usize];
        let mut accum: Vec<f32> = vec![0.0; HEAD_DIM as usize];
        for dim in 0..HEAD_DIM as usize {
            scaled_query[dim] = scale * unsafe { *query.add(dim) }.to_f32().unwrap();
        }

        let mut max_score = if has_sinks {
            unsafe { *sinks.unwrap().add(head_idx as usize) }
        } else {
            f32::NEG_INFINITY
        };
        let mut sum_exp_score = if has_sinks {
            1.0
        } else {
            0.0
        };

        for page_slot in 0..selected_page_count as usize {
            let page_start = unsafe { *selected_pages.add(page_slot) as usize } * page_size as usize;
            let page_end = page_start + page_size as usize;
            for token in page_start..page_end {
                let key = unsafe { keys.add(token * k_seq_stride as usize) };
                let mut score = 0.0f32;
                for dim in 0..HEAD_DIM as usize {
                    score += scaled_query[dim] * unsafe { *key.add(dim) }.to_f32().unwrap();
                }

                let new_max = f32::max(max_score, score);
                let factor = (max_score - new_max).exp();
                let exp_score = (score - new_max).exp();
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;

                let value = unsafe { values.add(token * v_seq_stride as usize) };
                for dim in 0..HEAD_DIM as usize {
                    accum[dim] = accum[dim] * factor + exp_score * unsafe { *value.add(dim) }.to_f32().unwrap();
                }
            }
        }

        for token in recent_start as usize..prefix_length as usize {
            let key = unsafe { keys.add(token as usize * k_seq_stride as usize) };
            let mut score = 0.0f32;
            for dim in 0..HEAD_DIM as usize {
                score += scaled_query[dim] * unsafe { *key.add(dim) }.to_f32().unwrap();
            }

            let new_max = f32::max(max_score, score);
            let factor = (max_score - new_max).exp();
            let exp_score = (score - new_max).exp();
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            let value = unsafe { values.add(token as usize * v_seq_stride as usize) };
            for dim in 0..HEAD_DIM as usize {
                accum[dim] = accum[dim] * factor + exp_score * unsafe { *value.add(dim) }.to_f32().unwrap();
            }
        }

        let self_token = prefix_length as usize;
        let key = unsafe { keys.add(self_token * k_seq_stride as usize) };
        let mut score = 0.0f32;
        for dim in 0..HEAD_DIM as usize {
            score += scaled_query[dim] * unsafe { *key.add(dim) }.to_f32().unwrap();
        }

        let new_max = f32::max(max_score, score);
        let factor = (max_score - new_max).exp();
        let exp_score = (score - new_max).exp();
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        let value = unsafe { values.add(self_token * v_seq_stride as usize) };
        for dim in 0..HEAD_DIM as usize {
            accum[dim] = accum[dim] * factor + exp_score * unsafe { *value.add(dim) }.to_f32().unwrap();
        }

        for dim in 0..HEAD_DIM as usize {
            unsafe {
                *output.add(dim) = T::from(accum[dim] / sum_exp_score).unwrap();
            }
        }
    }
}
