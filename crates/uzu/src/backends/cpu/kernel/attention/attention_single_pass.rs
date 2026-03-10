use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AttentionSinglePass)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_single_pass<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    keys: *const T,
    values: *const T,
    out: *mut T,
    gqa_factor: u32,
    sequence_length: u32,
    k_head_stride: u32,
    k_seq_stride: u32,
    v_head_stride: u32,
    v_seq_stride: u32,
    scale: f32,
    #[optional(float_mask)] fmask: Option<*const T>,
    #[optional(has_mask)] mask_kv_seq_stride: Option<u32>,
    #[optional(has_mask)] mask_q_seq_stride: Option<u32>,
    #[optional(has_mask)] mask_head_stride: Option<u32>,
    #[optional(has_sinks)] sinks: Option<*const f32>,
    num_heads: u32,
    suffix_length: u32,
    #[specialize] float_mask: bool,
    #[specialize] has_mask: bool,
    #[specialize] has_sinks: bool,
    #[specialize] do_causal: bool,
) {
    let value_dim = HEAD_DIM;

    for head_idx in 0..num_heads {
        for q_seq_idx in 0..suffix_length {
            let kv_head_idx = head_idx / gqa_factor;
            let o_offset = q_seq_idx * num_heads + head_idx;
            let q_offset = head_idx * suffix_length + q_seq_idx;

            let queries: *const T = unsafe { queries.add((q_offset * HEAD_DIM) as usize) };
            let keys: *const T = unsafe { keys.add((kv_head_idx * k_head_stride) as usize) };
            let values: *const T = unsafe { values.add((kv_head_idx * v_head_stride) as usize) };
            let out: *mut T = unsafe { out.add((o_offset * value_dim) as usize) };

            let fmask: Option<*const T> = if let Some(fmask) = fmask {
                unsafe {
                    let offset = head_idx * mask_head_stride.unwrap_or(0) + q_seq_idx * mask_q_seq_stride.unwrap_or(0);
                    Some(fmask.add(offset as usize))
                }
            } else {
                None
            };

            // Read the query and 0 the output accumulator
            let mut q = vec![0.0f32; HEAD_DIM as usize];
            let mut o = vec![0.0f32; HEAD_DIM as usize];
            for j in 0..HEAD_DIM as usize {
                q[j] = scale * unsafe { *queries.add(j) }.to_f32().unwrap();
            }

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp_score = 0.0f32;
            if has_sinks {
                let q_head_idx = head_idx % num_heads;
                max_score = unsafe { *sinks.unwrap().add(q_head_idx as usize) };
                sum_exp_score = 1.0;
            }

            // For each key
            for i in 0..sequence_length {
                let use_key = !do_causal || i <= (sequence_length - suffix_length + q_seq_idx);

                if use_key {
                    let keys = unsafe { keys.add((i * k_seq_stride) as usize) };

                    // Compute the i-th score
                    let mut score = 0.0f32;
                    for j in 0..HEAD_DIM as usize {
                        score += q[j] * unsafe { *keys.add(j) }.to_f32().unwrap();
                    }
                    if float_mask {
                        let fmask = unsafe { fmask.unwrap().add((i * mask_kv_seq_stride.unwrap()) as usize) };
                        score += f32::max(-1e9, unsafe { *fmask }.to_f32().unwrap());
                    }

                    // Update the accumulators
                    let new_max = f32::max(max_score, score);
                    let factor = (max_score - new_max).exp();
                    let exp_score = (score - new_max).exp();

                    max_score = new_max;
                    sum_exp_score = sum_exp_score * factor + exp_score;

                    // Update the output accumulator
                    let values = unsafe { values.add((i * v_seq_stride) as usize) };
                    for j in 0..HEAD_DIM as usize {
                        o[j] = o[j] * factor + exp_score * unsafe { *values.add(j) }.to_f32().unwrap();
                    }
                }
            }

            // Write the output
            for j in 0..HEAD_DIM as usize {
                unsafe {
                    *out.add(j) = T::from(o[j] / sum_exp_score).unwrap();
                }
            }
        }
    }
}
