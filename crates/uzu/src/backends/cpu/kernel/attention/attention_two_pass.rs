use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

const TOTAL_BLOCKS_COUNT: u32 = 32;
const SEQUENCE_BLOCK_SIZE_1: u32 = 8;

#[kernel(AttentionTwoPass1)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_two_pass1<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    keys: *const T,
    values: *const T,
    out: *mut f32,
    sums: *mut f32,
    maxs: *mut f32,
    gqa_factor: u32,
    sequence_length: u32,
    k_head_stride: u32,
    k_seq_stride: u32,
    v_head_stride: u32,
    v_seq_stride: u32,
    scale: f32,
    num_heads: u32,
    suffix_length: u32,
    #[optional(float_mask)] fmask: Option<*const T>,
    #[optional(has_mask)] mask_kv_seq_stride: Option<u32>,
    #[optional(has_mask)] mask_q_seq_stride: Option<u32>,
    #[optional(has_mask)] mask_head_stride: Option<u32>,
    #[optional(has_sinks)] sinks: Option<*const f32>,
    #[specialize] float_mask: bool,
    #[specialize] has_mask: bool,
    #[specialize] has_sinks: bool,
    #[specialize] do_causal: bool,
) {
    let value_dim = HEAD_DIM;

    for head_idx in 0..num_heads {
        for q_seq_idx in 0..suffix_length {
            for block_idx in 0..TOTAL_BLOCKS_COUNT {
                let o_offset = q_seq_idx * num_heads + head_idx;
                let q_offset = head_idx * suffix_length + q_seq_idx;
                let kv_head_idx = head_idx / gqa_factor;

                let queries_base: *const T = unsafe { queries.add((q_offset * HEAD_DIM) as usize) };
                let keys_base: *const T = unsafe { keys.add((kv_head_idx * k_head_stride) as usize) };
                let values_base: *const T = unsafe { values.add((kv_head_idx * v_head_stride) as usize) };
                let out_base: *mut f32 =
                    unsafe { out.add((o_offset * TOTAL_BLOCKS_COUNT * value_dim + block_idx * value_dim) as usize) };

                let fmask_base: Option<*const T> = if float_mask {
                    let offset = head_idx * mask_head_stride.unwrap_or(0) + q_seq_idx * mask_q_seq_stride.unwrap_or(0);
                    Some(unsafe { fmask.unwrap().add(offset as usize) })
                } else {
                    None
                };

                // Read the query and scale
                let mut q = vec![0.0f32; HEAD_DIM as usize];
                for j in 0..HEAD_DIM as usize {
                    q[j] = scale * unsafe { *queries_base.add(j) }.to_f32().unwrap();
                }

                let mut o = vec![0.0f32; HEAD_DIM as usize];
                let mut max_score = -1e9f32;
                let mut sum_exp_score = 0.0f32;

                if has_sinks && block_idx == 0 {
                    let q_head_idx = head_idx % num_heads;
                    max_score = unsafe { *sinks.unwrap().add(q_head_idx as usize) };
                    sum_exp_score = 1.0;
                }

                // Iterate over sequence positions assigned to this block
                let mut chunk_start = block_idx * SEQUENCE_BLOCK_SIZE_1;
                while chunk_start < sequence_length {
                    for t in 0..SEQUENCE_BLOCK_SIZE_1 {
                        let i = chunk_start + t;
                        if i >= sequence_length {
                            break;
                        }

                        let use_key = !do_causal || i <= (sequence_length - suffix_length + q_seq_idx);

                        if use_key {
                            let keys_ptr = unsafe { keys_base.add((i * k_seq_stride) as usize) };

                            // Compute the i-th score
                            let mut score = 0.0f32;
                            for j in 0..HEAD_DIM as usize {
                                score += q[j] * unsafe { *keys_ptr.add(j) }.to_f32().unwrap();
                            }
                            if float_mask {
                                let fmask_ptr =
                                    unsafe { fmask_base.unwrap().add((i * mask_kv_seq_stride.unwrap()) as usize) };
                                score += f32::max(-1e9, unsafe { *fmask_ptr }.to_f32().unwrap());
                            }

                            // Update the accumulators
                            let new_max = f32::max(max_score, score);
                            let factor = (max_score - new_max).exp();
                            let exp_score = (score - new_max).exp();

                            max_score = new_max;
                            sum_exp_score = sum_exp_score * factor + exp_score;

                            // Update the output accumulator
                            let values_ptr = unsafe { values_base.add((i * v_seq_stride) as usize) };
                            for j in 0..HEAD_DIM as usize {
                                o[j] = o[j] * factor + exp_score * unsafe { *values_ptr.add(j) }.to_f32().unwrap();
                            }
                        }
                    }
                    chunk_start += TOTAL_BLOCKS_COUNT * SEQUENCE_BLOCK_SIZE_1;
                }

                // Write partial output, sum, and max for this block
                for j in 0..HEAD_DIM as usize {
                    unsafe {
                        *out_base.add(j) = o[j];
                    }
                }
                unsafe {
                    *sums.add((o_offset * TOTAL_BLOCKS_COUNT + block_idx) as usize) = sum_exp_score;
                    *maxs.add((o_offset * TOTAL_BLOCKS_COUNT + block_idx) as usize) = max_score;
                }
            }
        }
    }
}

#[kernel(AttentionTwoPass2)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_two_pass2<T: ArrayElement + Float, const HEAD_DIM: u32>(
    partials: *const f32,
    sums: *const f32,
    maxs: *const f32,
    out: *mut T,
    num_heads: u32,
    suffix_length: u32,
) {
    for head_idx in 0..num_heads {
        for q_seq_idx in 0..suffix_length {
            let o_offset = q_seq_idx * num_heads + head_idx;

            // Find global max across all blocks
            let mut global_max = f32::NEG_INFINITY;
            for block_idx in 0..TOTAL_BLOCKS_COUNT {
                let max_val = unsafe { *maxs.add((o_offset * TOTAL_BLOCKS_COUNT + block_idx) as usize) };
                global_max = f32::max(global_max, max_val);
            }

            // Compute global sum of exponentials with correction
            let mut global_sum = 0.0f32;
            for block_idx in 0..TOTAL_BLOCKS_COUNT {
                let sum_val = unsafe { *sums.add((o_offset * TOTAL_BLOCKS_COUNT + block_idx) as usize) };
                let max_val = unsafe { *maxs.add((o_offset * TOTAL_BLOCKS_COUNT + block_idx) as usize) };
                global_sum += sum_val * (max_val - global_max).exp();
            }

            // Combine partial outputs
            let out_ptr = unsafe { out.add((o_offset * HEAD_DIM) as usize) };
            for j in 0..HEAD_DIM as usize {
                let mut val = 0.0f32;
                for block_idx in 0..TOTAL_BLOCKS_COUNT {
                    let partial_val = unsafe {
                        *partials.add((o_offset * TOTAL_BLOCKS_COUNT * HEAD_DIM + block_idx * HEAD_DIM) as usize + j)
                    };
                    let max_val = unsafe { *maxs.add((o_offset * TOTAL_BLOCKS_COUNT + block_idx) as usize) };
                    val += partial_val * (max_val - global_max).exp();
                }
                unsafe {
                    *out_ptr.add(j) = T::from(val / global_sum).unwrap();
                }
            }
        }
    }
}
