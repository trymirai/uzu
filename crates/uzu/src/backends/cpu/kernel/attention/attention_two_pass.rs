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
    #[allow(unused)] queries: *const T,
    #[allow(unused)] keys: *const T,
    #[allow(unused)] values: *const T,
    #[allow(unused)] out: *mut f32,
    #[allow(unused)] sums: *mut f32,
    #[allow(unused)] maxs: *mut f32,
    #[allow(unused)] gqa_factor: u32,
    #[allow(unused)] sequence_length: u32,
    #[allow(unused)] k_head_stride: u32,
    #[allow(unused)] k_seq_stride: u32,
    #[allow(unused)] v_head_stride: u32,
    #[allow(unused)] v_seq_stride: u32,
    #[allow(unused)] scale: f32,
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] suffix_length: u32,
    #[allow(unused)]
    #[optional(float_mask)]
    fmask: Option<*const T>,
    #[allow(unused)]
    #[optional(has_mask)]
    mask_kv_seq_stride: Option<u32>,
    #[allow(unused)]
    #[optional(has_mask)]
    mask_q_seq_stride: Option<u32>,
    #[allow(unused)]
    #[optional(has_mask)]
    mask_head_stride: Option<u32>,
    #[allow(unused)]
    #[optional(has_sinks)]
    sinks: Option<*const f32>,
    #[allow(unused)]
    #[specialize]
    float_mask: bool,
    #[allow(unused)]
    #[specialize]
    has_mask: bool,
    #[allow(unused)]
    #[specialize]
    has_sinks: bool,
    #[allow(unused)]
    #[specialize]
    do_causal: bool,
) {
    let hd = HEAD_DIM as usize;

    for head_idx in 0..num_heads {
        let kv_head_idx = head_idx / gqa_factor;

        for q_seq_idx in 0..suffix_length {
            let o_offset = (q_seq_idx * num_heads + head_idx) as usize;
            let q_offset = (head_idx * suffix_length + q_seq_idx) as usize;

            for block_idx in 0..TOTAL_BLOCKS_COUNT {
                let mut max_score: f32 = -1e9;
                let mut sum_exp_score: f32 = 0.0;
                let mut o_acc = vec![0.0f32; hd];

                if has_sinks && block_idx == 0 {
                    let q_head_idx = head_idx % num_heads;
                    max_score = unsafe { *sinks.unwrap().add(q_head_idx as usize) };
                    sum_exp_score = 1.0;
                }

                let mut kv_pos =
                    block_idx * SEQUENCE_BLOCK_SIZE_1;

                while kv_pos < sequence_length {
                    for inner in 0..SEQUENCE_BLOCK_SIZE_1 {
                        let i = kv_pos + inner;
                        if i >= sequence_length {
                            break;
                        }

                        let mut use_key = true;
                        if do_causal {
                            use_key =
                                i <= (sequence_length - suffix_length + q_seq_idx);
                        }

                        if use_key {
                            // QK dot product
                            let mut score: f32 = 0.0;
                            for d in 0..hd {
                                let q_val = unsafe {
                                    (*queries.add(q_offset * hd + d)).to_f32().unwrap()
                                };
                                let k_val = unsafe {
                                    (*keys.add(
                                        kv_head_idx as usize * k_head_stride as usize
                                            + i as usize * k_seq_stride as usize
                                            + d,
                                    ))
                                    .to_f32()
                                    .unwrap()
                                };
                                score += q_val * k_val;
                            }
                            score *= scale;

                            if float_mask {
                                let mask_val = unsafe {
                                    (*fmask.unwrap().add(
                                        head_idx as usize
                                            * mask_head_stride.unwrap() as usize
                                            + i as usize
                                                * mask_kv_seq_stride.unwrap() as usize
                                            + q_seq_idx as usize
                                                * mask_q_seq_stride.unwrap() as usize,
                                    ))
                                    .to_f32()
                                    .unwrap()
                                };
                                score += mask_val.max(-1e9);
                            }

                            // Online softmax update
                            let new_max = max_score.max(score);
                            let factor = (max_score - new_max).exp();
                            let exp_score = (score - new_max).exp();

                            max_score = new_max;
                            sum_exp_score = sum_exp_score * factor + exp_score;

                            for d in 0..hd {
                                let v_val = unsafe {
                                    (*values.add(
                                        kv_head_idx as usize * v_head_stride as usize
                                            + i as usize * v_seq_stride as usize
                                            + d,
                                    ))
                                    .to_f32()
                                    .unwrap()
                                };
                                o_acc[d] = o_acc[d] * factor + exp_score * v_val;
                            }
                        }
                    }
                    kv_pos += TOTAL_BLOCKS_COUNT * SEQUENCE_BLOCK_SIZE_1;
                }

                // Write partial results
                let out_base =
                    o_offset * TOTAL_BLOCKS_COUNT as usize * hd + block_idx as usize * hd;
                for d in 0..hd {
                    unsafe { *out.add(out_base + d) = o_acc[d] };
                }
                let sm_base = o_offset * TOTAL_BLOCKS_COUNT as usize + block_idx as usize;
                unsafe {
                    *sums.add(sm_base) = sum_exp_score;
                    *maxs.add(sm_base) = max_score;
                };
            }
        }
    }
}

#[kernel(AttentionTwoPass2)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_two_pass2<T: ArrayElement + Float, const HEAD_DIM: u32>(
    #[allow(unused)] partials: *const f32,
    #[allow(unused)] sums: *const f32,
    #[allow(unused)] maxs: *const f32,
    #[allow(unused)] out: *mut T,
    #[allow(unused)] num_heads: u32,
    #[allow(unused)] suffix_length: u32,
) {
    let hd = HEAD_DIM as usize;
    let nb = TOTAL_BLOCKS_COUNT as usize;

    for head_idx in 0..num_heads {
        for q_seq_idx in 0..suffix_length {
            let o_offset = (q_seq_idx * num_heads + head_idx) as usize;

            // Find global max across all blocks
            let mut global_max: f32 = f32::NEG_INFINITY;
            for b in 0..nb {
                let m = unsafe { *maxs.add(o_offset * nb + b) };
                if m > global_max {
                    global_max = m;
                }
            }

            // Combine partial sums with rescaling
            let mut total_sum: f32 = 0.0;
            for b in 0..nb {
                let m = unsafe { *maxs.add(o_offset * nb + b) };
                let s = unsafe { *sums.add(o_offset * nb + b) };
                total_sum += s * (m - global_max).exp();
            }

            // Combine partial outputs
            let out_base = o_offset * hd;
            for d in 0..hd {
                let mut acc: f32 = 0.0;
                for b in 0..nb {
                    let m = unsafe { *maxs.add(o_offset * nb + b) };
                    let factor = (m - global_max).exp();
                    let partial =
                        unsafe { *partials.add(o_offset * nb * hd + b * hd + d) };
                    acc += partial * factor;
                }
                let result = acc / total_sum;
                unsafe { *out.add(out_base + d) = T::from(result).unwrap() };
            }
        }
    }
}
