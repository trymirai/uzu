use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[inline]
fn decode_shearkv_value(
    packed_values: *const u8,
    bits: u32,
    row_bytes: usize,
    dim: usize,
    scale: f32,
    bias: f32,
) -> f32 {
    let _ = row_bytes;
    let quantized = match bits {
        8 => (unsafe { *packed_values.add(dim) }) as f32,
        4 => {
            let byte = unsafe { *packed_values.add(dim / 2) };
            if dim % 2 == 0 {
                (byte & 0x0F) as f32
            } else {
                (byte >> 4) as f32
            }
        },
        _ => panic!("ShearKV kernel only supports 4-bit or 8-bit values"),
    };
    bias + scale * quantized
}

#[kernel(AttentionShearSingleDecode)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_DIM, 64, 128, 256)]
pub fn attention_shearkv_single_decode<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    keys: *const T,
    packed_values: *const u8,
    value_scales: *const f32,
    value_biases: *const f32,
    dense_values: *const T,
    out: *mut T,
    gqa_factor: u32,
    prefix_length: u32,
    sequence_capacity: u32,
    k_head_stride: u32,
    k_seq_stride: u32,
    packed_value_row_bytes: u32,
    scale: f32,
    #[optional(has_sinks)] sinks: Option<*const f32>,
    num_heads: u32,
    bits: u32,
    #[specialize] has_sinks: bool,
) {
    let head_dim = HEAD_DIM as usize;
    let row_bytes = packed_value_row_bytes as usize;
    let packed_value_head_stride = sequence_capacity as usize * row_bytes;
    let dense_value_head_stride = sequence_capacity as usize * HEAD_DIM as usize;

    for head_idx in 0..num_heads {
        let kv_head_idx = head_idx / gqa_factor;
        let query = unsafe { queries.add((head_idx * HEAD_DIM) as usize) };
        let head_keys = unsafe { keys.add((kv_head_idx * k_head_stride) as usize) };
        let head_values = unsafe { packed_values.add(kv_head_idx as usize * packed_value_head_stride) };
        let head_scales = unsafe { value_scales.add(kv_head_idx as usize * sequence_capacity as usize) };
        let head_biases = unsafe { value_biases.add(kv_head_idx as usize * sequence_capacity as usize) };
        let head_dense_values = unsafe { dense_values.add(kv_head_idx as usize * dense_value_head_stride) };
        let output = unsafe { out.add((head_idx * HEAD_DIM) as usize) };

        let mut scaled_query = vec![0.0f32; head_dim];
        let mut accum = vec![0.0f32; head_dim];
        for dim in 0..head_dim {
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

        for token in 0..prefix_length as usize {
            let key = unsafe { head_keys.add(token * k_seq_stride as usize) };
            let mut score = 0.0f32;
            for dim in 0..head_dim {
                score += scaled_query[dim] * unsafe { *key.add(dim) }.to_f32().unwrap();
            }

            let new_max = f32::max(max_score, score);
            let factor = (max_score - new_max).exp();
            let exp_score = (score - new_max).exp();
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            let row_codes = unsafe { head_values.add(token * row_bytes) };
            let row_scale = unsafe { *head_scales.add(token) };
            let row_bias = unsafe { *head_biases.add(token) };
            for dim in 0..head_dim {
                let decoded = decode_shearkv_value(row_codes, bits, row_bytes, dim, row_scale, row_bias);
                accum[dim] = accum[dim] * factor + exp_score * decoded;
            }
        }

        let self_key = unsafe { head_keys.add(prefix_length as usize * k_seq_stride as usize) };
        let mut self_score = 0.0f32;
        for dim in 0..head_dim {
            self_score += scaled_query[dim] * unsafe { *self_key.add(dim) }.to_f32().unwrap();
        }
        let new_max = f32::max(max_score, self_score);
        let factor = (max_score - new_max).exp();
        let exp_score = (self_score - new_max).exp();
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for dim in 0..head_dim {
            let value = unsafe { *head_dense_values.add(prefix_length as usize * head_dim + dim) }.to_f32().unwrap();
            accum[dim] = accum[dim] * factor + exp_score * value;
            unsafe {
                *output.add(dim) = T::from(accum[dim] / sum_exp_score).unwrap();
            }
        }
    }
}
