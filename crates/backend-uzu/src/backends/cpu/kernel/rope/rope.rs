use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

fn get_paired_val<T: ArrayElement + Float>(
    values: *const T,
    dim_index: usize,
    rotary_pair_stride: usize,
    token_index: usize,
    total_heads: usize,
    head_dim: usize,
    head_index: usize,
) -> T {
    if dim_index < rotary_pair_stride {
        let value = unsafe {
            *values.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index + rotary_pair_stride)
        };
        T::zero() - value
    } else {
        unsafe {
            *values.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index - rotary_pair_stride)
        }
    }
}

fn rotated_dimension_index(
    dim_index: usize,
    half_rope_dim: usize,
    rotary_pair_stride: usize,
) -> Option<usize> {
    // Gemma 4 proportional RoPE keeps NeoX-style pairs in the full head layout.
    // Example: head_dim=512, rope_dim=128, rotary_pair_stride=256 rotates dims
    // 0..63 and 256..319, while 64..255 and 320..511 pass through unchanged.
    if dim_index < half_rope_dim {
        Some(dim_index)
    } else if dim_index >= rotary_pair_stride && dim_index < rotary_pair_stride + half_rope_dim {
        Some(dim_index - rotary_pair_stride + half_rope_dim)
    } else {
        None
    }
}

#[kernel(Rope)]
#[variants(T, f32, f16, bf16)]
pub fn rope<T: ArrayElement + Float>(
    qkv: *const T,
    token_positions: *const i32,
    inverse_frequencies: *const f32,
    rotated_queries: *mut T,
    rotated_keys: *mut T,
    head_dim: u32,
    rope_dim: u32,
    rotary_pair_stride: u32,
    inverse_frequency_count: u32,
    rope_max_sequence_length: u32,
    rope_attention_scaling_factor: f32,
    num_heads: u32,
    num_groups: u32,
    suffix_length: u32,
) {
    if num_groups == 0
        || head_dim & 1 != 0
        || rope_dim & 1 != 0
        || rope_dim > head_dim
        || (rope_dim != 0 && inverse_frequency_count < rope_dim / 2)
        || rotary_pair_stride < rope_dim / 2
        || rotary_pair_stride + rope_dim / 2 > head_dim
        || num_heads % num_groups != 0
    {
        return;
    }

    let head_dim = head_dim as usize;
    let rope_dim = rope_dim as usize;
    let rotary_pair_stride = rotary_pair_stride as usize;
    let num_heads = num_heads as usize;
    let num_groups = num_groups as usize;
    let suffix_length = suffix_length as usize;
    let rope_max_sequence_length = rope_max_sequence_length as usize;

    let half_rope_dim = rope_dim / 2;
    let total_heads = num_heads + 2 * num_groups;
    let heads_per_group = num_heads / num_groups;

    for head_index in 0..num_heads {
        let group_index = head_index / heads_per_group;

        for token_index in 0..suffix_length {
            let raw_position = unsafe { *token_positions.add(token_index) } as usize;
            let absolute_position = if raw_position >= rope_max_sequence_length {
                0
            } else {
                raw_position
            };

            for dim_index in 0..head_dim {
                if let Some(rotary_dim_index) = rotated_dimension_index(dim_index, half_rope_dim, rotary_pair_stride) {
                    let frequency_index = rotary_dim_index % half_rope_dim;
                    let frequency = unsafe { *inverse_frequencies.add(frequency_index) };
                    let angle = absolute_position as f32 * frequency;
                    let cos_val = T::from(angle.cos() * rope_attention_scaling_factor).unwrap();
                    let sin_val = T::from(angle.sin() * rope_attention_scaling_factor).unwrap();

                    let q_val =
                        unsafe { *qkv.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index) };
                    let q_paired = get_paired_val(
                        qkv,
                        dim_index,
                        rotary_pair_stride,
                        token_index,
                        total_heads,
                        head_dim,
                        head_index,
                    );
                    let q_result = q_val * cos_val + q_paired * sin_val;
                    unsafe {
                        *rotated_queries
                            .add(head_index * suffix_length * head_dim + token_index * head_dim + dim_index) = q_result;
                    }

                    if head_index == group_index * heads_per_group {
                        let key_head_index = num_heads + group_index;
                        let k_val = unsafe {
                            *qkv.add(token_index * total_heads * head_dim + key_head_index * head_dim + dim_index)
                        };
                        let k_paired = get_paired_val(
                            qkv,
                            dim_index,
                            rotary_pair_stride,
                            token_index,
                            total_heads,
                            head_dim,
                            key_head_index,
                        );
                        let k_result = k_val * cos_val + k_paired * sin_val;
                        unsafe {
                            *rotated_keys
                                .add(group_index * suffix_length * head_dim + token_index * head_dim + dim_index) =
                                k_result;
                        }
                    }
                } else {
                    let q_val =
                        unsafe { *qkv.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index) };
                    unsafe {
                        *rotated_queries
                            .add(head_index * suffix_length * head_dim + token_index * head_dim + dim_index) = q_val;
                    }

                    if head_index == group_index * heads_per_group {
                        let key_head_index = num_heads + group_index;
                        let k_val = unsafe {
                            *qkv.add(token_index * total_heads * head_dim + key_head_index * head_dim + dim_index)
                        };
                        unsafe {
                            *rotated_keys
                                .add(group_index * suffix_length * head_dim + token_index * head_dim + dim_index) =
                                k_val;
                        }
                    }
                }
            }
        }
    }
}
