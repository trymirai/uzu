use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

fn get_paired_val<T: ArrayElement + Float>(
    values: *const T,
    dim_index: usize,
    half_dim: usize,
    token_index: usize,
    total_heads: usize,
    head_dim: usize,
    head_index: usize,
) -> T {
    if dim_index < half_dim {
        let v =
            unsafe { *values.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index + half_dim) };
        T::zero() - v
    } else {
        unsafe { *values.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index - half_dim) }
    }
}

#[kernel(Rope)]
#[variants(T, f32, f16, bf16)]
pub fn rope<T: ArrayElement + Float>(
    qkv: *const T,
    cosines: *const T,
    sines: *const T,
    token_positions: *const i32,
    rotated_queries: *mut T,
    rotated_keys: *mut T,
    head_dim: u32,
    rope_dim: u32,
    num_heads: u32,
    num_groups: u32,
    suffix_length: u32,
    max_sequence_length: u32,
    #[specialize] has_gate: bool,
) {
    if num_groups == 0 || head_dim & 1 != 0 || rope_dim & 1 != 0 || rope_dim > head_dim || num_heads % num_groups != 0 {
        return;
    }

    let head_dim = head_dim as usize;
    let rope_dim = rope_dim as usize;
    let num_heads = num_heads as usize;
    let num_groups = num_groups as usize;
    let suffix_length = suffix_length as usize;
    let max_sequence_length = max_sequence_length as usize;

    let half_rope_dim = rope_dim / 2;
    let gate_heads = if has_gate {
        num_heads
    } else {
        0
    };
    let total_heads = num_heads + 2 * num_groups + gate_heads;
    let heads_per_group = num_heads / num_groups;

    for head_index in 0..num_heads {
        let group_index = head_index / heads_per_group;

        for token_index in 0..suffix_length {
            let raw_position = unsafe { *token_positions.add(token_index) } as usize;
            let absolute_position = if raw_position >= max_sequence_length {
                0
            } else {
                raw_position
            };

            // Rotated dimensions: apply RoPE to dims 0..rope_dim
            for dim_index in 0..rope_dim {
                let cos_val = unsafe { *cosines.add(absolute_position * rope_dim + dim_index) };
                let sin_val = unsafe { *sines.add(absolute_position * rope_dim + dim_index) };

                // Query rotation
                let q_val =
                    unsafe { *qkv.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index) };
                let q_paired =
                    get_paired_val(qkv, dim_index, half_rope_dim, token_index, total_heads, head_dim, head_index);
                let q_result = q_val * cos_val + q_paired * sin_val;
                unsafe {
                    *rotated_queries.add(head_index * suffix_length * head_dim + token_index * head_dim + dim_index) =
                        q_result;
                }

                // Key rotation (only first head in each group)
                if head_index == group_index * heads_per_group {
                    let key_head_index = num_heads + group_index;
                    let k_val = unsafe {
                        *qkv.add(token_index * total_heads * head_dim + key_head_index * head_dim + dim_index)
                    };
                    let k_paired = get_paired_val(
                        qkv,
                        dim_index,
                        half_rope_dim,
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
            }

            // Pass-through dimensions: copy dims rope_dim..head_dim unchanged
            for dim_index in rope_dim..head_dim {
                let q_val =
                    unsafe { *qkv.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index) };
                unsafe {
                    *rotated_queries.add(head_index * suffix_length * head_dim + token_index * head_dim + dim_index) =
                        q_val;
                }

                if head_index == group_index * heads_per_group {
                    let key_head_index = num_heads + group_index;
                    let k_val = unsafe {
                        *qkv.add(token_index * total_heads * head_dim + key_head_index * head_dim + dim_index)
                    };
                    unsafe {
                        *rotated_keys
                            .add(group_index * suffix_length * head_dim + token_index * head_dim + dim_index) = k_val;
                    }
                }
            }
        }
    }
}
