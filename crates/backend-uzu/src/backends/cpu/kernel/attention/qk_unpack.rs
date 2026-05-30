use half::{bf16, f16};
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(QkUnpack)]
#[variants(T, f32, f16, bf16)]
pub fn qk_unpack<T: ArrayElement>(
    qkv: *const T,
    unpacked_queries: *mut T,
    unpacked_keys: *mut T,
    head_dim: u32,
    num_heads: u32,
    num_groups: u32,
    suffix_length: u32,
) {
    if num_groups == 0 || num_heads % num_groups != 0 {
        return;
    }

    let head_dim = head_dim as usize;
    let num_heads = num_heads as usize;
    let num_groups = num_groups as usize;
    let suffix_length = suffix_length as usize;

    let total_heads = num_heads + 2 * num_groups;
    let heads_per_group = num_heads / num_groups;

    for head_index in 0..num_heads {
        let group_index = head_index / heads_per_group;
        let first_head_in_group = group_index * heads_per_group;

        for token_index in 0..suffix_length {
            for dim_index in 0..head_dim {
                let q_val =
                    unsafe { *qkv.add(token_index * total_heads * head_dim + head_index * head_dim + dim_index) };
                unsafe {
                    *unpacked_queries.add(head_index * suffix_length * head_dim + token_index * head_dim + dim_index) =
                        q_val;
                }

                if head_index == first_head_in_group {
                    let key_head_index = num_heads + group_index;
                    let k_val = unsafe {
                        *qkv.add(token_index * total_heads * head_dim + key_head_index * head_dim + dim_index)
                    };
                    unsafe {
                        *unpacked_keys
                            .add(group_index * suffix_length * head_dim + token_index * head_dim + dim_index) = k_val;
                    }
                }
            }
        }
    }
}
