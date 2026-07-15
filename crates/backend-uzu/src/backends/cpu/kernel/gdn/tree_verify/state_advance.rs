use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(StateAdvance)]
#[variants(T, f32, bf16)]
#[variants(HEAD_K_DIM, 128)]
pub fn state_advance<T: ArrayElement + Float, const HEAD_K_DIM: u32>(
    k_norm: *const T,
    v: *const T,
    log_decay_buf: *const f32,
    beta_buf: *const f32,
    accepted_indices: *const u32,
    state: *mut f32,
    accepted_len: u32,
    num_v_heads: u32,
    num_k_heads: u32,
) {
    let accepted_len = accepted_len as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_k_dim = HEAD_K_DIM as usize;
    let head_v_dim = head_k_dim;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let v_heads_per_k_head = num_v_heads / num_k_heads;

    for hv_idx in 0..num_v_heads {
        let hk_idx = hv_idx / v_heads_per_k_head;
        for dv_idx in 0..head_v_dim {
            let state_row_offset = (hv_idx * head_v_dim + dv_idx) * head_k_dim;
            for accepted_idx in 0..accepted_len {
                let tree_idx = unsafe { *accepted_indices.add(accepted_idx) } as usize;
                let tree_head_offset = tree_idx * num_v_heads + hv_idx;
                let decay = unsafe { *log_decay_buf.add(tree_head_offset) }.exp();
                let beta = unsafe { *beta_buf.add(tree_head_offset) };
                let k_offset = tree_idx * key_dim + hk_idx * head_k_dim;

                let mut kv_mem = 0.0;
                for dk_idx in 0..head_k_dim {
                    let state_element_ptr = unsafe { state.add(state_row_offset + dk_idx) };
                    unsafe { *state_element_ptr *= decay };
                    kv_mem += unsafe { *state_element_ptr * (*k_norm.add(k_offset + dk_idx)).to_f32().unwrap() };
                }
                let v_value =
                    unsafe { (*v.add(tree_idx * value_dim + hv_idx * head_v_dim + dv_idx)).to_f32().unwrap() };
                let delta = beta * (v_value - kv_mem);
                for dk_idx in 0..head_k_dim {
                    unsafe {
                        *state.add(state_row_offset + dk_idx) +=
                            (*k_norm.add(k_offset + dk_idx)).to_f32().unwrap() * delta
                    };
                }
            }
        }
    }
}
