use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(BuildTreeOut)]
#[variants(T, f32, bf16)]
#[variants(USE_MXU, false, true)]
pub fn build_tree_out<T: ArrayElement + Float, const USE_MXU: bool>(
    q: *const T,
    prefix: *const f32,
    qkd: *const f32,
    u: *const T,
    #[optional(use_h0)] h0: Option<*const T>,
    #[optional(use_h0)] h0_indices: Option<*const i32>,
    o: *mut T,
    scale: f32,
    batch_size: u32,
    tree_size: u32,
    qk_heads: u32,
    value_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    #[allow(unused)]
    #[specialize]
    use_h0: bool,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let qk_heads = qk_heads as usize;
    let value_heads = value_heads as usize;
    let head_k_dim = head_k_dim as usize;
    let head_v_dim = head_v_dim as usize;
    let value_heads_per_qk_head = value_heads / qk_heads;
    let h0 = h0.unwrap_or(std::ptr::null());
    let h0_indices = h0_indices.unwrap_or(std::ptr::null());

    for batch in 0..batch_size {
        let h0_index = if use_h0 {
            unsafe { *h0_indices.add(batch) }
        } else {
            -1
        };
        for value_head in 0..value_heads {
            let qk_head = value_head / value_heads_per_qk_head;
            let q_head_base = (batch * tree_size * qk_heads + qk_head) * head_k_dim;
            let prefix_base = batch * tree_size * value_heads + value_head;
            let qkd_base = (batch * value_heads + value_head) * tree_size * tree_size;
            let u_base = (batch * value_heads + value_head) * tree_size * head_v_dim;
            let out_base = ((batch * tree_size) * value_heads + value_head) * head_v_dim;

            for row in 0..tree_size {
                let q_row = q_head_base + row * qk_heads * head_k_dim;
                for value_col in 0..head_v_dim {
                    let mut acc = 0.0f32;

                    if use_h0 && h0_index >= 0 {
                        let h0_base =
                            ((h0_index as usize * value_heads + value_head) * head_v_dim + value_col) * head_k_dim;
                        let mut dot = 0.0f32;
                        for dim in 0..head_k_dim {
                            let qv = unsafe { (*q.add(q_row + dim)).to_f32().unwrap() };
                            let hv = unsafe { (*h0.add(h0_base + dim)).to_f32().unwrap() };
                            dot += qv * hv;
                        }
                        let prefix_row = unsafe { *prefix.add(prefix_base + row * value_heads) };
                        acc += prefix_row.exp() * scale * dot;
                    }

                    for col in 0..tree_size {
                        let qkd_value = unsafe { *qkd.add(qkd_base + row * tree_size + col) };
                        let u_value = unsafe { (*u.add(u_base + col * head_v_dim + value_col)).to_f32().unwrap() };
                        acc += qkd_value * u_value;
                    }

                    unsafe {
                        *o.add(out_base + row * value_heads * head_v_dim + value_col) = T::from(acc).unwrap();
                    }
                }
            }
        }
    }
}
