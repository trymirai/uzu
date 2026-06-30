use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

/// Solves chunked GDN update coefficients: `(I + A) U = beta * (v - exp(prefix) * k @ h0)`.
/// `batch_value_head_idx = batch * num_v_heads + hv` indexes A, Ainv, and U.
#[kernel(GdnTreeUpdateSolve)]
#[variants(T, f32, bf16)]
#[variants(HEAD_K_DIM, 128)]
#[variants(BT, 16)]
#[variants(BV, 16)]
pub fn gdn_tree_update_solve<T: ArrayElement + Float, const HEAD_K_DIM: u32, const BT: u32, const BV: u32>(
    k: *const T,
    v: *const T,
    prefix: *const f32,
    beta: *const f32,
    a: *const f32,
    a_inv: *const f32,
    h0: *const T,
    h0_idx: *const i32,
    u: *mut f32,
    batch_size: u32,
    tree_size: u32,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let block_size = BT as usize;
    let head_k_dim = HEAD_K_DIM as usize;

    if batch_size == 0 || tree_size == 0 || num_v_heads == 0 || num_k_heads == 0 || head_v_dim == 0 || block_size == 0 {
        return;
    }

    debug_assert!(num_v_heads.is_multiple_of(num_k_heads));
    let groups_per_head = num_v_heads / num_k_heads;
    let num_blocks = tree_size.div_ceil(block_size);

    for batch in 0..batch_size {
        let h0_slot = unsafe { *h0_idx.add(batch) };

        for hv in 0..num_v_heads {
            let hk = hv / groups_per_head;
            let batch_value_head_idx = batch * num_v_heads + hv;

            for block in 0..num_blocks {
                let token_base = block * block_size;
                let mut acc = vec![0.0f32; block_size * head_v_dim];

                for local_token in 0..block_size {
                    let token = token_base + local_token;
                    if token >= tree_size {
                        continue;
                    }

                    let prefix_idx = (batch * tree_size + token) * num_v_heads + hv;
                    let beta_val = unsafe { *beta.add(prefix_idx) };
                    let decay_from_h0 = unsafe { *prefix.add(prefix_idx) }.exp();

                    for dv in 0..head_v_dim {
                        let v_idx = (((batch * tree_size + token) * num_v_heads + hv) * head_v_dim) + dv;
                        let v_val = unsafe { (*v.add(v_idx)).to_f32().unwrap() };

                        let mut kh0 = 0.0f32;
                        if h0_slot >= 0 {
                            let h0_slot = h0_slot as usize;
                            for dk in 0..head_k_dim {
                                let k_idx = (((batch * tree_size + token) * num_k_heads + hk) * head_k_dim) + dk;
                                let h0_idx = (((h0_slot * num_v_heads + hv) * head_v_dim + dv) * head_k_dim) + dk;
                                kh0 += unsafe { (*k.add(k_idx)).to_f32().unwrap() }
                                    * unsafe { (*h0.add(h0_idx)).to_f32().unwrap() };
                            }
                        }

                        acc[local_token * head_v_dim + dv] = beta_val * (v_val - decay_from_h0 * kh0);
                    }
                }

                for prev_token in 0..token_base {
                    for local_token in 0..block_size {
                        let token = token_base + local_token;
                        if token >= tree_size {
                            continue;
                        }

                        let a_idx = (batch_value_head_idx * tree_size + token) * tree_size + prev_token;
                        let a_val = unsafe { *a.add(a_idx) };
                        for dv in 0..head_v_dim {
                            let u_prev_idx = (batch_value_head_idx * tree_size + prev_token) * head_v_dim + dv;
                            acc[local_token * head_v_dim + dv] -= a_val * unsafe { *u.add(u_prev_idx) };
                        }
                    }
                }

                for local_token in 0..block_size {
                    let token = token_base + local_token;
                    if token >= tree_size {
                        continue;
                    }

                    for dv in 0..head_v_dim {
                        let mut sum = 0.0f32;
                        for local_prev_token in 0..block_size {
                            let source_token = token_base + local_prev_token;
                            if source_token >= tree_size {
                                continue;
                            }
                            let inv_idx = (batch_value_head_idx * tree_size + token) * tree_size + source_token;
                            sum += unsafe { *a_inv.add(inv_idx) } * acc[local_prev_token * head_v_dim + dv];
                        }

                        let u_idx = (batch_value_head_idx * tree_size + token) * head_v_dim + dv;
                        unsafe { *u.add(u_idx) = sum };
                    }
                }
            }
        }
    }
}
