use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

/// Solves chunked GDN update coefficients: `(I + A) U = beta * (v - exp(prefix) * kh0)`.
/// `kh0 = k @ h0^T` is precomputed by BuildTreeGram.
#[kernel(TreeUpdateSolve)]
#[variants(T, f32, bf16)]
#[variants(BV, 16, 32)]
#[variants(USE_MXU, false)]
pub fn tree_update_solve<T: ArrayElement + Float, const BV: u32, const USE_MXU: bool>(
    #[optional(use_h0)] kh0: Option<*const T>,
    v: *const T,
    prefix: *const f32,
    beta: *const f32,
    a_packed: *const f32,
    a_inv: *const f32,
    #[optional(use_h0)] h0_idx: Option<*const i32>,
    u: *mut f32,
    batch_size: u32,
    tree_size: u32,
    value_heads: u32,
    head_v_dim: u32,
    #[specialize] use_h0: bool,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let value_heads = value_heads as usize;
    let head_v_dim = head_v_dim as usize;
    let block_size = 16usize;

    if batch_size == 0 || tree_size == 0 || value_heads == 0 || head_v_dim == 0 {
        return;
    }

    let num_blocks = tree_size.div_ceil(block_size);
    let num_col_pairs = num_blocks.div_ceil(2);

    let kh0 = kh0.unwrap_or(std::ptr::null());
    let h0_idx = h0_idx.unwrap_or(std::ptr::null());

    for batch in 0..batch_size {
        let h0_slot = if use_h0 {
            unsafe { *h0_idx.add(batch) }
        } else {
            -1
        };

        for hv in 0..value_heads {
            let batch_value_head_idx = batch * value_heads + hv;

            for block in 0..num_blocks {
                let token_base = block * block_size;
                let mut acc = vec![0.0f32; block_size * head_v_dim];

                for local_token in 0..block_size {
                    let token = token_base + local_token;
                    if token >= tree_size {
                        continue;
                    }

                    let prefix_idx = (batch * tree_size + token) * value_heads + hv;
                    let beta_val = unsafe { *beta.add(prefix_idx) };
                    let decay_from_h0 = unsafe { *prefix.add(prefix_idx) }.exp();

                    for dv in 0..head_v_dim {
                        let v_idx = (((batch * tree_size + token) * value_heads + hv) * head_v_dim) + dv;
                        let v_val = unsafe { (*v.add(v_idx)).to_f32().unwrap() };

                        let kh0_val = if h0_slot >= 0 {
                            unsafe { (*kh0.add(v_idx)).to_f32().unwrap() }
                        } else {
                            0.0
                        };

                        acc[local_token * head_v_dim + dv] = beta_val * (v_val - decay_from_h0 * kh0_val);
                    }
                }

                for prev_token in 0..token_base {
                    let prev_block = prev_token / block_size;
                    let prev_local = prev_token % block_size;
                    for local_token in 0..block_size {
                        let token = token_base + local_token;
                        if token >= tree_size {
                            continue;
                        }

                        // A is packed [B*HV, NB, ceil(NB/2), 16, 32] block-pair tiles.
                        let a_idx = ((batch_value_head_idx * num_blocks + block) * num_col_pairs + prev_block / 2)
                            * (block_size * 2 * block_size)
                            + local_token * (2 * block_size)
                            + (prev_block % 2) * block_size
                            + prev_local;
                        let a_val = unsafe { *a_packed.add(a_idx) };
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
                            // Ainv is compact [B * HV, ceil(T/BT), BT, BT] diagonal blocks.
                            let inv_idx = ((batch_value_head_idx * num_blocks + block) * block_size + local_token)
                                * block_size
                                + local_prev_token;
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
