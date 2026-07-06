use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::array::ArrayElement;

fn chunked_g<const RECOMPUTE_G: bool>(
    g_or_log_decay: *const f32,
    num_v_heads: usize,
    hv: usize,
    token_base: usize,
    local_t: usize,
) -> f32 {
    if RECOMPUTE_G {
        let mut acc = 0.0f32;
        for i in 0..=local_t {
            acc += unsafe { *g_or_log_decay.add((token_base + i) * num_v_heads + hv) };
        }
        acc
    } else {
        unsafe { *g_or_log_decay.add((token_base + local_t) * num_v_heads + hv) }
    }
}

// Merged BuildW + BuildU: emits both W = T . diag(beta e^g) . K and
// U = T . diag(beta) . V from the shared block inverse (a_packed / a_inv).
#[kernel(DeltaNetChunkedBuildWU)]
#[variants(T, f32, f16, bf16)]
#[variants(O, f32, bf16)]
#[variants(HEAD_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(BV, 16, 32)]
#[variants(RECOMPUTE_G, false, true)]
#[allow(clippy::too_many_arguments)]
pub fn delta_net_chunked_build_wu<
    T: ArrayElement + Float,
    O: ArrayElement + NumCast,
    const HEAD_DIM: u32,
    const CHUNK_SIZE: u32,
    const BV: u32,
    const RECOMPUTE_G: bool,
>(
    k_norm: *const f32,
    in_proj: *const T,
    beta: *const f32,
    g_or_log_decay: *const f32,
    a_packed: *const f32,
    a_inv: *const f32,
    w_out: *mut O,
    u_out: *mut O,
    num_v_heads: u32,
    num_k_heads: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
) {
    let _ = BV;
    let head_dim = HEAD_DIM as usize;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let num_k_heads = num_k_heads as usize;
    let key_dim = key_dim as usize;
    let value_dim = value_dim as usize;
    let suffix_len = suffix_len as usize;
    let conv_dim = 2 * key_dim + value_dim;
    let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let groups_per_head = num_v_heads / num_k_heads;
    let block = 16usize;
    let num_blocks = chunk_size.div_ceil(block);
    let num_col_pairs = num_blocks.div_ceil(2);

    for chunk in 0..num_chunks {
        let token_base = chunk * chunk_size;
        for hv in 0..num_v_heads {
            let hk = hv / groups_per_head;
            for block_idx in 0..num_blocks {
                let row_base = block_idx * block;
                let mut acc_w = vec![0.0f32; block * head_dim];
                let mut acc_u = vec![0.0f32; block * head_dim];

                for row in 0..block {
                    let token = token_base + row_base + row;
                    if token < suffix_len {
                        let beta_row = unsafe { *beta.add(token * num_v_heads + hv) };
                        let scale = beta_row
                            * chunked_g::<RECOMPUTE_G>(g_or_log_decay, num_v_heads, hv, token_base, row_base + row)
                                .exp();
                        let k_base = token * key_dim + hk * head_dim;
                        for dk in 0..head_dim {
                            acc_w[row * head_dim + dk] = scale * unsafe { *k_norm.add(k_base + dk) };
                        }
                        for dv in 0..head_dim {
                            let v = unsafe {
                                (*in_proj.add(token * total_proj_dim + 2 * key_dim + hv * head_dim + dv))
                                    .to_f32()
                                    .unwrap()
                            };
                            acc_u[row * head_dim + dv] = beta_row * v;
                        }
                    }
                }

                for prev_block in 0..block_idx {
                    for row in 0..block {
                        for prev_row in 0..block {
                            let local_col = (prev_block % 2) * block + prev_row;
                            let a_idx = (((chunk * num_v_heads + hv) * num_blocks + block_idx) * num_col_pairs
                                + prev_block / 2)
                                * (block * 2 * block)
                                + row * (2 * block)
                                + local_col;
                            let a = unsafe { *a_packed.add(a_idx) };
                            let prev_token = prev_block * block + prev_row;
                            let prev_out_base = ((chunk * num_v_heads + hv) * chunk_size + prev_token) * head_dim;
                            for d in 0..head_dim {
                                let w_prev: f32 = NumCast::from(unsafe { *w_out.add(prev_out_base + d) }).unwrap();
                                let u_prev: f32 = NumCast::from(unsafe { *u_out.add(prev_out_base + d) }).unwrap();
                                acc_w[row * head_dim + d] -= a * w_prev;
                                acc_u[row * head_dim + d] -= a * u_prev;
                            }
                        }
                    }
                }

                for row in 0..block {
                    let local_token = row_base + row;
                    let out_base = ((chunk * num_v_heads + hv) * chunk_size + local_token) * head_dim;
                    for d in 0..head_dim {
                        let mut value_w = 0.0f32;
                        let mut value_u = 0.0f32;
                        for source_row in 0..block {
                            let inv_idx = ((chunk * num_v_heads + hv) * num_blocks + block_idx) * block * block
                                + row * block
                                + source_row;
                            let inv = unsafe { *a_inv.add(inv_idx) };
                            value_w += inv * acc_w[source_row * head_dim + d];
                            value_u += inv * acc_u[source_row * head_dim + d];
                        }
                        unsafe {
                            *w_out.add(out_base + d) = NumCast::from(value_w).unwrap();
                            *u_out.add(out_base + d) = NumCast::from(value_u).unwrap();
                        }
                    }
                }
            }
        }
    }
}
