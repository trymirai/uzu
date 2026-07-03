use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::trie::TrieNode};

const BLOCK: usize = 16;

#[kernel(BuildTreeGram)]
#[variants(T, f32, bf16)]
#[variants(USE_MXU, false)]
pub fn build_tree_gram<T: ArrayElement + Float, const USE_MXU: bool>(
    q: *const T,
    k: *const T,
    trie: *const TrieNode,
    prefix: *const f32,
    beta: *const f32,
    #[optional(use_h0)] h0: Option<*const T>,
    #[optional(use_h0)] h0_idx: Option<*const i32>,
    a_packed: *mut f32,
    qkd: *mut f32,
    a_inv: *mut f32,
    #[optional(use_h0)] kh0: Option<*mut f32>,
    scale: f32,
    batch_size: u32,
    tree_size: u32,
    k_heads: u32,
    value_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    #[specialize] use_h0: bool,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let k_heads = k_heads as usize;
    let value_heads = value_heads as usize;
    let head_k_dim = head_k_dim as usize;
    let head_v_dim = head_v_dim as usize;

    let value_heads_per_key_head = value_heads / k_heads;
    let num_blocks = tree_size.div_ceil(BLOCK);
    let num_col_pairs = num_blocks.div_ceil(2);

    for batch in 0..batch_size {
        for hv in 0..value_heads {
            let hk = hv / value_heads_per_key_head;

            // qkd[row, col] = scale * exp(prefix[row] - prefix[col]) * dot(q[row], k[col])
            // for ancestor-or-self, else 0. Dense [T, T].
            let mat_base = (batch * value_heads + hv) * tree_size * tree_size;
            for row in 0..tree_size {
                let prefix_row = unsafe { *prefix.add((batch * tree_size + row) * value_heads + hv) };
                let row_off = ((batch * tree_size + row) * k_heads + hk) * head_k_dim;
                for col in 0..tree_size {
                    let out = mat_base + row * tree_size + col;
                    let node = unsafe { *trie.add(batch * tree_size + col) };
                    if (row as u32) < node.trie_start || (row as u32) > node.trie_end {
                        unsafe { *qkd.add(out) = 0.0 };
                        continue;
                    }
                    let col_off = ((batch * tree_size + col) * k_heads + hk) * head_k_dim;
                    let mut qk = 0.0f32;
                    for d in 0..head_k_dim {
                        qk += unsafe { (*q.add(row_off + d)).to_f32().unwrap() }
                            * unsafe { (*k.add(col_off + d)).to_f32().unwrap() };
                    }
                    let prefix_col = unsafe { *prefix.add((batch * tree_size + col) * value_heads + hv) };
                    unsafe { *qkd.add(out) = (prefix_row - prefix_col).exp() * scale * qk };
                }
            }

            // A[row, col] = beta[row] * exp(prefix[row] - prefix[col]) * dot(k[row], k[col])
            // for proper ancestors, else 0; packed [NB, ceil(NB/2), 16, 32] tiles,
            // only pairs touching the block lower triangle (same tiles the GPU writes).
            let a_base = (batch * value_heads + hv) * num_blocks * num_col_pairs * BLOCK * 2 * BLOCK;
            for block in 0..num_blocks {
                for pair in 0..=block / 2 {
                    let tile_base = a_base + (block * num_col_pairs + pair) * BLOCK * 2 * BLOCK;
                    for local_row in 0..BLOCK {
                        for local_col in 0..2 * BLOCK {
                            let row = block * BLOCK + local_row;
                            let col = pair * 2 * BLOCK + local_col;
                            let mut value = 0.0f32;
                            if row != col && row < tree_size && col < tree_size {
                                let node = unsafe { *trie.add(batch * tree_size + col) };
                                if (row as u32) >= node.trie_start && (row as u32) <= node.trie_end {
                                    let row_off = ((batch * tree_size + row) * k_heads + hk) * head_k_dim;
                                    let col_off = ((batch * tree_size + col) * k_heads + hk) * head_k_dim;
                                    let mut kk = 0.0f32;
                                    for d in 0..head_k_dim {
                                        kk += unsafe { (*k.add(row_off + d)).to_f32().unwrap() }
                                            * unsafe { (*k.add(col_off + d)).to_f32().unwrap() };
                                    }
                                    let prefix_row =
                                        unsafe { *prefix.add((batch * tree_size + row) * value_heads + hv) };
                                    let prefix_col =
                                        unsafe { *prefix.add((batch * tree_size + col) * value_heads + hv) };
                                    let beta_row = unsafe { *beta.add((batch * tree_size + row) * value_heads + hv) };
                                    value = beta_row * (prefix_row - prefix_col).exp() * kk;
                                }
                            }
                            unsafe { *a_packed.add(tile_base + local_row * 2 * BLOCK + local_col) = value };
                        }
                    }
                }
            }

            // a_inv = (I + A_diag)^-1 per block, compact [NB, 16, 16], identity-padded;
            // A is f32 so the diagonal values are read back from a_packed unrounded.
            for block in 0..num_blocks {
                let block_size = BLOCK.min(tree_size - block * BLOCK);
                let block_base = ((batch * value_heads + hv) * num_blocks + block) * BLOCK * BLOCK;
                let diag_tile = a_base + (block * num_col_pairs + block / 2) * BLOCK * 2 * BLOCK;
                let diag_col = (block % 2) * BLOCK;
                for row in 0..BLOCK {
                    for col in 0..BLOCK {
                        unsafe {
                            *a_inv.add(block_base + row * BLOCK + col) = if row == col {
                                1.0
                            } else {
                                0.0
                            }
                        };
                    }
                }
                for row in 0..block_size {
                    for col in 0..row {
                        let mut sum = 0.0f32;
                        for prev_row in col..row {
                            sum += unsafe { *a_packed.add(diag_tile + row * 2 * BLOCK + diag_col + prev_row) }
                                * unsafe { *a_inv.add(block_base + prev_row * BLOCK + col) };
                        }
                        unsafe { *a_inv.add(block_base + row * BLOCK + col) = -sum };
                    }
                }
            }

            // kh0[b, token, hv, dv] = dot(k[b, token, hk], h0[slot, hv, dv])
            if !use_h0 {
                continue;
            }
            let h0 = h0.unwrap();
            let h0_idx = h0_idx.unwrap();
            let kh0 = kh0.unwrap();
            let h0_slot = unsafe { *h0_idx.add(batch) };
            if h0_slot >= 0 {
                let h0_head = ((h0_slot as usize) * value_heads + hv) * head_v_dim * head_k_dim;
                for token in 0..tree_size {
                    let k_off = ((batch * tree_size + token) * k_heads + hk) * head_k_dim;
                    let kh0_off = ((batch * tree_size + token) * value_heads + hv) * head_v_dim;
                    for dv in 0..head_v_dim {
                        let mut sum = 0.0f32;
                        for d in 0..head_k_dim {
                            sum += unsafe { (*k.add(k_off + d)).to_f32().unwrap() }
                                * unsafe { (*h0.add(h0_head + dv * head_k_dim + d)).to_f32().unwrap() };
                        }
                        unsafe { *kh0.add(kh0_off + dv) = sum };
                    }
                }
            }
        }
    }
}
