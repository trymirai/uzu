use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::trie::TrieNode};
#[kernel(BuildTreeGram)]
#[variants(T, f32, bf16)]
#[variants(USE_MXU, false)]
pub fn build_tree_gram<T: ArrayElement + Float, const USE_MXU: bool>(
    q: *const T,
    k: *const T,
    trie: *const TrieNode,
    prefix: *const f32,
    beta: *const f32,
    a_mat: *mut f32,
    qkd: *mut f32,
    ainv: *mut f32,
    scale: f32,
    batch_size: u32,
    tree_size: u32,
    k_heads: u32,
    value_heads: u32,
    head_k_dim: u32,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let k_heads = k_heads as usize;
    let value_heads = value_heads as usize;
    let head_k_dim = head_k_dim as usize;

    let groups_per_head = value_heads / k_heads;

    for batch in 0..batch_size {
        for hv in 0..value_heads {
            let hk = hv / groups_per_head;
            let mat_base = (batch * value_heads + hv) * tree_size * tree_size;
            let trie_base = batch * tree_size;

            for i in 0..tree_size {
                let row_idx = i as u32;
                let prefix_i = unsafe { *prefix.add((batch * tree_size + i) * value_heads + hv) };
                let beta_i = unsafe { *beta.add((batch * tree_size + i) * value_heads + hv) };
                let row_off = ((batch * tree_size + i) * k_heads + hk) * head_k_dim;
                for j in 0..tree_size {
                    let out = mat_base + i * tree_size + j;
                    let node = unsafe { *trie.add(trie_base + j) };
                    let incl = row_idx >= node.trie_start && row_idx <= node.trie_end;
                    if !incl {
                        unsafe {
                            *a_mat.add(out) = 0.0;
                            *qkd.add(out) = 0.0;
                        }
                        continue;
                    }
                    let col_off = ((batch * tree_size + j) * k_heads + hk) * head_k_dim;
                    let mut kk = 0.0f32;
                    let mut qk = 0.0f32;
                    for d in 0..head_k_dim {
                        let ki = unsafe { (*k.add(row_off + d)).to_f32().unwrap() };
                        let kj = unsafe { (*k.add(col_off + d)).to_f32().unwrap() };
                        let qi = unsafe { (*q.add(row_off + d)).to_f32().unwrap() };
                        kk += ki * kj;
                        qk += qi * kj;
                    }
                    let prefix_j = unsafe { *prefix.add((batch * tree_size + j) * value_heads + hv) };
                    let dexp = (prefix_i - prefix_j).exp();
                    unsafe {
                        *a_mat.add(out) = if i != j {
                            beta_i * dexp * kk
                        } else {
                            0.0
                        };
                        *qkd.add(out) = dexp * scale * qk;
                    }
                }
            }

            invert_tree_gram_matrix(a_mat, ainv, mat_base, tree_size);
        }
    }
}

fn invert_tree_gram_matrix(
    a_mat: *const f32,
    ainv: *mut f32,
    mat_base: usize,
    tree_size: usize,
) {
    for i in 0..tree_size {
        for j in 0..tree_size {
            unsafe {
                *ainv.add(mat_base + i * tree_size + j) = if i == j {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
    for i in 0..tree_size {
        for j in 0..i {
            let mut s = 0.0f32;
            for kx in j..i {
                unsafe {
                    s += *a_mat.add(mat_base + i * tree_size + kx) * *ainv.add(mat_base + kx * tree_size + j);
                }
            }
            unsafe {
                *ainv.add(mat_base + i * tree_size + j) = -s;
            }
        }
    }
}
