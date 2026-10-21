use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{backends::common::gpu_types::trie::TrieNode, data_type::DataTypeRepr};

#[kernel(TreeVerify)]
#[variants(T, f32, bf16)]
#[variants(NB, 1, 2)]
pub fn tree_verify<T: DataTypeRepr + Float, const NB: u32>(
    q: *const T,
    k: *const T,
    v: *const T,
    trie: *const TrieNode,
    log_decay: *const f32,
    beta: *const f32,
    h0: *const f32,
    in_proj: *const T,
    norm_weight: *const f32,
    o: *mut T,
    scale: f32,
    norm_epsilon: f32,
    batch_size: u32,
    tree_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    conv_dim: u32,
    total_proj_dim: u32,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let num_k_heads = num_k_heads as usize;
    let num_v_heads = num_v_heads as usize;
    let head_k_dim = head_k_dim as usize;
    let head_v_dim = head_v_dim as usize;
    assert!(tree_size <= 16 * NB as usize);

    let load = |ptr: *const T, index: usize| unsafe { (*ptr.add(index)).to_f32().unwrap() };
    for batch in 0..batch_size {
        for hv_idx in 0..num_v_heads {
            let hk_idx = hv_idx / (num_v_heads / num_k_heads);
            let qk_off = |token: usize| ((batch * tree_size + token) * num_k_heads + hk_idx) * head_k_dim;
            let scalar = |token: usize| (batch * tree_size + token) * num_v_heads + hv_idx;
            let nodes: Vec<TrieNode> = (0..tree_size).map(|c| unsafe { *trie.add(batch * tree_size + c) }).collect();
            let in_subtree =
                |row: usize, col: usize| (row as u32) >= nodes[col].trie_start && (row as u32) <= nodes[col].trie_end;

            let prefix: Vec<f32> = (0..tree_size)
                .map(|row| {
                    (0..tree_size).filter(|&c| in_subtree(row, c)).map(|c| unsafe { *log_decay.add(scalar(c)) }).sum()
                })
                .collect();
            let dot =
                |a: &dyn Fn(usize) -> f32, b: &dyn Fn(usize) -> f32| (0..head_k_dim).map(|d| a(d) * b(d)).sum::<f32>();
            let mut a = vec![0.0f32; tree_size * tree_size];
            let mut qkd = vec![0.0f32; tree_size * tree_size];
            for row in 0..tree_size {
                for col in 0..tree_size {
                    if !in_subtree(row, col) {
                        continue;
                    }
                    let decay = (prefix[row] - prefix[col]).exp();
                    qkd[row * tree_size + col] =
                        scale * decay * dot(&|d| load(q, qk_off(row) + d), &|d| load(k, qk_off(col) + d));
                    if row != col {
                        let kk = dot(&|d| load(k, qk_off(row) + d), &|d| load(k, qk_off(col) + d));
                        a[row * tree_size + col] = unsafe { *beta.add(scalar(row)) } * decay * kk;
                    }
                }
            }

            let h0_base = (batch * num_v_heads + hv_idx) * head_v_dim * head_k_dim;
            for dv in 0..head_v_dim {
                let h0_row = |d: usize| unsafe { *h0.add(h0_base + dv * head_k_dim + d) };
                let mut u = vec![0.0f32; tree_size];
                for row in 0..tree_size {
                    let kh0 = dot(&|d| load(k, qk_off(row) + d), &h0_row);
                    let v_value = load(v, scalar(row) * head_v_dim + dv);
                    let rhs = unsafe { *beta.add(scalar(row)) } * (v_value - prefix[row].exp() * kh0);
                    u[row] = rhs - (0..row).map(|c| a[row * tree_size + c] * u[c]).sum::<f32>();
                }
                for row in 0..tree_size {
                    let qh0 = dot(&|d| load(q, qk_off(row) + d), &h0_row);
                    let value = prefix[row].exp() * scale * qh0
                        + (0..tree_size).map(|c| qkd[row * tree_size + c] * u[c]).sum::<f32>();
                    unsafe { *o.add(scalar(row) * head_v_dim + dv) = T::from(value).unwrap() };
                }
            }
        }
    }
    super::super::norm_gate::delta_net_norm_gate::<T>(
        o,
        in_proj,
        norm_weight,
        num_v_heads as u32,
        head_v_dim as u32,
        (num_v_heads * head_v_dim) as u32,
        conv_dim,
        total_proj_dim,
        norm_epsilon,
        (batch_size * tree_size) as u32,
    );
}
