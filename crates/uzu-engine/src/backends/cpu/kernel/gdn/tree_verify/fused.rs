use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::common::gpu_types::{ActivationType, trie::TrieNode},
};

/// The gated delta rule walked along every tree path (DFS order: a node's state is its parent's
/// state advanced by one token), then the gated RMS norm. Same contract as the Metal kernel.
#[kernel(TreeVerifyFused)]
#[variants(T, bf16)]
#[variants(MAX_TREE, 16, 32)]
pub fn tree_verify_fused<T: ArrayElement + Float, const MAX_TREE: u32>(
    q: *const T,
    k: *const T,
    v: *const T,
    trie: *const TrieNode,
    log_decay: *const f32,
    beta: *const f32,
    h0: *const f32,
    in_proj: *const T,
    norm_weight: *const f32,
    out: *mut T,
    tree_size: u32,
    k_heads: u32,
    value_heads: u32,
    conv_dim: u32,
    total_proj_dim: u32,
    norm_epsilon: f32,
) {
    const HEAD_DIM: usize = 128;
    let tree_size = tree_size as usize;
    let k_heads = k_heads as usize;
    let value_heads = value_heads as usize;
    let conv_dim = conv_dim as usize;
    let total_proj_dim = total_proj_dim as usize;
    assert!(tree_size <= MAX_TREE as usize);

    // Parent of a node = its nearest earlier node whose DFS interval contains it.
    let parents: Vec<Option<usize>> = (0..tree_size)
        .map(|node| {
            (0..node).rev().find(|&col| {
                let interval = unsafe { *trie.add(col) };
                interval.trie_start <= node as u32 && node as u32 <= interval.trie_end
            })
        })
        .collect();

    for hv in 0..value_heads {
        let hk = hv / (value_heads / k_heads);
        let mut states: Vec<Vec<f32>> = Vec::with_capacity(tree_size);
        for node in 0..tree_size {
            let mut state = match parents[node] {
                None => unsafe { std::slice::from_raw_parts(h0.add(hv * HEAD_DIM * HEAD_DIM), HEAD_DIM * HEAD_DIM) }
                    .to_vec(),
                Some(parent) => states[parent].clone(),
            };
            let k_row = unsafe { std::slice::from_raw_parts(k.add((node * k_heads + hk) * HEAD_DIM), HEAD_DIM) };
            let q_row = unsafe { std::slice::from_raw_parts(q.add((node * k_heads + hk) * HEAD_DIM), HEAD_DIM) };
            let v_row = unsafe { std::slice::from_raw_parts(v.add((node * value_heads + hv) * HEAD_DIM), HEAD_DIM) };
            let decay = unsafe { *log_decay.add(node * value_heads + hv) }.exp();
            let beta = unsafe { *beta.add(node * value_heads + hv) };
            let mut o = [0.0f32; HEAD_DIM];
            for dv in 0..HEAD_DIM {
                let row = &mut state[dv * HEAD_DIM..][..HEAD_DIM];
                let mut kv_mem = 0.0f32;
                for dk in 0..HEAD_DIM {
                    row[dk] *= decay;
                    kv_mem += row[dk] * k_row[dk].to_f32().unwrap();
                }
                let delta = beta * (v_row[dv].to_f32().unwrap() - kv_mem);
                for dk in 0..HEAD_DIM {
                    row[dk] += delta * k_row[dk].to_f32().unwrap();
                }
                o[dv] = (0..HEAD_DIM).map(|dk| row[dk] * q_row[dk].to_f32().unwrap()).sum();
            }
            let inv_rms = (o.iter().map(|x| x * x).sum::<f32>() / HEAD_DIM as f32 + norm_epsilon).sqrt().recip();
            for dv in 0..HEAD_DIM {
                let z =
                    unsafe { *in_proj.add(node * total_proj_dim + conv_dim + hv * HEAD_DIM + dv) }.to_f32().unwrap();
                let value = o[dv] * inv_rms * unsafe { *norm_weight.add(dv) } * ActivationType::SILU.activate(z);
                unsafe { *out.add((node * value_heads + hv) * HEAD_DIM + dv) = T::from(value).unwrap() };
            }
            states.push(state);
        }
    }
}
