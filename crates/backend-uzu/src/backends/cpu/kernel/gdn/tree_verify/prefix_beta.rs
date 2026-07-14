use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::common::gpu_types::{ActivationType, trie::TrieNode},
};

#[kernel(BuildPrefixBeta)]
#[variants(T, f32, f16, bf16)]
pub fn build_prefix_beta<T: ArrayElement + Float>(
    trie: *const TrieNode,
    a_transposed: *const T,
    b: *const T,
    a_log: *const f32,
    dt_bias: *const f32,
    prefix: *mut f32,
    beta: *mut f32,
    batch_size: u32,
    tree_size: u32,
    value_heads: u32,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let value_heads = value_heads as usize;

    for batch in 0..batch_size {
        let trie_batch = batch * tree_size;
        let batch_offset = batch * tree_size * value_heads;

        for row in 0..tree_size {
            let row_idx = row as u32;
            for head in 0..value_heads {
                let out_idx = batch_offset + row * value_heads + head;
                let b_val = unsafe { *b.add(out_idx) }.to_f32().unwrap();
                unsafe { *beta.add(out_idx) = 1.0 / (1.0 + (-b_val).exp()) };

                let dt = unsafe { *dt_bias.add(head) };
                let scale = unsafe { *a_log.add(head) }.exp();

                let mut sum = 0.0f32;
                for col in 0..tree_size {
                    // col is an ancestor-or-self of row iff row falls in col's
                    // DFS subtree interval [trie_start, trie_end].
                    let node = unsafe { *trie.add(trie_batch + col) };
                    if row_idx < node.trie_start || row_idx > node.trie_end {
                        continue;
                    }
                    let a_val = unsafe { *a_transposed.add(batch_offset + head * tree_size + col) }.to_f32().unwrap();
                    let sp = ActivationType::SOFTPLUS.activate(a_val + dt);
                    sum -= scale * sp;
                }
                unsafe { *prefix.add(out_idx) = sum };
            }
        }
    }
}
