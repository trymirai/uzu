use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildTreeGramKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{assert::assert_eq_float, helpers::allocation_to_vec},
};

fn get_output<B: Backend>(
    q: &[f32],
    k: &[f32],
    trie: &[u32],
    prefix: &[f32],
    beta: &[f32],
    scale: f32,
    batch_size: u32,
    tree_size: u32,
    k_heads: u32,
    value_heads: u32,
    head_k_dim: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::BuildTreeGramKernel::new(&context, DataType::F32)
        .expect("Failed to create BuildTreeGramKernel");

    let output_len = batch_size as usize * value_heads as usize * tree_size as usize * tree_size as usize;
    let q = context.create_array_from(&[q.len()], q);
    let k = context.create_array_from(&[k.len()], k);
    let trie = context.create_array_from(&[trie.len()], trie);
    let prefix = context.create_array_from(&[prefix.len()], prefix);
    let beta = context.create_array_from(&[beta.len()], beta);
    let mut a_mat = context.create_array_uninitialized(&[output_len], DataType::F32).into_allocation();
    let mut qkd = context.create_array_uninitialized(&[output_len], DataType::F32).into_allocation();
    let mut ainv = context.create_array_uninitialized(&[output_len], DataType::F32).into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        q.allocation(),
        k.allocation(),
        trie.allocation(),
        prefix.allocation(),
        beta.allocation(),
        &mut a_mat,
        &mut qkd,
        &mut ainv,
        scale,
        batch_size,
        tree_size,
        k_heads,
        value_heads,
        head_k_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec(&a_mat), allocation_to_vec(&qkd), allocation_to_vec(&ainv))
}

fn build_trie(tree_size: usize) -> Vec<u32> {
    let last = tree_size as u32 - 1;
    let mut trie = Vec::with_capacity(2 * tree_size * 3);
    for i in 0..tree_size as u32 {
        trie.extend_from_slice(&[i, last, i]);
    }
    trie.extend_from_slice(&[0, last, 0]);
    for i in 1..tree_size as u32 {
        trie.extend_from_slice(&[i, i, 1]);
    }
    trie
}

#[uzu_test]
fn test_build_tree_gram_matches_cpu() {
    let batch_size = 2;
    let k_heads = 2;
    let value_heads = 6;
    let head_k_dim = 128;
    let scale = (head_k_dim as f32).sqrt().recip();

    for tree_size in [17, 64] {
        let q_len = batch_size * tree_size * k_heads * head_k_dim;
        let kv_len = batch_size * tree_size * value_heads;
        let trie = build_trie(tree_size);
        let q = (0..q_len).map(|i| ((i as f32 * 0.017).sin() * 0.2) + 0.01).collect::<Vec<_>>();
        let k = (0..q_len).map(|i| ((i as f32 * 0.019).cos() * 0.18) - 0.02).collect::<Vec<_>>();
        let prefix = (0..kv_len)
            .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % value_heads) as f32) * 0.003)
            .collect::<Vec<_>>();
        let beta = (0..kv_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect::<Vec<_>>();

        let expected = get_output::<Cpu>(
            &q,
            &k,
            &trie,
            &prefix,
            &beta,
            scale,
            batch_size as u32,
            tree_size as u32,
            k_heads as u32,
            value_heads as u32,
            head_k_dim as u32,
        );

        for_each_non_cpu_backend!(|B| {
            let actual = get_output::<B>(
                &q,
                &k,
                &trie,
                &prefix,
                &beta,
                scale,
                batch_size as u32,
                tree_size as u32,
                k_heads as u32,
                value_heads as u32,
                head_k_dim as u32,
            );
            let msg = format!("backend {} tree_size {tree_size}", std::any::type_name::<B>());
            assert_eq_float::<f32>(&expected.0, &actual.0, 5e-3, &format!("a_mat {msg}"));
            assert_eq_float::<f32>(&expected.1, &actual.1, 5e-3, &format!("qkd {msg}"));
            assert_eq_float::<f32>(&expected.2, &actual.2, 1e-2, &format!("ainv {msg}"));
        });
    }
}
