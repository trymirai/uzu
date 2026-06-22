use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildPrefixBetaKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{assert::assert_eq_float, helpers::allocation_to_vec},
};

fn get_output<B: Backend>(
    trie: &[u32],
    a_transposed: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    batch_size: u32,
    tree_size: u32,
    value_heads: u32,
) -> (Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::BuildPrefixBetaKernel::new(&context, DataType::F32)
        .expect("Failed to create BuildPrefixBetaKernel");

    let output_len = a_transposed.len();
    let trie = context.create_array_from(&[trie.len()], trie);
    let a_transposed = context.create_array_from(&[a_transposed.len()], a_transposed);
    let b = context.create_array_from(&[b.len()], b);
    let a_log = context.create_array_from(&[a_log.len()], a_log);
    let dt_bias = context.create_array_from(&[dt_bias.len()], dt_bias);
    let mut prefix = context.create_array_uninitialized(&[output_len], DataType::F32).into_allocation();
    let mut beta = context.create_array_uninitialized(&[output_len], DataType::F32).into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        trie.allocation(),
        a_transposed.allocation(),
        b.allocation(),
        a_log.allocation(),
        dt_bias.allocation(),
        &mut prefix,
        &mut beta,
        batch_size,
        tree_size,
        value_heads,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec(&prefix), allocation_to_vec(&beta))
}

// chain (batch 0) + broom (batch 1); the broom's varied [start, end] exercises
// trie_end, which the chain (end always T-1) can't.
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
fn test_build_prefix_beta_matches_cpu() {
    let batch_size = 2;
    let value_heads = 5;

    for tree_size in [5, 33, 129] {
        let len = batch_size * tree_size * value_heads;
        let trie = build_trie(tree_size);
        let a_transposed = (0..len).map(|i| (i as f32 * 0.13).sin() - 0.4).collect::<Vec<_>>();
        let b = (0..len).map(|i| (i as f32 * 0.11).cos() * 0.8).collect::<Vec<_>>();
        let a_log = (0..value_heads).map(|i| -0.2 + i as f32 * 0.03).collect::<Vec<_>>();
        let dt_bias = (0..value_heads).map(|i| 0.1 - i as f32 * 0.02).collect::<Vec<_>>();
        let expected = get_output::<Cpu>(
            &trie,
            &a_transposed,
            &b,
            &a_log,
            &dt_bias,
            batch_size as u32,
            tree_size as u32,
            value_heads as u32,
        );

        for_each_non_cpu_backend!(|B| {
            let output = get_output::<B>(
                &trie,
                &a_transposed,
                &b,
                &a_log,
                &dt_bias,
                batch_size as u32,
                tree_size as u32,
                value_heads as u32,
            );
            let msg = format!("backend {} tree_size {tree_size}", std::any::type_name::<B>());
            assert_eq_float::<f32>(&expected.0, &output.0, 1e-4, &format!("prefix {msg}"));
            assert_eq_float::<f32>(&expected.1, &output.1, 1e-5, &format!("beta {msg}"));
        });
    }
}
