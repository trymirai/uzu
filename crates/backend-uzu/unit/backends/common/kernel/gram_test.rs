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

const BATCH_SIZE: usize = 2;
const K_HEADS: usize = 2;
const VALUE_HEADS: usize = 6;
const HEAD_K_DIM: usize = 128;

fn get_output<B: Backend>(
    q: &[f32],
    k: &[f32],
    trie: &[u32],
    prefix: &[f32],
    beta: &[f32],
    tree_size: usize,
    scale: f32,
    use_mxu: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::BuildTreeGramKernel::new(&context, DataType::F32, use_mxu)
        .expect("Failed to create BuildTreeGramKernel");

    let output_len = BATCH_SIZE * VALUE_HEADS * tree_size * tree_size;
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
        BATCH_SIZE as u32,
        tree_size as u32,
        K_HEADS as u32,
        VALUE_HEADS as u32,
        HEAD_K_DIM as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec(&a_mat), allocation_to_vec(&qkd), allocation_to_vec(&ainv))
}

fn build_trie(tree_size: usize) -> Vec<u32> {
    let last = tree_size as u32 - 1;
    let mut trie = Vec::with_capacity(BATCH_SIZE * tree_size * 3);
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
    let scale = (HEAD_K_DIM as f32).sqrt().recip();

    for tree_size in [17, 64, 128] {
        let q_len = BATCH_SIZE * tree_size * K_HEADS * HEAD_K_DIM;
        let kv_len = BATCH_SIZE * tree_size * VALUE_HEADS;
        let trie = build_trie(tree_size);
        let q = (0..q_len).map(|i| ((i as f32 * 0.017).sin() * 0.2) + 0.01).collect::<Vec<_>>();
        let k = (0..q_len).map(|i| ((i as f32 * 0.019).cos() * 0.18) - 0.02).collect::<Vec<_>>();
        let prefix = (0..kv_len)
            .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % VALUE_HEADS) as f32) * 0.003)
            .collect::<Vec<_>>();
        let beta = (0..kv_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect::<Vec<_>>();

        let expected = get_output::<Cpu>(&q, &k, &trie, &prefix, &beta, tree_size, scale, false);

        for_each_non_cpu_backend!(|B| {
            for use_mxu in [false, true] {
                let actual = get_output::<B>(&q, &k, &trie, &prefix, &beta, tree_size, scale, use_mxu);
                let path = if use_mxu {
                    "mxu"
                } else {
                    "simdgroup"
                };
                let msg = format!("backend {} {path} tree_size {tree_size}", std::any::type_name::<B>());
                assert_eq_float::<f32>(&expected.0, &actual.0, 5e-3, &format!("a_mat {msg}"));
                assert_eq_float::<f32>(&expected.1, &actual.1, 5e-3, &format!("qkd {msg}"));

                for (expected_matrix, actual_matrix) in
                    expected.2.chunks(tree_size * tree_size).zip(actual.2.chunks(tree_size * tree_size))
                {
                    for block_start in (0..tree_size).step_by(16) {
                        let block_end = (block_start + 16).min(tree_size);
                        for i in block_start..block_end {
                            for j in block_start..block_end {
                                let idx = i * tree_size + j;
                                let expected = expected_matrix[idx];
                                let actual = actual_matrix[idx];
                                assert!(
                                    (expected - actual).abs() < 1e-2,
                                    "ainv {msg}. Mismatch at matrix index {idx}: expected {expected}, got {actual}"
                                );
                            }
                        }
                    }
                }
            }
        });
    }
}
