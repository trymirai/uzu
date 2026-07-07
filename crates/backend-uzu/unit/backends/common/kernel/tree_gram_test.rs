use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

#[cfg(metal_backend)]
use crate::backends::metal::Metal;
use crate::{
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildTreeGramKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

const BATCH_SIZE: usize = 2;
const K_HEADS: usize = 2;
const VALUE_HEADS: usize = 6;
const HEAD_K_DIM: usize = 128;
// 80 = two full 32-wide kh0 dv chunks plus a ragged 16-wide one.
const HEAD_V_DIM: usize = 80;

struct Inputs {
    trie: Vec<u32>,
    q: Vec<f32>,
    k: Vec<f32>,
    prefix: Vec<f32>,
    beta: Vec<f32>,
    h0: Vec<f32>,
    h0_idx: Vec<i32>,
}

fn make_inputs(tree_size: usize) -> Inputs {
    let q_len = BATCH_SIZE * tree_size * K_HEADS * HEAD_K_DIM;
    let scalar_len = BATCH_SIZE * tree_size * VALUE_HEADS;
    let h0_len = BATCH_SIZE * VALUE_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    Inputs {
        trie: build_trie(tree_size),
        q: (0..q_len).map(|i| ((i as f32 * 0.017).sin() * 0.2) + 0.01).collect(),
        k: (0..q_len).map(|i| ((i as f32 * 0.019).cos() * 0.18) - 0.02).collect(),
        prefix: (0..scalar_len)
            .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % VALUE_HEADS) as f32) * 0.003)
            .collect(),
        beta: (0..scalar_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect(),
        h0: (0..h0_len).map(|i| ((i as f32 * 0.007).sin() * 0.05) - 0.01).collect(),
        // Batch 1 has no initial state: covers the kh0-skip path.
        h0_idx: vec![0, -1],
    }
}

fn get_output<B: Backend>(
    q: &[f32],
    k: &[f32],
    trie: &[u32],
    prefix: &[f32],
    beta: &[f32],
    h0: &[f32],
    h0_idx: &[i32],
    tree_size: usize,
    scale: f32,
    use_mxu: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::BuildTreeGramKernel::new(&context, DataType::F32, use_mxu, true)
        .expect("Failed to create BuildTreeGramKernel");

    let output_len = BATCH_SIZE * VALUE_HEADS * tree_size * tree_size;
    let num_blocks = tree_size.div_ceil(16);
    let a_len = BATCH_SIZE * VALUE_HEADS * num_blocks * num_blocks.div_ceil(2) * 16 * 32;
    let a_inv_len = BATCH_SIZE * VALUE_HEADS * num_blocks * 16 * 16;
    let kh0_len = BATCH_SIZE * tree_size * VALUE_HEADS * HEAD_V_DIM;
    let q = alloc_allocation_with_data::<B, f32>(&context, q);
    let k = alloc_allocation_with_data::<B, f32>(&context, k);
    let trie = alloc_allocation_with_data::<B, u32>(&context, trie);
    let prefix = alloc_allocation_with_data::<B, f32>(&context, prefix);
    let beta = alloc_allocation_with_data::<B, f32>(&context, beta);
    let h0 = alloc_allocation_with_data::<B, f32>(&context, h0);
    let h0_idx = alloc_allocation_with_data::<B, i32>(&context, h0_idx);
    // The GPU only writes packed-A tiles touching the block lower triangle;
    // zero-init so CPU and GPU buffers stay comparable elsewhere.
    let mut a_packed = alloc_allocation_with_data::<B, f32>(&context, &vec![0.0f32; a_len]);
    let mut qkd = alloc_allocation::<B, f32>(&context, output_len);
    let mut a_inv = alloc_allocation::<B, f32>(&context, a_inv_len);
    // kh0 is only written for batches with h0_idx >= 0; zero-init so CPU and GPU
    // outputs stay comparable on the skipped batch.
    let mut kh0 = alloc_allocation_with_data::<B, f32>(&context, &vec![0.0f32; kh0_len]);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &q,
        &k,
        &trie,
        &prefix,
        &beta,
        Some(&h0),
        Some(&h0_idx),
        &mut a_packed,
        &mut qkd,
        &mut a_inv,
        Some(&mut kh0),
        scale,
        BATCH_SIZE as u32,
        tree_size as u32,
        K_HEADS as u32,
        VALUE_HEADS as u32,
        HEAD_K_DIM as u32,
        HEAD_V_DIM as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec(&a_packed), allocation_to_vec(&qkd), allocation_to_vec(&a_inv), allocation_to_vec(&kh0))
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
        let i = make_inputs(tree_size);
        let expected =
            get_output::<Cpu>(&i.q, &i.k, &i.trie, &i.prefix, &i.beta, &i.h0, &i.h0_idx, tree_size, scale, false);

        for_each_non_cpu_backend!(|B| {
            let actual =
                get_output::<B>(&i.q, &i.k, &i.trie, &i.prefix, &i.beta, &i.h0, &i.h0_idx, tree_size, scale, false);
            let msg = format!("backend {} simdgroup tree_size {tree_size}", std::any::type_name::<B>());
            assert_eq_float::<f32>(&expected.0, &actual.0, 5e-3, &format!("a_packed {msg}"));
            assert_eq_float::<f32>(&expected.1, &actual.1, 5e-3, &format!("qkd {msg}"));
            assert_eq_float::<f32>(&expected.2, &actual.2, 1e-2, &format!("a_inv {msg}"));
            assert_eq_float::<f32>(&expected.3, &actual.3, 5e-3, &format!("kh0 {msg}"));
        });

        #[cfg(metal_backend)]
        if <Metal as Backend>::Context::new().expect("Failed to create Context").supports_mxu() {
            let actual =
                get_output::<Metal>(&i.q, &i.k, &i.trie, &i.prefix, &i.beta, &i.h0, &i.h0_idx, tree_size, scale, true);
            let msg = format!("backend {} path MXU tree_size {tree_size}", std::any::type_name::<Metal>());
            assert_eq_float::<f32>(&expected.0, &actual.0, 5e-3, &format!("a_packed {msg}"));
            assert_eq_float::<f32>(&expected.1, &actual.1, 5e-3, &format!("qkd {msg}"));
            assert_eq_float::<f32>(&expected.2, &actual.2, 1e-2, &format!("a_inv {msg}"));
            assert_eq_float::<f32>(&expected.3, &actual.3, 5e-3, &format!("kh0 {msg}"));
        }
    }
}
