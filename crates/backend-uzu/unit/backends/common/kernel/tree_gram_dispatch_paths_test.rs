#![cfg(metal_backend)]

use proc_macros::uzu_test;

use super::tree_gram_test::{BATCH_SIZE, HEAD_K_DIM, HEAD_V_DIM, K_HEADS, VALUE_HEADS, build_trie, get_output};
use crate::{
    backends::{
        common::Context,
        cpu::Cpu,
        metal::{Metal, MetalContext},
    },
    tests::assert::assert_eq_float,
};

fn build_tree_gram_paths(context: &MetalContext) -> Vec<(&'static str, bool)> {
    let mut paths = vec![("Simdgroup", false)];
    if context.supports_mxu() {
        paths.push(("MXU", true));
    }
    paths
}

#[uzu_test]
fn test_build_tree_gram_dispatch_paths() {
    let scale = (HEAD_K_DIM as f32).sqrt().recip();
    let context = MetalContext::new().expect("Failed to create Context");

    for tree_size in [17, 64, 128] {
        let q_len = BATCH_SIZE * tree_size * K_HEADS * HEAD_K_DIM;
        let scalar_len = BATCH_SIZE * tree_size * VALUE_HEADS;
        let trie = build_trie(tree_size);
        let q = (0..q_len).map(|i| ((i as f32 * 0.017).sin() * 0.2) + 0.01).collect::<Vec<_>>();
        let k = (0..q_len).map(|i| ((i as f32 * 0.019).cos() * 0.18) - 0.02).collect::<Vec<_>>();
        let prefix = (0..scalar_len)
            .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % VALUE_HEADS) as f32) * 0.003)
            .collect::<Vec<_>>();
        let beta = (0..scalar_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect::<Vec<_>>();
        let h0_len = BATCH_SIZE * VALUE_HEADS * HEAD_V_DIM * HEAD_K_DIM;
        let h0 = (0..h0_len).map(|i| ((i as f32 * 0.007).sin() * 0.05) - 0.01).collect::<Vec<_>>();
        let h0_idx: Vec<i32> = vec![0, -1];
        let expected = get_output::<Cpu>(&q, &k, &trie, &prefix, &beta, &h0, &h0_idx, tree_size, scale, false);

        for (path, use_mxu) in build_tree_gram_paths(&context) {
            let actual = get_output::<Metal>(&q, &k, &trie, &prefix, &beta, &h0, &h0_idx, tree_size, scale, use_mxu);
            let msg = format!("backend {} path {path} tree_size {tree_size}", std::any::type_name::<Metal>());
            assert_eq_float::<f32>(&expected.0, &actual.0, 5e-3, &format!("a_packed {msg}"));
            assert_eq_float::<f32>(&expected.1, &actual.1, 5e-3, &format!("qkd {msg}"));
            assert_eq_float::<f32>(&expected.2, &actual.2, 1e-2, &format!("a_inv {msg}"));
            assert_eq_float::<f32>(&expected.3, &actual.3, 5e-3, &format!("kh0 {msg}"));
        }
    }
}
