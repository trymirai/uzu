#![cfg(metal_backend)]

use proc_macros::uzu_test;

// Simdgroup is already covered on Metal by tree_gram_test.rs; only MXU is new here.
use super::tree_gram_test::{HEAD_K_DIM, get_output, make_inputs};
use crate::{
    backends::{
        common::Context,
        cpu::Cpu,
        metal::{Metal, MetalContext},
    },
    tests::assert::assert_eq_float,
};

#[uzu_test]
fn test_build_tree_gram_dispatch_paths() {
    let context = MetalContext::new().expect("Failed to create Context");
    if !context.supports_mxu() {
        return;
    }
    let scale = (HEAD_K_DIM as f32).sqrt().recip();

    for tree_size in [17, 64, 128] {
        let i = make_inputs(tree_size);
        let expected =
            get_output::<Cpu>(&i.q, &i.k, &i.trie, &i.prefix, &i.beta, &i.h0, &i.h0_idx, tree_size, scale, false);
        let actual =
            get_output::<Metal>(&i.q, &i.k, &i.trie, &i.prefix, &i.beta, &i.h0, &i.h0_idx, tree_size, scale, true);
        let msg = format!("backend {} path MXU tree_size {tree_size}", std::any::type_name::<Metal>());
        assert_eq_float::<f32>(&expected.0, &actual.0, 5e-3, &format!("a_packed {msg}"));
        assert_eq_float::<f32>(&expected.1, &actual.1, 5e-3, &format!("qkd {msg}"));
        assert_eq_float::<f32>(&expected.2, &actual.2, 1e-2, &format!("a_inv {msg}"));
        assert_eq_float::<f32>(&expected.3, &actual.3, 5e-3, &format!("kh0 {msg}"));
    }
}
