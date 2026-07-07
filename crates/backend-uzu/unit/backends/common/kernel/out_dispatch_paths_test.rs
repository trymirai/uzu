#![cfg(metal_backend)]

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;

use super::out_test::{BUILD_TREE_OUT_PATHS, Shape, make_inputs, run_build_tree_out};
use crate::{
    array::ArrayElement,
    backends::{
        common::Context,
        cpu::Cpu,
        metal::{Metal, MetalContext},
    },
    tests::assert::assert_eq_float,
};

fn build_tree_out_mxu_paths(context: &MetalContext) -> impl Iterator<Item = (&'static str, bool, bool)> + '_ {
    BUILD_TREE_OUT_PATHS.iter().copied().filter(move |&(_, use_mxu, _)| use_mxu && context.supports_mxu())
}

fn check_shape<T: ArrayElement + Float + std::fmt::Display>(
    shape: Shape,
    eps: f32,
) {
    let inputs = make_inputs::<T>(shape);
    let context = MetalContext::new().expect("Failed to create Context");

    for use_h0 in [false, true] {
        let expected = run_build_tree_out::<Cpu, T>(shape, &inputs, use_h0, false, false);
        for (path, use_mxu, transposed_h0) in build_tree_out_mxu_paths(&context) {
            if transposed_h0 && !use_h0 {
                continue;
            }
            let actual = run_build_tree_out::<Metal, T>(shape, &inputs, use_h0, use_mxu, transposed_h0);
            let msg = format!(
                "backend {} path {path} use_h0 {use_h0} B{}_T{}_QK{}_HV{}_K{}_V{}",
                std::any::type_name::<Metal>(),
                shape.batch_size,
                shape.tree_size,
                shape.qk_heads,
                shape.value_heads,
                shape.head_k_dim,
                shape.head_v_dim
            );
            assert_eq_float::<T>(&expected, &actual, eps, &msg);
        }
    }
}

#[uzu_test]
fn test_build_tree_out_dispatch_paths() {
    let small = Shape {
        batch_size: 2,
        tree_size: 17,
        qk_heads: 2,
        value_heads: 6,
        head_k_dim: 32,
        head_v_dim: 32,
    };
    let gdn = Shape {
        batch_size: 1,
        tree_size: 49,
        qk_heads: 16,
        value_heads: 48,
        head_k_dim: 128,
        head_v_dim: 128,
    };
    for shape in [small, gdn] {
        check_shape::<bf16>(shape, 0.08);
        check_shape::<f32>(shape, 5e-3);
    }
}
