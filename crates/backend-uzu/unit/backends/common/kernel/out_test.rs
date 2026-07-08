#![cfg(metal_backend)]

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildTreeOutKernel},
        cpu::Cpu,
        metal::Metal,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

#[derive(Clone, Copy)]
struct Shape {
    batch_size: usize,
    tree_size: usize,
    qk_heads: usize,
    value_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
}

struct Inputs<T> {
    q: Vec<T>,
    prefix: Vec<f32>,
    qkd: Vec<f32>,
    u: Vec<T>,
    h0: Vec<f32>,
    h0_indices: Vec<i32>,
}

const BUILD_TREE_OUT_PATHS: &[(&str, bool, bool)] = &[
    ("Simdgroup/H0Direct/Rows8_VCols32_SG4", false, false),
    ("Simdgroup/H0Transposed/Rows8_VCols32_SG4", false, true),
    ("MXU/H0Direct/Rows16_VCols32_SG4", true, false),
];

fn make_inputs<T: ArrayElement + Float>(shape: Shape) -> Inputs<T> {
    let q_len = shape.batch_size * shape.tree_size * shape.qk_heads * shape.head_k_dim;
    let prefix_len = shape.batch_size * shape.tree_size * shape.value_heads;
    let qkd_len = shape.batch_size * shape.value_heads * shape.tree_size * shape.tree_size;
    let u_len = shape.batch_size * shape.value_heads * shape.tree_size * shape.head_v_dim;
    let h0_pool = shape.batch_size + 1;
    let h0_len = h0_pool * shape.value_heads * shape.head_v_dim * shape.head_k_dim;

    Inputs {
        q: (0..q_len).map(|i| T::from(((i as f32 * 0.017).sin() * 0.2) + 0.01).unwrap()).collect(),
        prefix: (0..prefix_len)
            .map(|i| -((i % shape.tree_size) as f32) * 0.01 - ((i % shape.value_heads) as f32) * 0.003)
            .collect(),
        qkd: (0..qkd_len).map(|i| ((i as f32 * 0.013).cos() * 0.1) - 0.02).collect(),
        u: (0..u_len).map(|i| T::from(((i as f32 * 0.011).sin() * 0.3) + 0.04).unwrap()).collect(),
        h0: (0..h0_len).map(|i| ((i as f32 * 0.019).cos() * 0.2) - 0.01).collect(),
        h0_indices: (0..shape.batch_size)
            .map(|i| {
                if i + 1 == shape.batch_size {
                    -1
                } else {
                    i as i32
                }
            })
            .collect(),
    }
}

fn run_build_tree_out<B: Backend, T: ArrayElement + Float>(
    shape: Shape,
    inputs: &Inputs<T>,
    use_h0: bool,
    use_mxu: bool,
    transposed_h0: bool,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::BuildTreeOutKernel::new(
        &context,
        T::data_type(),
        use_mxu,
        transposed_h0,
        use_h0,
    )
    .expect("BuildTreeOutKernel");
    let q = alloc_allocation_with_data::<B, T>(&context, &inputs.q);
    let prefix = alloc_allocation_with_data::<B, f32>(&context, &inputs.prefix);
    let qkd = alloc_allocation_with_data::<B, f32>(&context, &inputs.qkd);
    let u = alloc_allocation_with_data::<B, T>(&context, &inputs.u);
    let h0 = use_h0.then(|| alloc_allocation_with_data::<B, f32>(&context, &inputs.h0));
    let h0_indices = use_h0.then(|| alloc_allocation_with_data::<B, i32>(&context, &inputs.h0_indices));
    let mut o =
        alloc_allocation::<B, T>(&context, shape.batch_size * shape.tree_size * shape.value_heads * shape.head_v_dim);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &q,
        &prefix,
        &qkd,
        &u,
        h0.as_ref(),
        h0_indices.as_ref(),
        &mut o,
        (shape.head_k_dim as f32).sqrt().recip(),
        shape.batch_size as u32,
        shape.tree_size as u32,
        shape.qk_heads as u32,
        shape.value_heads as u32,
        shape.head_k_dim as u32,
        shape.head_v_dim as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&o)
}

fn check_shape<T: ArrayElement + Float + std::fmt::Display>(
    shape: Shape,
    eps: f32,
) {
    let inputs = make_inputs::<T>(shape);

    for use_h0 in [false, true] {
        let expected = run_build_tree_out::<Cpu, T>(shape, &inputs, use_h0, false, false);
        for_each_non_cpu_backend!(|B| {
            for &(path, use_mxu, transposed_h0) in BUILD_TREE_OUT_PATHS {
                if use_mxu || transposed_h0 && !use_h0 {
                    continue;
                }
                let actual = run_build_tree_out::<B, T>(shape, &inputs, use_h0, use_mxu, transposed_h0);
                let msg = format!(
                    "backend {} path {path} use_h0 {use_h0} B{}_T{}_QK{}_HV{}_K{}_V{}",
                    std::any::type_name::<B>(),
                    shape.batch_size,
                    shape.tree_size,
                    shape.qk_heads,
                    shape.value_heads,
                    shape.head_k_dim,
                    shape.head_v_dim
                );
                assert_eq_float::<T>(&expected, &actual, eps, &msg);
            }
        });

        if <Metal as Backend>::Context::new().expect("Failed to create Context").supports_mxu() {
            for &(path, use_mxu, transposed_h0) in BUILD_TREE_OUT_PATHS {
                if !use_mxu || transposed_h0 && !use_h0 {
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
}

#[uzu_test]
fn test_build_tree_out_paths() {
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
