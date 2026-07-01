use half::bf16;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildTreeOutKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{assert::assert_eq_float, helpers::allocation_to_vec},
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

struct Inputs {
    q: Vec<bf16>,
    prefix: Vec<f32>,
    qkd: Vec<f32>,
    u: Vec<bf16>,
    h0: Vec<bf16>,
    h0_indices: Vec<i32>,
}

fn make_inputs(shape: Shape) -> Inputs {
    let q_len = shape.batch_size * shape.tree_size * shape.qk_heads * shape.head_k_dim;
    let prefix_len = shape.batch_size * shape.tree_size * shape.value_heads;
    let qkd_len = shape.batch_size * shape.value_heads * shape.tree_size * shape.tree_size;
    let u_len = shape.batch_size * shape.value_heads * shape.tree_size * shape.head_v_dim;
    let h0_pool = shape.batch_size + 1;
    let h0_len = h0_pool * shape.value_heads * shape.head_v_dim * shape.head_k_dim;

    Inputs {
        q: (0..q_len).map(|i| bf16::from_f32(((i as f32 * 0.017).sin() * 0.2) + 0.01)).collect(),
        prefix: (0..prefix_len)
            .map(|i| -((i % shape.tree_size) as f32) * 0.01 - ((i % shape.value_heads) as f32) * 0.003)
            .collect(),
        qkd: (0..qkd_len).map(|i| ((i as f32 * 0.013).cos() * 0.1) - 0.02).collect(),
        u: (0..u_len).map(|i| bf16::from_f32(((i as f32 * 0.011).sin() * 0.3) + 0.04)).collect(),
        h0: (0..h0_len).map(|i| bf16::from_f32(((i as f32 * 0.019).cos() * 0.2) - 0.01)).collect(),
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

fn run_build_tree_out<B: Backend>(
    shape: Shape,
    inputs: &Inputs,
    use_h0: bool,
    use_mxu: bool,
) -> Vec<bf16> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel =
        <<B as Backend>::Kernels as Kernels>::BuildTreeOutKernel::new(&context, DataType::BF16, use_mxu, use_h0)
            .expect("BuildTreeOutKernel");
    let q = context.as_ref().create_array_from(&[inputs.q.len()], &inputs.q).into_allocation();
    let prefix = context.as_ref().create_array_from(&[inputs.prefix.len()], &inputs.prefix).into_allocation();
    let qkd = context.as_ref().create_array_from(&[inputs.qkd.len()], &inputs.qkd).into_allocation();
    let u = context.as_ref().create_array_from(&[inputs.u.len()], &inputs.u).into_allocation();
    let h0 = use_h0.then(|| context.as_ref().create_array_from(&[inputs.h0.len()], &inputs.h0).into_allocation());
    let h0_indices = use_h0
        .then(|| context.as_ref().create_array_from(&[inputs.h0_indices.len()], &inputs.h0_indices).into_allocation());
    let mut o = context
        .as_ref()
        .create_array_uninitialized(
            &[shape.batch_size * shape.tree_size * shape.value_heads * shape.head_v_dim],
            DataType::BF16,
        )
        .into_allocation();

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

fn check_shape(shape: Shape) {
    let inputs = make_inputs(shape);
    let kernel_paths = [("Simdgroup", false), ("MXU", true)];

    for use_h0 in [false, true] {
        let expected = run_build_tree_out::<Cpu>(shape, &inputs, use_h0, false);
        for_each_non_cpu_backend!(|B| {
            let context = <B as Backend>::Context::new().expect("Failed to create Context");
            for &(kernel_path, use_mxu) in &kernel_paths {
                if use_mxu && !context.supports_mxu() {
                    continue;
                }
                let actual = run_build_tree_out::<B>(shape, &inputs, use_h0, use_mxu);
                let msg = format!(
                    "backend {} kernel_path {kernel_path} use_h0 {use_h0} B{}_T{}_QK{}_HV{}_K{}_V{}",
                    std::any::type_name::<B>(),
                    shape.batch_size,
                    shape.tree_size,
                    shape.qk_heads,
                    shape.value_heads,
                    shape.head_k_dim,
                    shape.head_v_dim
                );
                assert_eq_float::<bf16>(&expected, &actual, 0.08, &msg);
            }
        });
    }
}

#[uzu_test]
fn test_build_tree_out_small() {
    check_shape(Shape {
        batch_size: 2,
        tree_size: 17,
        qk_heads: 2,
        value_heads: 6,
        head_k_dim: 32,
        head_v_dim: 32,
    });
}

#[uzu_test]
fn test_build_tree_out_gdn_shape() {
    check_shape(Shape {
        batch_size: 1,
        tree_size: 49,
        qk_heads: 16,
        value_heads: 48,
        head_k_dim: 128,
        head_v_dim: 128,
    });
}
