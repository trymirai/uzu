use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::DeltaNetConvTreeScanKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

const CONV_DIM: usize = 37;
const TOTAL_PROJ_DIM: usize = 48;
const KERNEL_SIZE: usize = 4;
const STATE_STRIDE: usize = KERNEL_SIZE - 1;

fn parents(
    tree_size: usize,
    shape: &str,
) -> Vec<i32> {
    (0..tree_size)
        .map(|node| {
            if node == 0 {
                -1
            } else {
                match shape {
                    "chain" => node as i32 - 1,
                    "star" => 0,
                    "binary" => ((node - 1) / 2) as i32,
                    "random" => (((node ^ (node >> 1)) * 17 + 3) % node) as i32,
                    _ => unreachable!(),
                }
            }
        })
        .collect()
}

fn run<B: Backend>(
    tree_size: usize,
    shape: &str,
    perturb: bool,
) -> (Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("context");
    let kernel = <<B as Backend>::Kernels as Kernels>::DeltaNetConvTreeScanKernel::new(
        &context,
        DataType::F32,
        KERNEL_SIZE as u32,
        true,
    )
    .expect("kernel");
    let mut input = (0..tree_size * TOTAL_PROJ_DIM).map(|i| (i % 37) as f32 * 0.01 - 0.2).collect::<Vec<_>>();
    if perturb {
        input[TOTAL_PROJ_DIM..TOTAL_PROJ_DIM + CONV_DIM].iter_mut().for_each(|value| *value += 0.5);
    }
    let weights = (0..CONV_DIM * KERNEL_SIZE).map(|i| (i % 11) as f32 * 0.02 - 0.1).collect::<Vec<_>>();
    let bias = (0..CONV_DIM).map(|i| i as f32 * 0.003 - 0.04).collect::<Vec<_>>();
    let base_state = (0..CONV_DIM * STATE_STRIDE).map(|i| (i % 13) as f32 * 0.01 - 0.05).collect::<Vec<_>>();

    let input = alloc_allocation_with_data::<B, f32>(&context, &input);
    let weights = alloc_allocation_with_data::<B, f32>(&context, &weights);
    let bias = alloc_allocation_with_data::<B, f32>(&context, &bias);
    let base_state_buffer = alloc_allocation_with_data::<B, f32>(&context, &base_state);
    let parents = alloc_allocation_with_data::<B, i32>(&context, &parents(tree_size, shape));
    let mut output = alloc_allocation::<B, f32>(&context, tree_size * TOTAL_PROJ_DIM);
    let mut suffix_state = alloc_allocation::<B, f32>(&context, tree_size * CONV_DIM * STATE_STRIDE);

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    kernel.encode(
        &input,
        &weights,
        Some(&bias),
        &base_state_buffer,
        &parents,
        &mut output,
        &mut suffix_state,
        tree_size as u32,
        TOTAL_PROJ_DIM as u32,
        CONV_DIM as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    assert_eq_float(
        &base_state,
        &allocation_to_vec(&base_state_buffer),
        0.0,
        &format!("base state {shape} T={tree_size}"),
    );
    (allocation_to_vec(&output), allocation_to_vec(&suffix_state))
}

#[uzu_test]
fn test_delta_net_conv_tree_scan() {
    for tree_size in [49, 64, 128] {
        for shape in ["chain", "star", "binary", "random"] {
            let expected = run::<Cpu>(tree_size, shape, false);
            let label = format!("{shape} T={tree_size}");
            for_each_non_cpu_backend!(|B| {
                let actual = run::<B>(tree_size, shape, false);
                assert_eq_float(&expected.0, &actual.0, 1e-5, &format!("output {label}"));
                assert_eq_float(&expected.1, &actual.1, 1e-6, &format!("suffix state {label}"));
            });
        }
    }

    let baseline = run::<Cpu>(49, "binary", false);
    let perturbed = run::<Cpu>(49, "binary", true);
    let output = 2 * TOTAL_PROJ_DIM..3 * TOTAL_PROJ_DIM;
    let state = 2 * CONV_DIM * STATE_STRIDE..3 * CONV_DIM * STATE_STRIDE;
    assert_eq_float(&baseline.0[output.clone()], &perturbed.0[output], 0.0, "sibling output");
    assert_eq_float(&baseline.1[state.clone()], &perturbed.1[state], 0.0, "sibling state");
}
