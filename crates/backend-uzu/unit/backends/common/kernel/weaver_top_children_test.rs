use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    backends::{
        common::{Backend, Encoder, Kernels, kernel::WeaverTopChildrenKernel},
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, create_context},
};

const CANDIDATES: usize = 512;
const CHILDREN: usize = 8;

fn top_children<B: Backend>(
    residual: &[f32],
    scores: &[f32],
    ids: &[u64],
) -> (Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let residual = alloc_allocation_with_data::<B, f32>(&context, residual);
    let scores = alloc_allocation_with_data::<B, f32>(&context, scores);
    let ids = alloc_allocation_with_data::<B, u64>(&context, ids);
    let mut output_ids = alloc_allocation::<B, u32>(&context, CHILDREN);
    let mut output_logprobs = alloc_allocation::<B, f32>(&context, CHILDREN);
    let kernel = <B::Kernels as Kernels>::WeaverTopChildrenKernel::new(&context).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &residual,
        &scores,
        &ids,
        &mut output_ids,
        &mut output_logprobs,
        1,
        CANDIDATES as u32,
        CHILDREN as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    (allocation_to_vec(&output_ids), allocation_to_vec(&output_logprobs))
}

#[uzu_test]
fn weaver_top_children_matches_cpu() {
    let residual =
        (0..CANDIDATES).map(|index| ((index as f32 * 0.017).cos() * 4.0).round() * 0.125).collect::<Vec<_>>();
    let scores = (0..CANDIDATES).map(|index| ((index as f32 * 0.011).sin() * 3.0).round() * 0.125).collect::<Vec<_>>();
    let ids = (0..CANDIDATES).rev().map(|index| index as u64).collect::<Vec<_>>();

    for_each_non_cpu_backend!(|B| {
        let expected = top_children::<Cpu>(&residual, &scores, &ids);
        let actual = top_children::<B>(&residual, &scores, &ids);
        assert_eq!(actual.0, expected.0);
        for (actual, expected) in actual.1.iter().zip(expected.1) {
            assert!((actual - expected).abs() < 1e-5);
        }
    });
}
