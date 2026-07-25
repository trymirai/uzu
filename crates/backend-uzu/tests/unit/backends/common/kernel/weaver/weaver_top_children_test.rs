use half::bf16;
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
    residual: &[bf16],
    candidate_logits: &[f32],
    ids: &[u32],
) -> (Vec<u32>, Vec<f32>) {
    let rows = residual.len() / CANDIDATES;
    assert_eq!(residual.len(), rows * CANDIDATES);
    assert_eq!(candidate_logits.len(), rows * CANDIDATES);
    assert_eq!(ids.len(), rows * CANDIDATES);
    let context = create_context::<B>();
    let residual = alloc_allocation_with_data::<B, bf16>(&context, residual);
    let candidate_logits = alloc_allocation_with_data::<B, f32>(&context, candidate_logits);
    let ids = alloc_allocation_with_data::<B, u32>(&context, ids);
    let mut output_token_ids = alloc_allocation::<B, u32>(&context, rows * CHILDREN);
    let mut output_model_logprobs = alloc_allocation::<B, f32>(&context, rows * CHILDREN);
    let kernel = <B::Kernels as Kernels>::WeaverTopChildrenKernel::new(&context).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &residual,
        &candidate_logits,
        &ids,
        &mut output_token_ids,
        &mut output_model_logprobs,
        rows as u32,
        CANDIDATES as u32,
        CHILDREN as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    (allocation_to_vec(&output_token_ids), allocation_to_vec(&output_model_logprobs))
}

#[uzu_test]
fn weaver_top_children_matches_cpu() {
    const ROWS: usize = 3;
    let residual = (0..ROWS * CANDIDATES)
        .map(|index| bf16::from_f32(((index as f32 * 0.017).cos() * 4.0).round() * 0.125))
        .collect::<Vec<_>>();
    let candidate_logits =
        (0..ROWS * CANDIDATES).map(|index| ((index as f32 * 0.011).sin() * 3.0).round() * 0.125).collect::<Vec<_>>();
    let ids = (0..ROWS)
        .flat_map(|row| (0..CANDIDATES).rev().map(move |index| 70_000 + (row * CANDIDATES + index) as u32))
        .collect::<Vec<_>>();

    for_each_non_cpu_backend!(|B| {
        let expected = top_children::<Cpu>(&residual, &candidate_logits, &ids);
        let actual = top_children::<B>(&residual, &candidate_logits, &ids);
        assert_eq!(actual.0, expected.0);
        for (actual, expected) in actual.1.iter().zip(expected.1) {
            assert!((actual - expected).abs() < 1e-5);
        }
    });
}
