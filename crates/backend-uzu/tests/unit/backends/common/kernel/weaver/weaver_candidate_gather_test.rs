use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    backends::{
        common::{
            Backend, Encoder, Kernels,
            kernel::{WeaverCandidateGatherKernel, weaver::METADATA_LANE_DEPTH},
        },
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec, create_context},
};

const ROWS: usize = 8;
const SIZE: usize = 512;

fn gather<B: Backend>() -> (Vec<u32>, Vec<u32>) {
    let context = create_context::<B>();
    let ids = (0..16 * SIZE).map(|i| i as u32).collect::<Vec<_>>();
    let scores = (0..16 * SIZE).map(|i| -(i as f32) * 0.031_25).collect::<Vec<_>>();
    let depths = [0, 1, 15, 7, 0, 3, 16, 20];
    let mut metadata = vec![0; 3 * ROWS];
    for row in 0..ROWS {
        metadata[METADATA_LANE_DEPTH * ROWS + row] = depths[row];
    }
    let ids = alloc_allocation_with_data::<B, u32>(&context, &ids);
    let scores = alloc_allocation_with_data::<B, f32>(&context, &scores);
    let metadata = alloc_allocation_with_data::<B, u32>(&context, &metadata);
    let mut output_ids = alloc_allocation_with_data::<B, u32>(&context, &[0; ROWS * SIZE]);
    let mut output_scores = alloc_allocation_with_data::<B, f32>(&context, &[0.0; ROWS * SIZE]);
    let kernel = <B::Kernels as Kernels>::WeaverCandidateGatherKernel::new(&context).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &ids,
        &scores,
        &metadata,
        &mut output_ids,
        &mut output_scores,
        ROWS as u32,
        16,
        SIZE as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    (
        allocation_to_vec(&output_ids),
        allocation_to_vec::<B, f32>(&output_scores).into_iter().map(f32::to_bits).collect(),
    )
}

#[uzu_test]
fn weaver_candidate_gather_matches_cpu() {
    for_each_non_cpu_backend!(|B| assert_eq!(gather::<B>(), gather::<Cpu>()));
}
