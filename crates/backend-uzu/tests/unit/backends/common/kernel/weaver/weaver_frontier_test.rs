use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    backends::{
        common::{
            Backend, Encoder, Kernels,
            gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx},
            kernel::{WeaverFrontierScatterKernel, WeaverFrontierSelectKernel},
        },
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec, create_context},
};

fn select<B: Backend>() -> Vec<u32> {
    let context = create_context::<B>();
    let mut frontier = vec![0; FrontierIdx::COUNT * 8];
    for (slot, (token, parent, depth, cum, key, active)) in [
        (9, 1, 1, 0x3f00_0000, 100, 1),
        (8, 0, 2, 0x3f00_0001, 100, 1),
        (7, 0, 2, 0x3f00_0002, 100, 1),
        (7, 0, 2, 0x3f00_0003, 100, 1),
        (2, 1, 3, 0x3f00_0004, 90, 1),
        (0, 0, 0, 0x3f00_0005, 200, 0),
        (4, 1, 3, 0x3f00_0006, 80, 1),
        (5, 1, 1, 0x3f00_0007, 70, 1),
    ]
    .into_iter()
    .enumerate()
    {
        for (lane, value) in [token, parent, depth, cum, 0xbf80_0000, key, active].into_iter().enumerate() {
            frontier[lane * 8 + slot] = value;
        }
    }
    let mut frontier = alloc_allocation_with_data::<B, u32>(&context, &frontier);
    let mut tree = alloc_allocation_with_data::<B, u32>(&context, &[55; TreeIdx::COUNT * 7]);
    let mut slot_ancestors = alloc_allocation_with_data::<B, u32>(&context, &(0u32..7 * 3).collect::<Vec<_>>());
    let mut token = alloc_allocation_with_data::<B, u32>(&context, &[66; 4]);
    let mut metadata = alloc_allocation_with_data::<B, u32>(&context, &[77; 3 * 4]);
    let mut ancestors = alloc_allocation_with_data::<B, u32>(&context, &[88; 4 * 3]);
    let mut valid = alloc_allocation_with_data::<B, u32>(&context, &[99; 4]);
    let candidate_pool_ids = alloc_allocation_with_data::<B, u32>(&context, &(0..12).collect::<Vec<_>>());
    let candidate_pool_scores =
        alloc_allocation_with_data::<B, f32>(&context, &(0..12).map(|value| value as f32).collect::<Vec<_>>());
    let mut candidate_ids = alloc_allocation_with_data::<B, u32>(&context, &[0; 4 * 3]);
    let mut candidate_scores = alloc_allocation_with_data::<B, f32>(&context, &[0.0; 4 * 3]);
    let kernel = <B::Kernels as Kernels>::WeaverFrontierSelectKernel::new(&context).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &mut frontier,
        &mut tree,
        &mut slot_ancestors,
        &mut token,
        &mut metadata,
        &mut ancestors,
        &mut valid,
        &candidate_pool_ids,
        &candidate_pool_scores,
        &mut candidate_ids,
        &mut candidate_scores,
        8,
        7,
        4,
        2,
        3,
        4,
        3,
        4,
        3,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    [frontier, tree, slot_ancestors, token, metadata, ancestors, valid, candidate_ids]
        .iter()
        .flat_map(allocation_to_vec)
        .chain(allocation_to_vec::<B, f32>(&candidate_scores).into_iter().map(f32::to_bits))
        .collect()
}

fn scatter<B: Backend>() -> Vec<u32> {
    let context = create_context::<B>();
    let mut tree = vec![0; TreeIdx::COUNT * 4];
    tree[TreeIdx::PathLogprobBits as usize * 4..(TreeIdx::PathLogprobBits as usize + 1) * 4]
        .copy_from_slice(&[0.5, -1.0, 2.0, 4.0].map(f32::to_bits));
    tree[TreeIdx::Depth as usize * 4..(TreeIdx::Depth as usize + 1) * 4].copy_from_slice(&[0, 2, 4, 6]);
    let tree = alloc_allocation_with_data::<B, u32>(&context, &tree);
    let mut metadata = vec![0; MetadataIdx::COUNT * 3];
    metadata[MetadataIdx::TreeSlot as usize * 3..(MetadataIdx::TreeSlot as usize + 1) * 3].copy_from_slice(&[1, 3, 0]);
    let metadata = alloc_allocation_with_data::<B, u32>(&context, &metadata);
    let valid = alloc_allocation_with_data::<B, u32>(&context, &[1, 0, 1]);
    let ids = alloc_allocation_with_data::<B, u32>(&context, &(10..19).collect::<Vec<_>>());
    let scores = alloc_allocation_with_data::<B, f32>(&context, &[-0.1, -0.2, -0.3, 8.0, 8.0, 8.0, 0.1, 0.2, 0.3]);
    let mut frontier = alloc_allocation_with_data::<B, u32>(&context, &[42; FrontierIdx::COUNT * 16]);
    let kernel = <B::Kernels as Kernels>::WeaverFrontierScatterKernel::new(&context).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(&tree, &metadata, &valid, &ids, &scores, &mut frontier, 16, 4, 3, 3, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&frontier)
}

#[uzu_test]
fn weaver_frontier_kernels_match_cpu() {
    for_each_non_cpu_backend!(|B| {
        assert_eq!(select::<B>(), select::<Cpu>());
        assert_eq!(scatter::<B>(), scatter::<Cpu>());
    });
}
