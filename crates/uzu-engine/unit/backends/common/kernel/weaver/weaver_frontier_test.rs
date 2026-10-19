use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::weaver::{FrontierIdx, TreeIdx},
            kernel::WeaverFrontierSelectKernel,
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

fn select<B: Backend>() -> Vec<u32> {
    let context = create_context::<B>();
    let mut frontier = vec![0; FrontierIdx::COUNT * 8];
    for (slot, (token, parent, depth, cum, key, active)) in [
        (9, 1, 1, 0x3f00_0000, 100, 1),
        (8, 0, 2, 0x3f00_0001, 100, 1),
        (7, 0, 2, 0x3f00_0002, 100, 1),
        (7, 0, 2, 0x3f00_0003, 100, 1),
        (2, 1, 3, 0x3f00_0004, 300, 1),
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
    let mut frontier = create_buffer_with_data::<B, u32>(&context, &frontier);
    let mut tree = create_buffer_with_data::<B, u32>(&context, &[55; TreeIdx::COUNT * 7]);
    let mut slot_ancestors = create_buffer_with_data::<B, u32>(&context, &(0u32..7 * 3).collect::<Vec<_>>());
    let mut token = create_buffer_with_data::<B, u32>(&context, &[66; 4]);
    let mut metadata = create_buffer_with_data::<B, u32>(&context, &[77; 3 * 4]);
    let mut ancestors = create_buffer_with_data::<B, u32>(&context, &[88; 4 * 3]);
    let mut valid = create_buffer_with_data::<B, u32>(&context, &[99; 4]);
    let candidate_pool_ids = create_buffer_with_data::<B, u32>(&context, &(0..12).collect::<Vec<_>>());
    let candidate_pool_scores =
        create_buffer_with_data::<B, f32>(&context, &(0..12).map(|value| value as f32).collect::<Vec<_>>());
    let mut candidate_ids = create_buffer_with_data::<B, u32>(&context, &[0; 4 * 3]);
    let mut candidate_scores = create_buffer_with_data::<B, f32>(&context, &[0.0; 4 * 3]);
    let kernel = <B::Kernels as Kernels>::WeaverFrontierSelectKernel::new(&context).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
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
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    [frontier, tree, slot_ancestors, token, metadata, ancestors, valid, candidate_ids]
        .iter()
        .flat_map(buffer_to_vec)
        .chain(buffer_to_vec::<B, f32>(&candidate_scores).into_iter().map(f32::to_bits))
        .collect()
}

#[uzu_test]
fn weaver_frontier_select_matches_cpu() {
    for_each_non_cpu_backend!(|B| {
        assert_eq!(select::<B>(), select::<Cpu>());
    });
}
