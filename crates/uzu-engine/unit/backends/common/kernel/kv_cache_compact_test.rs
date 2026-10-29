use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::KVCacheCompactKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_non_cpu_backend},
};

fn run_kv_cache_compact<B: Backend>(
    keys: &[f32],
    accepted_indices: &[u32],
    suffix_base: u32,
    ring_write_start: u32,
    ring_capacity: u32,
) -> Vec<f32> {
    let context = block_on(B::Context::new()).expect("create context");
    let kernel = <<B as Backend>::Kernels as Kernels>::KVCacheCompactKernel::new(&context, DataType::F32)
        .expect("create KV cache compact kernel");

    let mut keys_buffer = create_buffer_with_data::<B, f32>(&context, keys);
    let mut values_buffer = create_buffer_with_data::<B, f32>(&context, keys);
    let accepted_indices = create_buffer_with_data::<B, u32>(&context, accepted_indices);

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(
        &mut keys_buffer,
        &mut values_buffer,
        &accepted_indices,
        suffix_base,
        ring_write_start,
        ring_capacity,
        1,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    buffer_to_vec::<B, f32>(&context, &keys_buffer)
}

#[uzu_test]
fn kv_cache_compact_shifts_full_cache_rows_by_rank() {
    let keys: Vec<f32> = (0..8).map(|i| i as f32).collect();
    // accepted suffix path of rows [0, 2, 3], deepest first
    let accepted_indices: Vec<u32> = vec![3, 3, 2, 0];

    let expected = {
        let mut keys = keys.clone();
        keys[5] = 6.0;
        keys[6] = 7.0;
        keys
    };

    let actual = run_kv_cache_compact::<Cpu>(&keys, &accepted_indices, 4, 0, 0);
    assert_eq!(actual, expected);

    for_each_non_cpu_backend!(|B| {
        let actual = run_kv_cache_compact::<B>(&keys, &accepted_indices, 4, 0, 0);
        assert_eq!(actual, expected);
    });
}

#[uzu_test]
fn kv_cache_compact_ring_skips_dead_copies() {
    // ring of capacity 2 holding rows 0 and 1; suffix rows 0..=2 all accepted
    // so only the last 2 survive and the first rank is dead
    let keys: Vec<f32> = (0..5).map(|i| i as f32).collect();
    let accepted_indices: Vec<u32> = vec![3, 2, 1, 0];

    let expected = {
        let mut keys = keys.clone();
        keys[0] = 3.0;
        keys[1] = 4.0;
        keys
    };

    let actual = run_kv_cache_compact::<Cpu>(&keys, &accepted_indices, 2, 1, 2);
    assert_eq!(actual, expected);

    for_each_non_cpu_backend!(|B| {
        let actual = run_kv_cache_compact::<B>(&keys, &accepted_indices, 2, 1, 2);
        assert_eq!(actual, expected);
    });
}
