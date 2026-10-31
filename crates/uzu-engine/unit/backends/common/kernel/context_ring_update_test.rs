use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        kernel::ContextRingUpdateKernel,
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_backend},
};

/// The CPU's rule on a `[offset, length, tokens..]` ring: each token takes the next free slot, or once the ring is
/// full the oldest one.
fn update(
    ring: &mut [u32],
    input: &[u32],
) {
    let capacity = ring.len() as u32 - 2;
    for &token in input {
        let slot = if ring[1] < capacity {
            let slot = (ring[0] + ring[1]) % capacity;
            ring[1] += 1;
            slot
        } else {
            let slot = ring[0];
            ring[0] = (slot + 1) % capacity;
            slot
        };
        ring[2 + slot as usize] = token;
    }
}

fn check<B: Backend>() {
    let context = create_context::<B>();
    let kernel = <B::Kernels as Kernels>::ContextRingUpdateKernel::new(&context).unwrap();
    for capacity in [1, 3, 16, 300] {
        // (1, 2) is no header of a one-token ring.
        for (offset, length) in
            [(0, 0), (1, 2), (capacity - 1, capacity)].into_iter().filter(|&(_, length)| length <= capacity)
        {
            for input_length in [0, 1, capacity - 1, capacity, capacity + 5, 600] {
                let mut expected: Vec<u32> = [offset, length].into_iter().chain(100_000..100_000 + capacity).collect();
                let mut ring = create_buffer_with_data::<B, u32>(&context, &expected);
                let inputs = [10_000, 20_000].map(|first| (first..first + input_length.max(1)).collect::<Vec<u32>>());
                let buffers = inputs.each_ref().map(|input| create_buffer_with_data::<B, u32>(&context, input));
                let mut command_buffer = context.create_command_buffer(None, None).unwrap();
                for (input, buffer) in inputs.iter().zip(&buffers) {
                    kernel.encode(buffer, &mut ring, capacity, input_length, &mut command_buffer);
                    update(&mut expected, &input[..input_length as usize]);
                }
                block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
                assert_eq!(
                    buffer_to_vec::<B, u32>(&context, &ring),
                    expected,
                    "capacity {capacity}, header ({offset}, {length}), input {input_length}"
                );
            }
        }
    }
}

/// Two updates back to back store the last `capacity` tokens and advance the header as the CPU does.
#[uzu_test]
fn context_ring_update_keeps_the_latest_tokens() {
    for_each_backend!(|B| check::<B>());
}
