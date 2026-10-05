use std::ptr::null_mut;

use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        amdgpu::{
            Amdgpu,
            hip::{hip, hip_call},
        },
        common::{CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
    },
    tests::helpers::{buffer_to_vec, create_buffer, create_context},
};

/// The zero-fill of a new scratch page runs before the commands that use the page. It used to be a hipMemset,
/// which goes to the legacy null stream, and the context's stream is non-blocking: it does not wait for the null
/// stream. With the null stream busy (contexts of other tests) the zero-fill landed after the first kernel had
/// written the page, and `radix_top_k_small_matches_cpu` failed in about half of the parallel test runs.
///
/// Here a blocking stream is kept busy: the legacy null stream waits for it, so a fill on the null stream would
/// land late in every round (`probe/host/null_stream_race.py --flood-stream blocking`). The test itself issues
/// nothing on the null stream: null-stream work of its own deadlocked HIP when tests ran in parallel. Its raw HIP
/// calls come after its contexts exist: a first hipMalloc from a thread that had not created a context, racing with
/// context creation on other threads, deadlocked HIP as well.
#[uzu_test]
fn new_scratch_page_is_zeroed_before_first_use() {
    const ROUNDS: usize = 8;
    const PAGE_BYTES: usize = 1 << 20;
    const FLOOD_BYTES: usize = 512 << 20;
    const FLOOD_FILLS: i32 = 4;
    const HIP_STREAM_DEFAULT: u32 = 0;
    // one context per round: the page cache of each is empty, so the scratch of each round is a new page
    let contexts: Vec<_> = (0..ROUNDS).map(|_| create_context::<Amdgpu>()).collect();
    let hip = hip().unwrap();
    let mut flood = null_mut();
    hip_call!(hip, hipMalloc(&mut flood, FLOOD_BYTES)).unwrap();
    let mut flood_stream = null_mut();
    hip_call!(hip, hipStreamCreateWithFlags(&mut flood_stream, HIP_STREAM_DEFAULT)).unwrap();
    let mut lost = 0;
    for context in contexts {
        for value in 0..FLOOD_FILLS {
            hip_call!(hip, hipMemsetAsync(flood, value, FLOOD_BYTES, flood_stream)).unwrap();
        }
        let pool = context.create_allocation_pool();
        let mut command_buffer = context.create_command_buffer(None, Some(pool.clone())).unwrap();
        let mut page = command_buffer.allocate_scratch(PAGE_BYTES).unwrap();
        command_buffer.encode_fill(&mut page, 0x5A);
        command_buffer.end_encoding().submit().wait_until_completed().unwrap();
        // the flood and anything queued behind it have run by now
        hip_call!(hip, hipDeviceSynchronize()).unwrap();
        let mut readback = create_buffer::<Amdgpu, u8>(&context, PAGE_BYTES);
        let mut command_buffer = context.create_command_buffer(None, Some(pool)).unwrap();
        command_buffer.encode_copy(&page, &mut readback);
        command_buffer.end_encoding().submit().wait_until_completed().unwrap();
        if buffer_to_vec::<Amdgpu, u8>(&readback).iter().any(|&byte| byte != 0x5A) {
            lost += 1;
        }
    }
    hip_call!(hip, hipStreamDestroy(flood_stream)).unwrap();
    hip_call!(hip, hipFree(flood)).unwrap();
    assert_eq!(lost, 0, "the zero-fill of a new scratch page overwrote the first command's write");
}
