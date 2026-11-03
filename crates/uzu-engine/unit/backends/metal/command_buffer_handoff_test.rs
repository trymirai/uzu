use std::time::{Duration, Instant};

use futures::executor::block_on;
use metal::{MTLDeviceExt, MTLSharedEvent};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            CpuBuffer, GlobalBufferCpuGpu,
        },
        metal::Metal,
    },
    tests::helpers::create_context,
};

const SIZE: usize = 1 << 20;

fn wait_for(
    condition: impl Fn() -> bool,
    timeout: Duration,
) -> bool {
    let started = Instant::now();
    while !condition() {
        if started.elapsed() > timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

#[uzu_test]
fn handoff_signals_after_prior_work_and_holds_later_work_until_released() {
    let context = create_context::<Metal>();
    let event = context.device.new_shared_event().unwrap();
    let mut before = context.create_buffer_cpu_gpu(SIZE).unwrap();
    let mut after = context.create_buffer_cpu_gpu(SIZE).unwrap();
    before.as_slice_mut::<u8>().fill(0);
    after.as_slice_mut::<u8>().fill(0);
    let before_ptr = before.cpu_ptr();
    let after_ptr = after.cpu_ptr();
    let mut before = before.handoff();
    let mut after = after.handoff();

    let mut encoding = context.create_command_buffer(None, None).unwrap().with_timestamps();
    {
        let encoding = &mut *encoding.debug_group("outer");
        encoding.encode_fill(&mut before, 1);
        encoding.handoff(&event, Some(1), Some(2)).unwrap();
        let encoding = &mut *encoding.debug_group("inner");
        encoding.encode_fill(&mut after, 2);
    }
    let pending = encoding.end_encoding().submit();

    let read = |pointer: std::ptr::NonNull<std::ffi::c_void>| unsafe { *pointer.as_ptr().cast::<u8>().add(SIZE - 1) };
    assert!(wait_for(|| event.signaled_value() >= 1, Duration::from_secs(5)), "handoff never signaled");
    assert_eq!(read(before_ptr), 1, "work before the handoff must be complete when it signals");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(read(after_ptr), 0, "work after the handoff must wait for the event");

    event.set_signaled_value(2);
    let completed = block_on(pending.wait_until_completed()).unwrap();
    assert_eq!(read(after_ptr), 2);

    let names: Vec<&str> = completed.timestamps().iter().map(|span| span.name.as_str()).collect();
    assert!(names.contains(&"outer") && names.contains(&"outer/inner"), "spans: {names:?}");
    let total = completed.gpu_execution_time().unwrap();
    assert!(total >= Duration::from_millis(40), "gpu time should span both command buffers: {total:?}");
}

#[uzu_test]
fn handoff_on_second_lane_signals_after_prior_work_and_holds_later_work() {
    let context = create_context::<Metal>();
    let event = context.device.new_shared_event().unwrap();
    let mut before = context.create_buffer_cpu_gpu(SIZE).unwrap();
    let mut after = context.create_buffer_cpu_gpu(SIZE).unwrap();
    before.as_slice_mut::<u8>().fill(0);
    after.as_slice_mut::<u8>().fill(0);
    let before_ptr = before.cpu_ptr();
    let after_ptr = after.cpu_ptr();
    let mut before = before.handoff();
    let mut after = after.handoff();

    let mut encoding = context.create_command_buffer(None, None).unwrap().with_timestamps();
    // Lane 1 is a queue of its own; point the encoding at it as `begin_independent_chunk` would.
    encoding.queue = context.lane(1).unwrap().queue.clone();
    encoding.lane = 1;
    {
        let encoding = &mut *encoding.debug_group("outer");
        encoding.encode_fill(&mut before, 1);
        encoding.handoff(&event, Some(1), Some(2)).unwrap();
        let encoding = &mut *encoding.debug_group("inner");
        encoding.encode_fill(&mut after, 2);
    }
    let pending = encoding.end_encoding().submit();

    let read = |pointer: std::ptr::NonNull<std::ffi::c_void>| unsafe { *pointer.as_ptr().cast::<u8>().add(SIZE - 1) };
    assert!(wait_for(|| event.signaled_value() >= 1, Duration::from_secs(5)), "handoff never signaled");
    assert_eq!(read(before_ptr), 1, "work before the handoff must be complete when it signals");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(read(after_ptr), 0, "work after the handoff must wait for the event");

    event.set_signaled_value(2);
    let completed = block_on(pending.wait_until_completed()).unwrap();
    assert_eq!(read(after_ptr), 2);

    let names: Vec<&str> = completed.timestamps().iter().map(|span| span.name.as_str()).collect();
    assert!(names.contains(&"outer") && names.contains(&"outer/inner"), "spans: {names:?}");
    let total = completed.gpu_execution_time().unwrap();
    assert!(total >= Duration::from_millis(40), "gpu time should span both command buffers: {total:?}");
}

/// A fill on lane 0 and a copy of the filled buffer on lane 1, ordered by lane 0's order event:
/// the copy must see the whole fill.
#[uzu_test]
fn lane_order_event_holds_the_other_lane_until_the_fill_is_done() {
    use metal::MTL4CommandQueue;
    const BYTES: usize = 256 << 20;
    let context = create_context::<Metal>();
    let (lane0, lane1) = (context.lane(0).unwrap(), context.lane(1).unwrap());
    for _ in 0..5 {
        let mut source = context.create_buffer_cpu_gpu(BYTES).unwrap();
        let mut destination = context.create_buffer_cpu_gpu(BYTES).unwrap();
        source.as_slice_mut::<u8>().fill(0);
        destination.as_slice_mut::<u8>().fill(0);
        let destination_ptr = destination.cpu_ptr();
        let mut source = source.handoff();
        let mut destination = destination.handoff();

        let value = lane0.reserve_values(1);
        let mut producer = context.create_command_buffer(None, None).unwrap();
        producer.encode_fill(&mut source, 7);
        producer.handoff(&lane0.event, Some(value), None).unwrap();
        producer.queue.signal_event_value(&lane0.order, value);
        let producer = producer.end_encoding().submit();

        let mut consumer = context.create_command_buffer(None, None).unwrap();
        consumer.queue = lane1.queue.clone();
        consumer.lane = 1;
        consumer.pending_waits.push((lane0.order.clone(), value));
        consumer.encode_copy(&source, &mut destination);
        let consumer = consumer.end_encoding().submit();
        block_on(consumer.wait_until_completed()).unwrap();
        block_on(producer.wait_until_completed()).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts(destination_ptr.as_ptr().cast::<u8>(), BYTES) };
        assert!(bytes.iter().all(|byte| *byte == 7), "the copy on lane 1 ran before the fill on lane 0 finished");
    }
}
