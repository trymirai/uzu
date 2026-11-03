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
fn encoding_without_handoff_is_unchanged() {
    let context = create_context::<Metal>();
    let mut buffer = context.create_buffer_cpu_gpu(SIZE).unwrap().handoff();
    let mut encoding = context.create_command_buffer(None, None).unwrap();
    encoding.encode_fill(&mut buffer, 3);
    let completed = block_on(encoding.end_encoding().submit().wait_until_completed()).unwrap();
    assert!(completed.gpu_execution_time().unwrap() < Duration::from_secs(1));
}
