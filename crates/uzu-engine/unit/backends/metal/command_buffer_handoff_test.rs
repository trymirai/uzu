use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures::executor::block_on;
use metal::{MTLDeviceExt, MTLSharedEvent};
use parking_lot::Mutex;
use uzu_engine_macros::uzu_test;

use super::DebugGroups;
use crate::{
    backends::{
        common::{
            CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, CpuBuffer,
            GlobalBufferCpuGpu,
        },
        metal::{Metal, command_buffer::Segments},
    },
    tests::helpers::create_context,
};

/// Work encoded after a `wait_event` runs once the event reaches its value, not before.
#[uzu_test]
fn work_after_wait_event_runs_once_signaled() {
    let context = create_context::<Metal>();
    let event = context.device.new_shared_event().unwrap();
    let mut buffer = context.create_buffer_cpu_gpu(1 << 20).unwrap();
    buffer.as_slice_mut::<u8>().fill(0);
    let pointer = buffer.cpu_ptr();
    let mut buffer = buffer.handoff().unwrap();

    let mut encoding = context.create_command_buffer(None, None).unwrap();
    let mut root = encoding.debug_group("root");
    let mut decoder = root.debug_group("decoder");
    let mut layer = decoder.debug_group("transformer layer 0");
    let mut attention = layer.debug_group("attention");
    let mut core = attention.debug_group("attention core");
    let mut projection = core.debug_group("matrix projection");
    let mut nested = projection.debug_group("a long nested label that spills safely 深度");
    nested.signal_event(&event, 1).unwrap();
    nested.wait_event(&event, 3).unwrap();
    nested.encode_fill(&mut buffer, 2);
    drop(nested);
    drop(projection);
    drop(core);
    drop(attention);
    drop(layer);
    drop(decoder);
    drop(root);
    let pending = encoding.end_encoding().submit();

    let read = || unsafe { *pointer.as_ptr().cast::<u8>() };
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!((event.signaled_value(), read()), (1, 0));
    event.set_signaled_value(3);
    block_on(pending.wait_until_completed()).unwrap();
    assert_eq!(read(), 2);
}

/// Common model paths fit in the inline storage, while unusually deep and long names retain exact
/// UTF-8 contents through the heap fallback used when a later segment replays the stack.
#[uzu_test]
fn debug_group_labels_keep_inline_and_heap_fallback_paths() {
    let mut ordinary = DebugGroups::default();
    for label in ["decoder", "transformer layer 17", "attention", "attention core"] {
        ordinary.push(label);
    }
    assert!(ordinary.is_inline());
    assert_eq!(
        ordinary.iter().collect::<Vec<_>>(),
        vec!["decoder", "transformer layer 17", "attention", "attention core"]
    );

    let mut longest_common = DebugGroups::default();
    longest_common.push("per layer embedding projection");
    assert!(longest_common.is_inline());

    let mut fallback = DebugGroups::default();
    let labels = [
        "root",
        "decoder",
        "transformer layer 17",
        "attention",
        "attention core",
        "matrix projection",
        "a long nested label that spills safely 深度",
    ];
    for label in labels {
        fallback.push(label);
    }
    assert!(fallback.0.spilled());
    assert!(fallback.0.last().unwrap().spilled());
    assert_eq!(fallback.iter().collect::<Vec<_>>(), labels.to_vec());
}

/// A failure recorded while the encoding is in flight fails awaiting it.
#[uzu_test]
fn a_failed_encoding_errors_when_awaited() {
    let encoding = create_context::<Metal>().create_command_buffer(None, None).unwrap();
    encoding.failure().fail("injected");
    assert!(block_on(encoding.end_encoding().submit().wait_until_completed()).is_err());
}

struct FinishSentinel(Arc<AtomicUsize>);

impl Drop for FinishSentinel {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// The report owner keeps its one finalizer until the last feedback, regardless of report order.
#[uzu_test]
fn segment_reports_finalize_once_after_out_of_order_feedback() {
    let segments = Arc::new(Segments::default());
    segments.state.lock().unreported = 2;

    let outcome = Arc::new(Mutex::new(None));
    let finalizations = Arc::new(AtomicUsize::new(0));
    let sentinel_drops = Arc::new(AtomicUsize::new(0));
    let callback_outcome = outcome.clone();
    let callback_finalizations = finalizations.clone();
    segments.finish_when_reported(FinishSentinel(sentinel_drops.clone()), move |error, gpu_time| {
        *callback_outcome.lock() = Some((error, gpu_time));
        callback_finalizations.fetch_add(1, Ordering::SeqCst);
    });

    // The later segment reports first. Its error remains the first feedback error, while the
    // aggregate span still includes the earlier segment's start time.
    segments.report(Some("later segment failed".to_string()), 20.0, 30.0);
    assert!(outcome.lock().is_none());
    assert_eq!(sentinel_drops.load(Ordering::SeqCst), 0);
    assert_eq!(finalizations.load(Ordering::SeqCst), 0);

    segments.report(Some("earlier segment failed".to_string()), 10.0, 15.0);
    assert_eq!(*outcome.lock(), Some((Some("later segment failed".to_string()), Duration::from_secs(20))));
    assert_eq!(sentinel_drops.load(Ordering::SeqCst), 1);
    assert_eq!(finalizations.load(Ordering::SeqCst), 1);
}
