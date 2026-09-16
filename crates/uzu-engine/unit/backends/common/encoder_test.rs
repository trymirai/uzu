use std::time::{Duration, Instant};

use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{Backend, Context, Encoder, Kernels, kernel::TensorCopyKernel},
    data_type::DataType,
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, for_each_backend},
};

const LENGTH: usize = 1 << 20;

fn stamps_around_one_copy<B: Backend>() -> Duration {
    let context = B::Context::new().expect("Failed to create context");
    let kernel = <<B as Backend>::Kernels as Kernels>::TensorCopyKernel::new(&context, DataType::F32)
        .expect("Failed to create TensorCopyKernel");
    let src = alloc_allocation_with_data::<B, f32>(&context, &vec![1.0f32; LENGTH]);
    let mut dst = alloc_allocation::<B, f32>(&context, LENGTH);

    let before = Instant::now();
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let start = encoder.timestamp();
    kernel.encode(&src, &mut dst, LENGTH as u32, &mut encoder);
    let end = encoder.timestamp();
    assert!(
        start.get().is_none() && end.get().is_none(),
        "stamps must stay unresolved until the command buffer completed"
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let after = Instant::now();
    let (start, end) = (*start.get().expect("start stamp resolved"), *end.get().expect("end stamp resolved"));
    assert!(before <= start && start <= end && end <= after, "{start:?}..{end:?} outside {before:?}..{after:?}");
    end.saturating_duration_since(start)
}

#[uzu_test]
fn timestamps_resolve_after_completion() {
    for_each_backend!(|B| {
        let duration = stamps_around_one_copy::<B>();
        assert!(
            duration > Duration::ZERO && duration < Duration::from_secs(1),
            "{duration:?} on {}",
            std::any::type_name::<B>()
        );
    });
}
