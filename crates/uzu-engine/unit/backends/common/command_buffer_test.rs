use std::{any::type_name, time::Instant};

use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, Kernels, kernel::TensorAddScaleKernel,
    },
    data_type::DataType,
    tests::helpers::{create_buffer_with_data, create_context, for_each_backend},
};

const LENGTH: usize = 1 << 22;
const COLUMNS: usize = 1024;

fn encode_sampled_kernels<B: Backend>(
    context: &B::Context,
    timing: bool,
    names: &[&str],
    trailing_kernels: usize,
) -> (Instant, <B::CommandBuffer as CommandBuffer>::Completed, Instant) {
    let kernel = <<B as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(context, DataType::F32, true)
        .expect("Failed to create TensorAddScaleKernel");
    let bias = create_buffer_with_data::<B, f32>(context, &vec![1.0; COLUMNS]);
    let mut buffer = create_buffer_with_data::<B, f32>(context, &vec![1.0; LENGTH]);
    let mut encode_kernel = |encoding: &mut <B::CommandBuffer as CommandBuffer>::Encoding| {
        kernel.encode(
            None::<&<B as Backend>::GlobalBuffer>,
            &bias,
            &mut buffer,
            COLUMNS as u32,
            LENGTH as u32,
            1.0,
            encoding,
        )
    };

    let before = Instant::now();
    let mut encoding = context.create_command_buffer(Some("test"), None).unwrap();
    if timing {
        encoding.enable_timestamps().unwrap();
    }
    for name in names {
        encode_kernel(&mut encoding);
        encoding.sample_timestamp(&name.to_string());
    }
    for _ in 0..trailing_kernels {
        encode_kernel(&mut encoding);
    }
    let completed = encoding.end_encoding().submit().wait_until_completed().unwrap();
    (before, completed, Instant::now())
}

#[uzu_test]
fn timestamp_samples_measure_kernels() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let names = ["test/first", "test/second", "test/third"];
        let (before, completed, after) = encode_sampled_kernels::<B>(&context, true, &names, 0);

        let samples = completed.timestamps();
        assert_eq!(samples.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(), names);
        for (name, timestamp) in samples {
            assert!(
                before <= *timestamp && *timestamp <= after,
                "{name} at {timestamp:?} outside {before:?}..{after:?} on {}",
                type_name::<B>()
            );
        }
        for pair in samples.windows(2) {
            assert!(pair[0].1 < pair[1].1, "{samples:?} on {}", type_name::<B>());
        }
    });
}

#[uzu_test]
fn timestamp_samples_belong_to_their_command_buffer() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        for name in ["test/first command buffer", "test/second command buffer"] {
            let (_, completed, _) = encode_sampled_kernels::<B>(&context, true, &[name], 0);
            let names = completed.timestamps().iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>();
            assert_eq!(names, [name], "on {}", type_name::<B>());
        }
    });
}

#[uzu_test]
fn timestamp_samples_need_timing_enabled() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let (_, completed, _) = encode_sampled_kernels::<B>(&context, false, &["test/untimed"], 0);
        assert!(completed.timestamps().is_empty(), "on {}", type_name::<B>());
    });
}

#[uzu_test]
fn timestamp_samples_exclude_work_after_the_last_sample() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let (_, completed, _) = encode_sampled_kernels::<B>(&context, true, &["test/first", "test/second"], 10);

        let [(_, first), (_, second)] = completed.timestamps() else {
            panic!("{:?} on {}", completed.timestamps(), type_name::<B>())
        };
        assert!(first < second, "{first:?} {second:?} on {}", type_name::<B>());
        let block_time = second.duration_since(*first);
        assert!(
            block_time * 3 < completed.gpu_execution_time(),
            "{block_time:?} includes the work after the sample ({:?} total) on {}",
            completed.gpu_execution_time(),
            type_name::<B>()
        );
    });
}
