use std::{any::type_name, time::Instant};

use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, Kernels, TimestampSampleEntry, kernel::TensorAddScaleKernel,
    },
    data_type::DataType,
    tests::helpers::{create_buffer_with_data, create_context, for_each_backend},
};

const LENGTH: usize = 1 << 22;
const COLUMNS: usize = 1024;

fn encode_blocks<B: Backend>(
    context: &B::Context,
    timing: bool,
    outer: Option<&str>,
    blocks: &[&str],
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
    let outer = outer.map(str::to_string);

    let before = Instant::now();
    let mut encoding = context.create_command_buffer(Some("test"), None).unwrap();
    if timing {
        encoding.enable_timestamps().unwrap();
    }
    if let Some(outer) = &outer {
        encoding.sample_start_timestamp(outer);
    }
    for block in blocks.iter().map(|block| block.to_string()) {
        encoding.sample_start_timestamp(&block);
        encode_kernel(&mut encoding);
        encoding.sample_end_timestamp(&block);
    }
    if let Some(outer) = &outer {
        encoding.sample_end_timestamp(outer);
    }
    for _ in 0..trailing_kernels {
        encode_kernel(&mut encoding);
    }
    let completed = encoding.end_encoding().submit().wait_until_completed().unwrap();
    (before, completed, Instant::now())
}

fn entries<Completed: CommandBufferCompleted>(completed: &Completed) -> Vec<(&'static str, &str, Instant)> {
    completed
        .timestamps()
        .iter()
        .map(|(entry, timestamp)| match entry {
            TimestampSampleEntry::Start(name) => ("start", name.as_str(), *timestamp),
            TimestampSampleEntry::End(name) => ("end", name.as_str(), *timestamp),
        })
        .collect()
}

#[uzu_test]
fn timestamp_samples_measure_blocks() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let (before, completed, after) =
            encode_blocks::<B>(&context, true, Some("test/outer"), &["test/first", "test/second", "test/third"], 0);

        let entries = entries(&completed);
        let kinds_and_names = entries.iter().map(|&(kind, name, _)| (kind, name)).collect::<Vec<_>>();
        assert_eq!(
            kinds_and_names,
            [
                ("start", "test/outer"),
                ("start", "test/first"),
                ("end", "test/first"),
                ("start", "test/second"),
                ("end", "test/second"),
                ("start", "test/third"),
                ("end", "test/third"),
                ("end", "test/outer"),
            ],
            "on {}",
            type_name::<B>()
        );
        for (kind, name, timestamp) in &entries {
            assert!(
                before <= *timestamp && *timestamp <= after,
                "{kind} {name} at {timestamp:?} outside {before:?}..{after:?} on {}",
                type_name::<B>()
            );
        }
        let time = |index: usize| entries[index].2;
        for (start, end) in [(1, 2), (3, 4), (5, 6)] {
            assert!(time(start) < time(end), "{entries:?} on {}", type_name::<B>());
        }
        assert!(time(2) <= time(3) && time(4) <= time(5), "{entries:?} on {}", type_name::<B>());
        assert!(time(0) <= time(1) && time(6) <= time(7), "{entries:?} on {}", type_name::<B>());
    });
}

#[uzu_test]
fn timestamp_samples_belong_to_their_command_buffer() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        for name in ["test/first command buffer", "test/second command buffer"] {
            let (_, completed, _) = encode_blocks::<B>(&context, true, None, &[name], 0);
            let kinds_and_names =
                entries(&completed).into_iter().map(|(kind, name, _)| (kind, name)).collect::<Vec<_>>();
            assert_eq!(kinds_and_names, [("start", name), ("end", name)], "on {}", type_name::<B>());
        }
    });
}

#[uzu_test]
fn timestamp_samples_need_timing_enabled() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let (_, completed, _) = encode_blocks::<B>(&context, false, None, &["test/untimed"], 0);
        assert!(completed.timestamps().is_empty(), "on {}", type_name::<B>());
    });
}

#[uzu_test]
fn timestamp_samples_exclude_work_after_the_last_end() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let (_, completed, _) = encode_blocks::<B>(&context, true, None, &["test/block"], 10);

        let [(_, _, start), (_, _, end)] = entries(&completed)[..] else {
            panic!("{:?} on {}", completed.timestamps(), type_name::<B>())
        };
        assert!(start < end, "{start:?} {end:?} on {}", type_name::<B>());
        let block_time = end.duration_since(start);
        assert!(
            block_time * 3 < completed.gpu_execution_time(),
            "{block_time:?} includes the work after the block ({:?} total) on {}",
            completed.gpu_execution_time(),
            type_name::<B>()
        );
    });
}
