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

    let before = Instant::now();
    let mut encoding = context.create_command_buffer(Some("test"), None, timing).unwrap();
    if let Some(outer) = outer {
        encoding.sample_start_timestamp(outer);
    }
    for &block in blocks {
        encoding.sample_start_timestamp(block);
        encode_kernel(&mut encoding);
        encoding.sample_end_timestamp();
    }
    if outer.is_some() {
        encoding.sample_end_timestamp();
    }
    for _ in 0..trailing_kernels {
        encode_kernel(&mut encoding);
    }
    let completed = encoding.end_encoding().submit().wait_until_completed().unwrap();
    (before, completed, Instant::now())
}

fn names<Completed: CommandBufferCompleted>(completed: &Completed) -> Box<[&str]> {
    completed.timestamps().iter().map(|span| span.name.as_str()).collect()
}

#[uzu_test]
fn timestamp_spans_measure_blocks() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let (before, completed, after) =
            encode_blocks::<B>(&context, true, Some("test/outer"), &["test/first", "test/second", "test/third"], 0);

        assert_eq!(
            *names(&completed),
            ["test/outer", "test/first", "test/second", "test/third"],
            "on {}",
            type_name::<B>()
        );
        let spans = completed.timestamps();
        for span in spans {
            assert!(
                before <= span.start && span.start < span.end && span.end <= after,
                "{span:?} outside {before:?}..{after:?} on {}",
                type_name::<B>()
            );
        }
        let [outer, first, second, third] = spans else {
            panic!("{spans:?} on {}", type_name::<B>())
        };
        assert!(outer.start <= first.start && third.end <= outer.end, "{spans:?} on {}", type_name::<B>());
        assert!(first.end <= second.start && second.end <= third.start, "{spans:?} on {}", type_name::<B>());
    });
}

#[uzu_test]
fn timestamp_spans_belong_to_their_command_buffer() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        for name in ["test/first command buffer", "test/second command buffer"] {
            let (_, completed, _) = encode_blocks::<B>(&context, true, None, &[name], 0);
            assert_eq!(*names(&completed), [name], "on {}", type_name::<B>());
        }
    });
}

#[uzu_test]
fn timestamp_spans_need_timing_enabled() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let (_, completed, _) = encode_blocks::<B>(&context, false, None, &["test/untimed"], 0);
        assert!(completed.timestamps().is_empty(), "on {}", type_name::<B>());
    });
}

#[uzu_test]
fn timestamp_spans_exclude_work_after_the_last_end() {
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let (_, completed, _) = encode_blocks::<B>(&context, true, None, &["test/block"], 10);

        let [span] = completed.timestamps() else {
            panic!("{:?} on {}", completed.timestamps(), type_name::<B>())
        };
        assert!(span.start < span.end, "{span:?} on {}", type_name::<B>());
        let block_time = span.end.duration_since(span.start);
        assert!(
            block_time * 3 < completed.gpu_execution_time(),
            "{block_time:?} includes the work after the block ({:?} total) on {}",
            completed.gpu_execution_time(),
            type_name::<B>()
        );
    });
}
