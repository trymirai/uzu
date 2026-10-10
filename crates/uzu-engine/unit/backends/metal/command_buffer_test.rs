use std::time::Instant;

use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferPending, Context, Kernels, kernel::TensorAddScaleKernel,
        },
        metal::{Metal, MetalContext},
    },
    data_type::DataType,
    tests::helpers::create_buffer_with_data,
};

const LENGTH: usize = 1 << 22;
const COLUMNS: usize = 1024;

fn encode_blocks<B: Backend>(
    context: &B::Context,
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
    let mut encoding = context.create_command_buffer(Some("test"), None).unwrap();
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
    let context = MetalContext::new_with_timestamps().unwrap();
    let (before, completed, after) =
        encode_blocks::<Metal>(&context, Some("test/outer"), &["test/first", "test/second", "test/third"], 0);

    assert_eq!(*names(&completed), ["test/outer", "test/first", "test/second", "test/third"]);
    let spans = completed.timestamps();
    for span in spans {
        assert!(
            before <= span.start && span.start < span.end && span.end <= after,
            "{span:?} outside {before:?}..{after:?}"
        );
    }
    let [outer, first, second, third] = spans else {
        panic!("{spans:?}")
    };
    assert!(outer.start <= first.start && third.end <= outer.end, "{spans:?}");
    assert!(first.end <= second.start && second.end <= third.start, "{spans:?}");
}

#[uzu_test]
fn timestamp_spans_belong_to_their_command_buffer() {
    let context = MetalContext::new_with_timestamps().unwrap();
    for name in ["test/first command buffer", "test/second command buffer"] {
        let (_, completed, _) = encode_blocks::<Metal>(&context, None, &[name], 0);
        assert_eq!(*names(&completed), [name]);
    }
}

#[uzu_test]
fn timestamp_spans_need_a_timestamp_context() {
    let context = MetalContext::new().unwrap();
    let (_, completed, _) = encode_blocks::<Metal>(&context, None, &["test/untimed"], 0);
    assert!(completed.timestamps().is_empty());
}

#[uzu_test]
fn timestamp_spans_exclude_work_after_the_last_end() {
    let context = MetalContext::new_with_timestamps().unwrap();
    let (_, completed, _) = encode_blocks::<Metal>(&context, None, &["test/block"], 10);

    let [span] = completed.timestamps() else {
        panic!("{:?}", completed.timestamps())
    };
    assert!(span.start < span.end, "{span:?}");
    let block_time = span.end.duration_since(span.start);
    assert!(
        block_time * 3 < completed.gpu_execution_time(),
        "{block_time:?} includes the work after the block ({:?} total)",
        completed.gpu_execution_time()
    );
}
