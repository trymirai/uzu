use criterion::Bencher;
use futures::executor::block_on;

use crate::backends::common::{
    Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
    CommandBufferPending, Context, capture_matcher::CAPTURE_MATCHER,
};

pub fn iter_encode_loop<B: Backend, F>(
    context: &B::Context,
    bencher: &mut Bencher,
    encode: F,
) where
    F: FnMut(&mut <B::CommandBuffer as CommandBuffer>::Encoding),
{
    iter_encode_loop_named::<B, _>(context, bencher, "unnamed_benchmark", encode);
}

pub fn iter_encode_loop_named<B: Backend, F>(
    context: &B::Context,
    bencher: &mut Bencher,
    benchmark_path: &str,
    mut encode: F,
) where
    F: FnMut(&mut <B::CommandBuffer as CommandBuffer>::Encoding),
{
    bencher.iter_custom(|n_iters| {
        let _capture_span = CAPTURE_MATCHER
            .capture::<B>(context, benchmark_path)
            .unwrap_or_else(|error| panic!("failed to capture benchmark {benchmark_path}: {error}"));
        let mut command_buffer = context.create_command_buffer(Some(benchmark_path), None).unwrap();
        for _ in 0..n_iters {
            encode(&mut command_buffer);
        }
        let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        completed.gpu_execution_time().unwrap()
    });
}
