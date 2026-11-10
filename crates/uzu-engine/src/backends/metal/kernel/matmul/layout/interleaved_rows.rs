use futures::executor::block_on;

use crate::backends::{
    common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        kernel::matmul::{MatmulShape, QuantParamsLayout},
    },
    metal::{
        Metal,
        context::MetalContext,
        error::MetalError,
        gpu_types::{
            gemm::GemmBLayout,
            matmul::{INTERLEAVED_ROWS_RELAYOUT_THREADS, INTERLEAVED_ROWS_STRIP_ROWS},
        },
        kernel::{
            PrepareInterleavedRowsMetalKernel,
            matmul::{MatmulMetalKernel, gemm::small_batch_format},
        },
    },
};

const CODE_BITS: u32 = 4;

fn words_per_thread(row_words: u32) -> Option<u32> {
    let words_per_thread = (INTERLEAVED_ROWS_STRIP_ROWS * row_words).div_ceil(INTERLEAVED_ROWS_RELAYOUT_THREADS);
    [16, 48, 96].into_iter().find(|&variant| variant >= words_per_thread)
}

/// Relays W4 weights read by the packed small-batch GEMM in place and returns true, or returns false if they are not.
pub fn prepare(
    kernel: &MatmulMetalKernel,
    shape: &MatmulShape,
    weights: &mut <Metal as Backend>::GlobalGpuBuffer,
    context: &MetalContext,
) -> Result<bool, MetalError> {
    let eligible = !context.supports_mxu
        && kernel.is_all_bf16()
        && shape.params_layout == Some(QuantParamsLayout::GroupOutput)
        && shape.n.is_multiple_of(INTERLEAVED_ROWS_STRIP_ROWS)
        && small_batch_format(*shape, GemmBLayout::InterleavedRows).is_some();
    let row_words = shape.k * CODE_BITS / u32::BITS;
    if !eligible {
        return Ok(false);
    }
    let Some(variant) = words_per_thread(row_words) else {
        return Ok(false);
    };
    let mut command_buffer = context.create_command_buffer(Some("prepare interleaved-row weights"), None)?;
    PrepareInterleavedRowsMetalKernel::new(context, variant)?.encode(
        weights,
        row_words,
        shape.n / INTERLEAVED_ROWS_STRIP_ROWS,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed())?;
    Ok(true)
}
