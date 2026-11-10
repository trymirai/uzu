use futures::executor::block_on;

use crate::backends::{
    common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, CpuBuffer,
        kernel::matmul::{MatmulKernel, MatmulShape},
    },
    metal::{
        Metal,
        context::MetalContext,
        error::MetalError,
        gpu_types::matmul::{MXU_TILED_BLOCK_ROWS, MXU_TILED_PREPARE_THREADS, MXU_TILED_W2_BLOCK_ROWS},
        kernel::{GemmPrepareBLayoutMetalKernel, matmul::MatmulMetalKernel},
    },
};

fn block_rows(bits: u32) -> u32 {
    if bits == 2 {
        MXU_TILED_W2_BLOCK_ROWS
    } else {
        MXU_TILED_BLOCK_ROWS
    }
}

pub fn words_per_thread(
    k: u32,
    bits: u32,
) -> Option<u32> {
    let words_per_thread = (block_rows(bits) * (k * bits / u32::BITS)).div_ceil(MXU_TILED_PREPARE_THREADS);
    [32, 72, 160].into_iter().find(|&variant| variant >= words_per_thread)
}

fn prepared_shape(shape: &MatmulShape) -> MatmulShape {
    MatmulShape {
        signed_codes: shape.signed_codes || shape.b_bits == Some(8),
        ..*shape
    }
}

pub fn eligible(
    kernel: &MatmulMetalKernel,
    shape: &MatmulShape,
    context: &MetalContext,
) -> bool {
    kernel.select_activation_quantization(&prepared_shape(shape), context).is_some()
}

/// Relays the weights in place and returns their code signedness.
pub fn prepare(
    shape: &MatmulShape,
    weights: &mut <Metal as Backend>::GlobalGpuBuffer,
    context: &MetalContext,
) -> Result<bool, MetalError> {
    let prepared_shape = prepared_shape(shape);
    if !shape.signed_codes && prepared_shape.signed_codes {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                weights.cpu_ptr().as_ptr().cast::<u8>(),
                crate::backends::common::Buffer::size(weights),
            )
        };
        let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(bytes);
        words.iter_mut().for_each(|word| *word ^= 0x8080_8080_8080_8080);
        prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= 0x80);
    }
    let bits = shape.b_bits.expect("quantized weights");
    let row_words = shape.k * bits / u32::BITS;
    let variant = words_per_thread(shape.k, bits).expect("int8 eligibility admits only tileable weights");
    let mut command_buffer = context.create_command_buffer(Some("prepare GEMM B layout"), None)?;
    GemmPrepareBLayoutMetalKernel::new(context, bits, variant)?.encode(
        weights,
        row_words,
        shape.n / block_rows(bits),
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed())?;
    Ok(prepared_shape.signed_codes)
}
