mod interleaved_rows;
mod mxu_tiled;

pub use mxu_tiled::words_per_thread as mxu_tiled_words_per_thread;

use crate::backends::{
    common::{Backend, kernel::matmul::MatmulShape},
    metal::{
        Metal, context::MetalContext, error::MetalError, gpu_types::gemm::GemmBLayout,
        kernel::matmul::MatmulMetalKernel,
    },
};

/// Picks the layout the weights are read in, relays them in place and returns it with their code signedness.
pub fn prepare(
    kernel: &MatmulMetalKernel,
    shape: &MatmulShape,
    weights: &mut <Metal as Backend>::GlobalGpuBuffer,
    context: &MetalContext,
) -> Result<(GemmBLayout, bool), MetalError> {
    if mxu_tiled::eligible(kernel, shape, context) {
        return Ok((GemmBLayout::MxuTiled, mxu_tiled::prepare(shape, weights, context)?));
    }
    if interleaved_rows::prepare(kernel, shape, weights, context)? {
        Ok((GemmBLayout::InterleavedRows, shape.signed_codes))
    } else {
        Ok((GemmBLayout::RowMajor, shape.signed_codes))
    }
}
