mod backend;
mod buffer;
mod command_buffer;
mod context;
mod dense_buffer;
mod error;
mod kernel;
mod metal_extensions;
mod sparse;

use metal::MTLGPUFamily;

use crate::backends::common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE;

const METAL_SIMD_SIZE: u32 = 32;
const LARGE_MIN_GPU_CORES: u32 = 30;

fn newest_supported_apple_gpu_family(mut supports_family: impl FnMut(MTLGPUFamily) -> bool) -> Option<MTLGPUFamily> {
    [MTLGPUFamily::Apple10, MTLGPUFamily::Apple9, MTLGPUFamily::Apple8]
        .into_iter()
        .find(|family| supports_family(*family))
}

const _: () = {
    assert!(HADAMARD_TRANSFORM_BLOCK_SIZE == METAL_SIMD_SIZE);
};

pub use backend::Metal;
pub use context::MetalContext;
#[cfg(test)]
pub use kernel::matmul::gemm::GemmEngine;
