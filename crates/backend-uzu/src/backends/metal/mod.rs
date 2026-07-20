mod backend;
mod buffer;
mod command_buffer;
mod context;
mod dense_buffer;
mod device_tier;
mod error;
mod kernel;
mod metal_extensions;
mod sparse;

use crate::backends::common::gpu_types::{ACTIVATION_QUANTIZATION_GROUP_SIZE, HADAMARD_TRANSFORM_BLOCK_SIZE};

pub const METAL_SIMD_SIZE: u32 = 32;

const _: () = {
    assert!(HADAMARD_TRANSFORM_BLOCK_SIZE == METAL_SIMD_SIZE as usize);
    assert!(ACTIVATION_QUANTIZATION_GROUP_SIZE == METAL_SIMD_SIZE);
};

pub use backend::Metal;
pub use context::MetalContext;
pub use kernel::matmul::GemmDispatchPath;
