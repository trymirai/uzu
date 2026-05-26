mod error;
mod kernel;
mod specialization;

pub use error::GemmSpecializationError;
pub use kernel::{GemmDispatchPath, GemmKernel};

// Matches `MxuMmaCore::THREADGROUP_BLOCK_K` in `common/mxu_mma_core.h`.
pub(crate) const MXU_THREADGROUP_BLOCK_K: u32 = 256;
