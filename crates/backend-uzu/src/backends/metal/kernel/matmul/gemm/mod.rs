mod dispatch;
mod error;
mod kernel;
mod specialization;
mod weights;

pub use error::GemmSpecializationError;
pub(crate) use kernel::GemmKernel;

// Matches `MxuMmaCore::THREADGROUP_BLOCK_K` in `common/mxu_mma_core.h`.
pub(crate) const MXU_THREADGROUP_BLOCK_K: u32 = 256;
