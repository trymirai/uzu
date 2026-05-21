mod dispatch;
mod error;
mod kernel;
mod specialization;
mod weights;

pub use error::GemmSpecializationError;
pub(crate) use kernel::GemmKernel;

#[allow(unused_imports)]
pub(crate) use crate::backends::common::gpu_types::gemm::GemmInputPrologueKind;

// Matches `MxuMmaCore::THREADGROUP_BLOCK_K` in `common/mxu_mma_core.h`.
pub(crate) const MXU_THREADGROUP_BLOCK_K: u32 = 256;
