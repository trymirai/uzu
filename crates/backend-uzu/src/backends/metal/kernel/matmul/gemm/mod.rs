mod dispatch;
pub(crate) mod fp;
mod kernel;
pub(crate) mod quant;

pub use dispatch::GemmSpecializationError;
pub(crate) use dispatch::{GemmAlignmentAxes, GemmDispatch, GemmWeights};
pub(crate) use kernel::GemmKernel;

pub(crate) use crate::backends::common::gpu_types::gemm::{GemmComputeKind, GemmInputPrologueKind};

// Matches `MxuMmaCore::THREADGROUP_BLOCK_K` in `common/mxu_mma_core.h`.
pub(crate) const MXU_THREADGROUP_BLOCK_K: u32 = 256;
