mod error;
mod kernel;
mod policy;
mod selection;
mod specialization;

pub use error::GemmSpecializationError;
pub use kernel::GemmKernel;
pub(super) use selection::GemmProblem;

use crate::backends::common::gpu_types::gemm::GemmTiling;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmEngine {
    Simdgroup,
    Mxu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmPlan {
    pub(super) engine: GemmEngine,
    pub(super) tiling: GemmTiling,
    pub(super) split_k: u32,
}
