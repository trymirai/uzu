mod error;
mod kernel;
mod selection;
mod specialization;

pub use error::GemmSpecializationError;
pub use kernel::GemmKernel;
pub(super) use selection::GemmProblem;

use crate::backends::common::gpu_types::gemm::GemmTiling;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GemmEngine {
    Simdgroup,
    Mxu,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GemmPlan {
    engine: GemmEngine,
    tiling: GemmTiling,
    split_k: u32,
}
