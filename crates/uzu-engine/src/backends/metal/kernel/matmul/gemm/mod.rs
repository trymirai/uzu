mod error;
mod kernel;
mod policy;
mod selection;
pub mod small_batch_units;
mod specialization;
mod trellis;

pub use error::GemmSpecializationError;
pub use kernel::GemmKernel;
pub use selection::{GemmPlanError, GemmProblem};
pub(super) use selection::{small_batch_format, small_m_weights};

use crate::backends::metal::gpu_types::gemm::GemmTiling;

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
