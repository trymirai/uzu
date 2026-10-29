mod error;
mod kernel;
mod policy;
mod selection;
mod specialization;
mod trellis;

pub use error::GemmSpecializationError;
pub use kernel::GemmKernel;
pub use selection::GemmPlanError;
pub(super) use selection::{GemmProblem, MXU_INT8_MIN_BATCH, supports_integer_right_operand};

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
