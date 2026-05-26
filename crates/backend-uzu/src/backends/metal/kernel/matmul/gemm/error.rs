use thiserror::Error;

use crate::backends::common::gpu_types::gemm::GemmTiling;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GemmSpecializationError {
    #[error("simdgroup K={simdgroup_k} exceeds group size {group_size}")]
    SimdgroupKExceedsGroupSize {
        simdgroup_k: u32,
        group_size: u32,
    },
    #[error("quantized B requires transposed layout")]
    QuantizedRequiresTransposedB,
    #[error("tiling {tiling} does not match use_mxu={use_mxu}")]
    TilingUseMxuMismatch {
        tiling: GemmTiling,
        use_mxu: bool,
    },
}
