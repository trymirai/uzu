use thiserror::Error;

use crate::backends::common::gpu_types::gemm::GemmTiling;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GemmSpecializationError {
    #[error("unsupported A group size {a_group_size:?}; expected None or 128")]
    InvalidAGroupSize {
        a_group_size: Option<u32>,
    },
    #[error("quantized B requires transposed layout")]
    QuantizedRequiresTransposedB,
    #[error("tiling {tiling} does not match use_mxu={use_mxu}")]
    TilingUseMxuMismatch {
        tiling: GemmTiling,
        use_mxu: bool,
    },
}
