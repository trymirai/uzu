use thiserror::Error;

use crate::backends::metal::{error::MetalError, kernel::matmul::MatmulMetalError};

#[derive(Debug, Error)]
pub enum AttentionMetalError {
    #[error("INT8 KV unsupported for this attention config")]
    UnsupportedInt8KV,
    #[error("Backend error: {0}")]
    Backend(#[from] MetalError),
    #[error("Matmul error: {0}")]
    Matmul(#[from] MatmulMetalError),
}
