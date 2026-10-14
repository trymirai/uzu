use thiserror::Error;

use crate::backends::metal::{error::MetalError, kernel::matmul::MatmulMetalError};

#[derive(Debug, Error)]
pub enum AttentionMetalError {
    #[error("Backend error: {0}")]
    Backend(#[from] MetalError),
    #[error("Matmul error: {0}")]
    Matmul(#[from] MatmulMetalError),
}
