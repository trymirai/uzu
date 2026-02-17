use crate::{DataType, backends::metal::MetalError};

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MetalError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
}
