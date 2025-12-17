use crate::{DataType, backends::metal::MTLError};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedEmbeddingError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
}
