use thiserror::Error;

use crate::data_type::DataType;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("Data type {0:?} has no safetensors equivalent")]
    UnsupportedDataType(DataType),
}
