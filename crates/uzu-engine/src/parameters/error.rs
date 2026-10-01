use thiserror::Error;

use super::safetensors_metadata::HeaderLoadingError;
use crate::{backends::common::Backend, data_type::DataType};

#[derive(Debug, Error)]
pub enum ParameterLoaderError<B: Backend> {
    #[error("Header loading error: {0}")]
    HeaderLoadingError(#[from] HeaderLoadingError),
    #[error("Array with key \"{0}\" not found.")]
    KeyNotFound(String),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Failed to read data")]
    ArrayLoadingError(#[from] std::io::Error),
    #[error("Failed to deserialize metadata")]
    MetadataDeserializationError(#[from] serde_json::Error),
    #[error("Invalid tensor: got {shape:?} @ {data_type:?}, expected {expected_shape:?} @ {expected_data_type:?}")]
    InvalidTensor {
        shape: Box<[u32]>,
        data_type: DataType,
        expected_shape: Box<[u32]>,
        expected_data_type: DataType,
    },
    #[error("Invalid tensor byte size: got {size} bytes for {shape:?} @ {data_type:?}, expected {expected_size} bytes")]
    InvalidTensorSize {
        shape: Box<[u32]>,
        data_type: DataType,
        size: usize,
        expected_size: usize,
    },
    #[error("Unvalidated tensors under {prefix:?}: {keys:?}")]
    UnvalidatedTensors {
        prefix: Option<String>,
        keys: Box<[String]>,
    },
}
