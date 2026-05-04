mod per_layer_embedding;
mod ple_layer;
mod tensor_finalize;
mod value_norm;

pub use per_layer_embedding::PerLayerEmbedding;
pub use ple_layer::PLELayer;
pub use tensor_finalize::TensorFinalize;
use thiserror::Error;
pub use value_norm::ValueNorm;

use crate::{
    DataType,
    backends::common::{Backend, kernel::matmul::MatmulError},
    encodable_block::{RMSNormError, linear::LinearBlockError},
    parameters::ParameterLoaderError,
};

#[derive(Debug, Error)]
pub enum ModelExtensionError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Matmul error: {0}")]
    MatmulError(#[from] MatmulError<B>),
    #[error("Linear error: {0}")]
    LinearError(#[from] LinearBlockError<B>),
    #[error("RMS norm error: {0}")]
    RMSNormError(#[from] RMSNormError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Invalid tensor: got {shape:?} @ {data_type:?}, expected {expected_shape:?} @ {expected_data_type:?}")]
    InvalidTensor {
        shape: Box<[usize]>,
        data_type: DataType,
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
    },
}

pub(super) fn validate_tensor<B: Backend>(
    shape: &[usize],
    data_type: DataType,
    expected_shape: &[usize],
    expected_data_type: DataType,
) -> Result<(), ModelExtensionError<B>> {
    if shape == expected_shape && data_type == expected_data_type {
        Ok(())
    } else {
        Err(ModelExtensionError::InvalidTensor {
            shape: shape.into(),
            data_type,
            expected_shape: expected_shape.into(),
            expected_data_type,
        })
    }
}
