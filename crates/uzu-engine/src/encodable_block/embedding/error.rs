use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::{linear::LinearMatmulError, weight_matrix::WeightMatrixError},
    parameters::ParameterLoaderError,
};

#[derive(Debug, Error)]
pub enum EmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error(transparent)]
    LinearMatmul(#[from] LinearMatmulError<B>),
}
