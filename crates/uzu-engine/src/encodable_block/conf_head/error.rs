use thiserror::Error;

use crate::{
    backends::Backend,
    encodable_block::{embedding::EmbeddingError, linear::LinearBlockError, normalization::NormalizationNewError},
    safetensors::ParameterLoaderError,
};

#[derive(Debug, Error)]
pub enum ConfidenceHeadError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("config error: {0}")]
    Config(#[from] serde_json::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("confidence head was trained with another model")]
    ModelMismatch,
    #[error("normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("parameter error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("tree mismatch: {0}")]
    TreeMismatch(String),
    #[error("unsupported confidence head: {0}")]
    Unsupported(String),
}
