use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::{
        embedding::EmbeddingError,
        linear::{LinearBlockError, LinearEncodeError},
        normalization::NormalizationNewError,
    },
    safetensors::ParameterLoaderError,
};

#[derive(Debug, Error)]
pub enum ConfidenceHeadError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("parameter error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("linear encode error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("the kernels do not support {0}")]
    UnsupportedShape(String),
}
