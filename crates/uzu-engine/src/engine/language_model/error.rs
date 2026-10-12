use std::io;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::Backend, encodable_block::decoder::DecoderError, parameters::ParameterLoaderError,
    speculators::dflash_tfm::DFlashSpeculatorLoadError,
};

#[derive(Debug, Error)]
pub enum EngineLoadLanguageModelError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("Speculator error: {0}")]
    Speculator(#[from] DFlashSpeculatorLoadError<B>),
    #[error("model does not support the loaded speculator")]
    UnsupportedSpeculator,
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}
