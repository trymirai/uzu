use derive_more::Debug;
use shoji::types::session::vision::VisionImage;
use thiserror::Error;

use crate::{
    backends::common::{Backend, capture_matcher::CaptureError},
    encodable_block::{decoder::DecoderError, vision::model::VisionEncodeError},
    engine::language_model::{
        LanguageModel,
        grammar::{Grammar, GrammarError},
        state::LanguageModelState,
    },
    speculators::dflash_tfm::DFlashTreeError,
    utils::{timestamps::TimestampSender, trace},
};
pub use crate::{
    encodable_block::sampling::SamplingMethod, engine::language_model::stream::stream::LanguageModelStream,
};

mod stream;

#[derive(Clone, Copy)]
pub enum LanguageModelInput<'a> {
    Image(&'a VisionImage),
    Tokens(&'a [u64]),
}

pub struct LanguageModelStreamOptions {
    pub sampling_method: SamplingMethod,
    pub timestamps: Option<TimestampSender>,
    pub grammar: Option<Grammar>,
}

#[derive(Debug, Error)]
pub enum LanguageModelStreamError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Capture error: {0}")]
    Capture(#[from] CaptureError<B>),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("Grammar error: {0}")]
    Grammar(#[from] GrammarError),
    #[error("Trie accept error: {0}")]
    TrieAccept(#[from] crate::trie::TrieAcceptError),
    #[error("Speculator error: {0}")]
    Speculator(#[from] DFlashTreeError<B>),
    #[error("No seed token (both state and input are empty)")]
    NoSeedToken,
    #[error("Context overflow")]
    ContextOverflow,
    #[error("Vision error: {0}")]
    Vision(#[from] VisionEncodeError<B>),
}

impl<B: Backend> LanguageModel<B> {
    /// Records a trace when `UZU_TRACE` is set, see [`trace::sender_from_env`].
    pub fn default_stream_options(&self) -> LanguageModelStreamOptions {
        LanguageModelStreamOptions {
            sampling_method: self.default_sampling_method(),
            timestamps: trace::sender_from_env(),
            grammar: None,
        }
    }

    pub fn stream<'a>(
        &'a self,
        input: &[LanguageModelInput<'_>],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, LanguageModelStreamError<B>> {
        LanguageModelStream::new(self, input, state, options)
    }
}
