//! Forward pass mode types for language model generator and classifier models.

use std::{cell::RefCell, rc::Rc};

use super::super::cache_layers::CacheLayers;
#[cfg(feature = "tracing")]
use super::super::traces::ActivationTrace;
use crate::{backends::metal::MetalArray, session::parameter::SamplingMethod};

type ArrayCell = RefCell<MetalArray>;

/// Mode-specific configuration and state.
pub enum ForwardPassMode {
    /// Language model generator (autoregressive generation) mode.
    LanguageModelGenerator(LanguageModelGeneratorModeState),
    /// Classifier (bidirectional) mode.
    Classifier(ClassifierModeState),
}

/// Language model generator specific state.
pub struct LanguageModelGeneratorModeState {
    pub cache_layers: Rc<RefCell<CacheLayers>>,
    pub token_seeds: ArrayCell,
    pub logits: ArrayCell,
    pub sampling_output: Option<ArrayCell>,
    pub sampling_method: Option<SamplingMethod>,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace>>,
    pub active_suffix_length: usize,
    pub is_prefilling: bool,
}

/// Classifier-specific state.
pub struct ClassifierModeState {
    pub pooling: ArrayCell,
    pub dense: ArrayCell,
    pub norm: ArrayCell,
    pub classifier_logits: ArrayCell,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace>>,
}
